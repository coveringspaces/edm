# training/koopman.py
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.func import jvp, vmap
from torch_utils import persistence
from torch_utils import training_stats
from torch.nn.functional import silu

from training.networks import DhariwalUNet
from training.networks import UNetBlock

def trigflow_alpha_s(t):
    alpha = torch.cos(t)
    s = torch.sin(t)
    return alpha, s

def cfm_dxdt_from_net(net, x, t, labels=None, sigma_data: float = 0.5):
    x0_hat = net(x, t, labels)
    alpha = torch.cos(t)
    s = torch.sin(t).clamp(min=1e-6)
    return (alpha * x - x0_hat) / (s * sigma_data)

#----------------------------------------------------------------------------
# DhariwalEncoderOnly steals the mapping + encoder from DhariwalUNet, but does NOT have a decoder.

@persistence.persistent_class
class DhariwalEncoderOnly(nn.Module):
    def __init__(self, **dhariwal_kwargs):
        super().__init__()
        unet = DhariwalUNet(**dhariwal_kwargs)

        # steal mapping + encoder
        self.label_dropout = unet.label_dropout
        self.map_noise = unet.map_noise
        self.map_augment = unet.map_augment
        self.map_layer0 = unet.map_layer0
        self.map_layer1 = unet.map_layer1
        self.map_label = unet.map_label
        self.enc = unet.enc

    def forward(self, x, noise_labels, class_labels, augment_labels=None):
        # --- identical mapping to DhariwalUNet.forward ---
        emb = self.map_noise(noise_labels)
        if self.map_augment is not None and augment_labels is not None:
            emb = emb + self.map_augment(augment_labels)
        emb = silu(self.map_layer0(emb))
        emb = self.map_layer1(emb)
        if self.map_label is not None:
            tmp = class_labels
            if self.training and self.label_dropout:
                tmp = tmp * (torch.rand([x.shape[0], 1], device=x.device) >= self.label_dropout).to(tmp.dtype)
            emb = emb + self.map_label(tmp)
        emb = silu(emb)

        # --- encoder only ---
        for block in self.enc.values():
            x = block(x, emb) if hasattr(block, 'emb_channels') else block(x)

        return x  # (B, Cb, Hb, Wb)

#----------------------------------------------------------------------------
# Koopman-Flow Matching (KFM) network.

@persistence.persistent_class
class KoopmanEigenNet(nn.Module):
    def __init__(self,
        img_resolution,
        img_channels,
        k,
        label_dim=0,
        augment_dim=0,
        use_fp16=False,
        sigma_data=0.5,
        # Dhariwal params:
        model_channels=192,
        channel_mult=[1,2,3,4],
        channel_mult_emb=4,
        num_blocks=3,
        attn_resolutions=[32,16,8],
        dropout=0.00,
        label_dropout=0,
    ):
        super().__init__()
        self.k = k
        self.label_dim = label_dim
        self.use_fp16 = use_fp16
        self.sigma_data = sigma_data

        self.encoder = DhariwalEncoderOnly(
            img_resolution=img_resolution,
            in_channels=img_channels,
            out_channels=img_channels,   # unused, but DhariwalUNet requires it
            label_dim=label_dim,
            augment_dim=augment_dim,
            model_channels=model_channels,
            channel_mult=channel_mult,
            channel_mult_emb=channel_mult_emb,
            num_blocks=num_blocks,
            attn_resolutions=attn_resolutions,
            dropout=dropout,
            label_dropout=label_dropout,
        )

        # bottleneck channels for Dhariwal encoder end:
        bottleneck_ch = model_channels * channel_mult[-1]

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(
            nn.Linear(bottleneck_ch, bottleneck_ch),
            nn.SiLU(),
            nn.Linear(bottleneck_ch, 2*k),
        )
        # Override default (Kaiming) init on the output projection so ψ starts at O(1) scale.
        # Default std ≈ sqrt(2/bottleneck_ch) ≈ 0.05 — too small for a loss that is
        # quadratic/quartic in ψ, causing gradients to vanish near ψ=0.
        nn.init.normal_(self.head[-1].weight, std=1.0)
        nn.init.zeros_(self.head[-1].bias)

    def forward(self, x, t, class_labels=None, force_fp32=False, augment_labels=None):
        x = x.to(torch.float32)
        t = t.to(torch.float32).reshape(-1, 1, 1, 1)

        class_labels = None if self.label_dim == 0 else (
            torch.zeros([1, self.label_dim], device=x.device)
            if class_labels is None else class_labels.to(torch.float32).reshape(-1, self.label_dim)
        )

        dtype = torch.float16 if (self.use_fp16 and not force_fp32 and x.device.type == 'cuda') else torch.float32

        c_in = 1 / self.sigma_data
        noise_labels = t.flatten()  # Dhariwal uses (B,)

        feat = self.encoder((c_in * x).to(dtype), noise_labels, class_labels, augment_labels=augment_labels)
        feat = feat.to(torch.float32)
        feat = self.pool(feat).flatten(1)   # (B, bottleneck_ch)
        psi_hat = self.head(feat)           # (B, 2k)

        return psi_hat

def _to_complex_vec(psi_hat, k: int):
    # psi_hat: (B,2k) -> complex (B,k)
    if torch.is_complex(psi_hat):
        return psi_hat
    B, C = psi_hat.shape
    assert C == 2*k
    psi_hat = psi_hat.view(B, k, 2)
    return psi_hat[..., 0] + 1j * psi_hat[..., 1]

class KoopmanPhases(nn.Module):
    """
    Train phases φ_i (real). These define unit-modulus eigenvalues e^{i φ_i}.
    """
    def __init__(self, k: int, init_zero: bool = True):
        super().__init__()
        phi0 = torch.zeros(k) if init_zero else 2 * torch.pi * torch.rand(k)
        self.phi = nn.Parameter(phi0)

    def forward(self, _dummy):
        # returns (cos φ, sin φ) each shape (k,)
        return torch.cos(self.phi), torch.sin(self.phi)


def complex_mul_reim(lam_re, lam_im, psi_re, psi_im):
    # (lam_re + i lam_im) * (psi_re + i psi_im)
    out_re = lam_re * psi_re - lam_im * psi_im
    out_im = lam_re * psi_im + lam_im * psi_re
    return out_re, out_im

def split_reim(psi_hat: torch.Tensor):
    """
    psi_hat: (B,2k) -> (psi_re, psi_im) each (B,k)
    """
    B, two_k = psi_hat.shape
    k = two_k // 2
    return psi_hat[:, :k], psi_hat[:, k:]

def complex_inner_batch(a_re, a_im, b_re, b_im):
    """
    a,b: (B,k)
    returns inner products per component: (k,) complex as (re,im)
      <a_i, b_i> over batch.
    """
    # conj(a) * b = (a_re - i a_im)(b_re + i b_im)
    prod_re = a_re * b_re + a_im * b_im
    prod_im = -a_im * b_re + a_re * b_im
    return prod_re.mean(dim=0), prod_im.mean(dim=0)

def complex_gram(psi_re, psi_im):
    """
    psi: (B,k) -> G: (k,k) complex (re,im) where G_ij = <psi_i, psi_j>.
    """
    B = psi_re.shape[0]
    # conj(psi)^T psi
    # real: psi_re^T psi_re + psi_im^T psi_im
    G_re = (psi_re.T @ psi_re + psi_im.T @ psi_im) / B
    # imag: -psi_im^T psi_re + psi_re^T psi_im
    G_im = (-psi_im.T @ psi_re + psi_re.T @ psi_im) / B
    return G_re, G_im

class NeuralSVDFunction(torch.autograd.Function):
    """
    Custom autograd function mirroring the official Neural SVD backward.

    Inputs
    ------
    psi_A_re, psi_A_im   : (B/2, k)  active batch — receives gradient
    Lpsi_A_re, Lpsi_A_im : (B/2, k)  Koopman operator on ψ_A (from JVP)
    psi_B_re, psi_B_im   : (B/2, k)  target batch — functionally detached
                                       via None returns
    cos_phi, sin_phi      : (k,)      learned eigenvalue phases
    operator_scale        : float     alignment-term normalisation

    Output
    ------
    scalar loss = (term1 + term2) / k
    """

    @staticmethod
    def forward(ctx, psi_A_re, psi_A_im, Lpsi_A_re, Lpsi_A_im,
                psi_B_re, psi_B_im, cos_phi, sin_phi, operator_scale, metric_weight):
        B_half, k = psi_A_re.shape
        dev = psi_A_re.device

        # ── Self-Gram of active batch A
        G_A_re = (psi_A_re.T @ psi_A_re + psi_A_im.T @ psi_A_im) / B_half   # (k,k)
        G_A_im = (-psi_A_im.T @ psi_A_re + psi_A_re.T @ psi_A_im) / B_half  # (k,k)

        # ── Self-Gram of target batch B  (treated as a fixed scalar matrix in bwd)
        G_B_re = (psi_B_re.T @ psi_B_re + psi_B_im.T @ psi_B_im) / B_half   # (k,k)
        G_B_im = (-psi_B_im.T @ psi_B_re + psi_B_re.T @ psi_B_im) / B_half  # (k,k)

        # ── Sequential nesting mask × phase weight
        # matrix_mask[l,m] = triu[l,m] · cos(φ_l−φ_m)  — used in bwd einsum
        triu        = torch.triu(torch.ones(k, k, device=dev))
        cos_dphi    = (cos_phi[:, None] * cos_phi[None, :]
                       + sin_phi[:, None] * sin_phi[None, :])                 # (k,k)
        matrix_mask = triu * cos_dphi                                         # (k,k)

        # ── Term 2: metric_weight · Σ_{l≤m} cos(φ_l−φ_m) · Re(conj(G_B[l,m]) · G_A[l,m])
        # D[l,m] = G_B_re[l,m]·G_A_re[l,m] + G_B_im[l,m]·G_A_im[l,m]
        D     = G_B_re * G_A_re + G_B_im * G_A_im                             # (k,k)
        triu_D = triu * D                                                      # (k,k) for phase bwd
        term2 = metric_weight * (matrix_mask * D).sum()

        # ── Per-eigenfunction inner products ⟨ψ_A[:,m], Lψ_A[:,m]⟩
        ip_re = (psi_A_re * Lpsi_A_re + psi_A_im * Lpsi_A_im).sum(0) / B_half  # (k,)
        ip_im = (-psi_A_im * Lpsi_A_re + psi_A_re * Lpsi_A_im).sum(0) / B_half # (k,)

        # ── Term 1: −2/scale · Σ_m Re(e^{iφ_m} · ⟨ψ_A_m, Lψ_A_m⟩)
        term1 = -2.0 * (cos_phi * ip_re - sin_phi * ip_im).sum() / operator_scale

        loss = (term1 + term2) / k

        ctx.save_for_backward(
            psi_A_re, psi_A_im, Lpsi_A_re, Lpsi_A_im,
            cos_phi, sin_phi,
            G_B_re, G_B_im, matrix_mask,
            triu_D,
            ip_re, ip_im,
        )
        ctx.operator_scale = operator_scale
        ctx.metric_weight = metric_weight
        ctx.B_half = B_half
        ctx.k = k
        return loss

    @staticmethod
    def backward(ctx, grad_output):
        (psi_A_re, psi_A_im, Lpsi_A_re, Lpsi_A_im,
         cos_phi, sin_phi,
         G_B_re, G_B_im, matrix_mask,
         triu_D,
         ip_re, ip_im) = ctx.saved_tensors
        scale         = ctx.operator_scale
        metric_weight = ctx.metric_weight
        B_half = ctx.B_half
        k      = ctx.k
        g = grad_output / k

        fac2 = g * 2.0 * metric_weight / B_half
        grad_psi_A_re = fac2 * (
            torch.einsum('lm,lm,bl->bm', matrix_mask, G_B_re, psi_A_re)
          - torch.einsum('lm,lm,bl->bm', matrix_mask, G_B_im, psi_A_im)
        )
        grad_psi_A_im = fac2 * (
            torch.einsum('lm,lm,bl->bm', matrix_mask, G_B_re, psi_A_im)
          + torch.einsum('lm,lm,bl->bm', matrix_mask, G_B_im, psi_A_re)
        )

        # ── Operator gradient: Term 1
        # ∂term1/∂ψ_A_re[b,m] = (−2/(scale·B)) · (cos_φ_m·Lψ_A_re[b,m] − sin_φ_m·Lψ_A_im[b,m])
        fac1 = g * (-2.0) / (scale * B_half)
        grad_psi_A_re = grad_psi_A_re + fac1 * (cos_phi * Lpsi_A_re - sin_phi * Lpsi_A_im)
        grad_psi_A_im = grad_psi_A_im + fac1 * (cos_phi * Lpsi_A_im + sin_phi * Lpsi_A_re)

        grad_Lpsi_A_re = fac1 * (cos_phi * psi_A_re + sin_phi * psi_A_im)
        grad_Lpsi_A_im = fac1 * (cos_phi * psi_A_im - sin_phi * psi_A_re)

        # ── Phase gradients from Term 1
        # ∂term1/∂cos_φ_m = −2/scale · ip_re[m]
        grad_cos_phi = g * (-2.0 / scale) * ip_re
        grad_sin_phi = g * (2.0 / scale) * ip_im

        # ── Phase gradients from Term 2
        # term2 = metric_weight · Σ_{l≤m} cos_dphi[l,m] · D[l,m]
        # cos_dphi[l,m] = cos_φ_l·cos_φ_m + sin_φ_l·sin_φ_m
        # ∂term2/∂cos_φ_m:
        #   column (j=m, l≤m): Σ_{l≤m} cos_φ_l·D[l,m]  = (triu_D).T[:,m] · cos_phi
        #   row    (l=m, j≥m): Σ_{j≥m} cos_φ_j·D[m,j]  = (triu_D)[m,:]   · cos_phi
        # Diagonal (l=m=j) appears once in each sum → net 2·cos_φ_m·D[m,m] 
        grad_cos_phi = grad_cos_phi + g * metric_weight * (triu_D.T @ cos_phi + triu_D @ cos_phi)
        grad_sin_phi = grad_sin_phi + g * metric_weight * (triu_D.T @ sin_phi + triu_D @ sin_phi)

        return (
            grad_psi_A_re,   # psi_A_re — active, receives gradient
            grad_psi_A_im,   # psi_A_im — active, receives gradient
            grad_Lpsi_A_re,  # Lpsi_A_re — flows back through JVP
            grad_Lpsi_A_im,  # Lpsi_A_im — flows back through JVP
            None,            # psi_B_re — functionally detached 
            None,            # psi_B_im — functionally detached
            grad_cos_phi,    # cos_phi — learned phase
            grad_sin_phi,    # sin_phi — learned phase
            None,            # operator_scale 
            None,            # metric_weight 
        )


#----------------------------------------------------------------------------
# Loss from equation (4) of Jon's writeup

@persistence.persistent_class
class KoopmanLoss:
    def __init__(self,
                 P_mean=-1.2, P_std=1.2, sigma_data=0.5,
                 sigma_min=1e-3, sigma_max=80, t_epsilon=1e-4,
                 operator_scale=100.0, metric_weight=1.0):
        self.P_mean = P_mean
        self.P_std = P_std
        self.sigma_data = sigma_data
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.t_epsilon = t_epsilon
        self.operator_scale = operator_scale
        self.metric_weight = metric_weight

        # torch.func.jvp exists in torch>=2.0
        try:
            from torch.func import jvp as _jvp
            self._jvp = _jvp
        except Exception:
            self._jvp = None

    def __call__(self, psi_net, phase_net, cfm_net, images, labels=None, augment_pipe=None):
        assert self._jvp is not None, "Need torch.func.jvp (PyTorch 2.x)."

        # ------------------------------------------------------------
        # (A) Sample (x,t) on the TrigFlow curve 
        # ------------------------------------------------------------
        y, augment_labels = augment_pipe(images) if augment_pipe is not None else (images, None)

        rnd_normal = torch.randn([y.shape[0], 1, 1, 1], device=y.device)
        sigma = (rnd_normal * self.P_std + self.P_mean).exp().clamp(self.sigma_min, self.sigma_max)

        t = torch.atan(sigma / self.sigma_data)
        t = t.clamp(min=self.t_epsilon, max=0.5 * torch.pi - self.t_epsilon)

        alpha = torch.cos(t)
        s = torch.sin(t)
        eps = torch.randn_like(y)
        x = alpha * y + s * self.sigma_data * eps

        # ------------------------------------------------------------
        # (B) Vector field for time-augmented dynamics: (xdot, 1)
        # ------------------------------------------------------------
        with torch.no_grad():
            xdot = cfm_dxdt_from_net(cfm_net, x, t, labels=labels, sigma_data=self.sigma_data)

        tdot = torch.ones_like(t)  # dt/ds = 1 in the time-augmented system

        # ------------------------------------------------------------
        # (C) One JVP: (psi, Lpsi) where Lpsi = ∂t psi + ∇x psi · xdot
        # ------------------------------------------------------------
        raw_psi_net = psi_net
        
        def f(x_in, t_in):
            # labels & augment_labels treated as constants (no grads through them)
            out = raw_psi_net(x_in, t_in, class_labels=labels, augment_labels=augment_labels, force_fp32=True)  # (B,2k)
            return out

        psi_hat, Lpsi_hat = self._jvp(f, (x, t), (xdot, tdot))  # both (B,2k)

        # ------------------------------------------------------------
        # (D) Convert to complex (re/im) blocks: psi = psi_re + i psi_im
        # ------------------------------------------------------------
        psi_re, psi_im     = split_reim(psi_hat)     # (B,k), (B,k)
        Lpsi_re, Lpsi_im   = split_reim(Lpsi_hat)    # (B,k), (B,k)

        B, k = psi_re.shape

        training_stats.report('Loss/psi_rms', (psi_re**2 + psi_im**2).mean().sqrt())

        # phase_net ignores t, the arg is a placeholder
        cos_phi, sin_phi = phase_net(t)  # (k,), (k,)

        # ------------------------------------------------------------
        # Sequential nesting via NeuralSVDFunction
        #
        # Split-batch cross-Gram estimator:
        #   ψ_A (first half, active)  — receives gradients
        #   ψ_B (second half, detached) — provides the target metric
        # ------------------------------------------------------------

        half = B // 2
        psi_A_re,  psi_A_im  = psi_re[:half],  psi_im[:half]   # (B/2,k) active
        psi_B_re,  psi_B_im  = psi_re[half:],  psi_im[half:]   # (B/2,k) target (functionally detached)
        Lpsi_A_re, Lpsi_A_im = Lpsi_re[:half], Lpsi_im[:half]  # (B/2,k) active

        loss_svd = NeuralSVDFunction.apply(
            psi_A_re, psi_A_im, Lpsi_A_re, Lpsi_A_im,
            psi_B_re, psi_B_im,
            cos_phi, sin_phi,
            self.operator_scale,
            self.metric_weight,
        )

        # Recover per-term values for stats — computed once without grad for logging only.
        with torch.no_grad():
            G_A_re_log = (psi_A_re.T @ psi_A_re + psi_A_im.T @ psi_A_im) / half
            G_A_im_log = (-psi_A_im.T @ psi_A_re + psi_A_re.T @ psi_A_im) / half
            G_B_re_log = (psi_B_re.T @ psi_B_re + psi_B_im.T @ psi_B_im) / half
            G_B_im_log = (-psi_B_im.T @ psi_B_re + psi_B_re.T @ psi_B_im) / half
            triu_log     = torch.triu(torch.ones(k, k, device=psi_re.device))
            cos_dphi_log = cos_phi[:, None] * cos_phi[None, :] + sin_phi[:, None] * sin_phi[None, :]
            D_log        = G_B_re_log * G_A_re_log + G_B_im_log * G_A_im_log
            term2_log    = self.metric_weight * (triu_log * cos_dphi_log * D_log).sum()
            ip_re_log    = (psi_A_re * Lpsi_A_re + psi_A_im * Lpsi_A_im).sum(0) / half
            ip_im_log    = (-psi_A_im * Lpsi_A_re + psi_A_re * Lpsi_A_im).sum(0) / half
            term1_log    = -2.0 * (cos_phi * ip_re_log - sin_phi * ip_im_log).sum() / self.operator_scale

        training_stats.report('Loss/term1', term1_log / k)
        training_stats.report('Loss/term2', term2_log / k)

        # ------------------------------------------------------------
        # (G) Eigenvalue magnitude stats  (use full non-detached psi)
        # λ_i = e^{iφ_i} * ||ψ_i||²  →  |λ_i| = ||ψ_i||²
        # ------------------------------------------------------------
        psi_sq      = (psi_re**2 + psi_im**2).mean(dim=0)   # (k,)
        lam_re_vals = cos_phi * psi_sq
        lam_im_vals = sin_phi * psi_sq
        lam_mag     = psi_sq

        training_stats.report('Eigenvalues/lam_mag_min',  lam_mag.min())
        training_stats.report('Eigenvalues/lam_mag_max',  lam_mag.max())
        training_stats.report('Eigenvalues/lam_mag_mean', lam_mag.mean())
        training_stats.report('Eigenvalues/lam_re_max',   lam_re_vals.max())
        training_stats.report('Eigenvalues/lam_im_max',   lam_im_vals.max())

        for i in range(k):
            training_stats.report(f'Eigenvalues/lam_mag_{i:02d}', lam_mag[i])

        # Store for histogram logging in the training loop.
        self._last_lam_mag = lam_mag.detach()

        return loss_svd






