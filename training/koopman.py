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

def cfm_dxdt_from_net(net, x, t, labels=None, augment_labels=None, sigma_data: float = 0.5):
    x0_hat = net(x, t, labels, augment_labels=augment_labels)
    alpha = torch.cos(t)
    s = torch.sin(t).clamp(min=1e-6)
    return (alpha * x - x0_hat) / s

# ─────────────────────────────────────────────────────────────────────────────
# QPSK toy-distribution helpers
# ─────────────────────────────────────────────────────────────────────────────

def sample_qpsk_toy(B, m=1.0, device='cpu'):
    """(B, 2) clean samples from the 4-point QPSK distribution {±m}².
    Each coordinate is i.i.d. uniform over {-m, +m}.
    """
    signs = torch.randint(0, 2, (B, 2), device=device).to(torch.float32) * 2.0 - 1.0
    return signs * m  # (B, 2)


def qpsk_closed_form_dxdt(x, t, sigma_data=0.5, m=1.0):
    """Analytic velocity field for QPSK on the TrigFlow curve.

    Derivation: for each coordinate i independently, the prior is
      P(y_i = +m) = P(y_i = -m) = 1/2
    and the noisy observation is x_i = alpha*y_i + s*sigma_data*eps_i.
    The posterior mean works out to:
      E[y_i | x_i] = m * tanh(alpha * m * x_i / (s^2 * sigma_data^2))
    The TrigFlow velocity is then (alpha*x - E[y|x]) / s,
    matching cfm_dxdt_from_net with x0_hat = E[y|x].

    x:           (B, 2)
    t:           (B, 1)  — t in [0, pi/2]
    returns:     (B, 2)
    """
    alpha = torch.cos(t)                         # (B, 1)
    s     = torch.sin(t).clamp(min=1e-6)         # (B, 1)
    posterior_mean = m * torch.tanh(alpha * m * x / (s**2 * sigma_data**2))
    return (alpha * x - posterior_mean) / s


# ─────────────────────────────────────────────────────────────────────────────
# Toy MLP psi-network for 2D inputs
# ─────────────────────────────────────────────────────────────────────────────

@persistence.persistent_class
class KoopmanToyMLP(nn.Module):
    """Lightweight MLP psi-network for 2D toy experiments.

    Input:  x ∈ R² concatenated with t ∈ R  →  (B, 3)
    Output: (B, 2k) real values representing k complex eigenfunctions,
            packed as [re_1…re_k | im_1…im_k] to match split_reim().
    """
    def __init__(self, k, hidden_dim=256, n_hidden=3,
                 label_dim=0, augment_dim=0, use_fp16=False, sigma_data=0.5,
                 img_resolution=None, img_channels=None, **unused_kwargs):
        super().__init__()
        self.k         = k
        self.label_dim = label_dim   # kept for API compat with training loop; toy is unconditional

        layers = [nn.Linear(3, hidden_dim), nn.SiLU()]
        for _ in range(n_hidden - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.SiLU()]
        layers.append(nn.Linear(hidden_dim, 2 * k))
        self.net = nn.Sequential(*layers)

    def forward(self, x, t, class_labels=None, force_fp32=False, augment_labels=None):
        """x: (B, 2),  t: broadcastable to (B, 1).  Returns (B, 2k)."""
        x = x.to(torch.float32)
        t = t.to(torch.float32).reshape(x.shape[0], 1)   # ensure (B, 1)
        return self.net(torch.cat([x, t], dim=-1))


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
        nn.init.kaiming_uniform_(self.head[-1].weight, a=math.sqrt(5))
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
        feat = self.pool(feat).flatten(1)              # (B, bottleneck_ch)
        psi_hat = self.head(feat)                      # (B, 2k)

        return psi_hat

class KoopmanPhases(nn.Module):
    """
    Train phases φ_i (real). These define unit-modulus eigenvalues e^{i φ_i}.
    If label_dim > 0, one phase vector per class (lookup by one-hot label).
    """
    def __init__(self, k: int, label_dim: int = 0, init_zero: bool = False):
        super().__init__()
        self.k = k
        self.label_dim = label_dim
        phi0 = torch.zeros(k) if init_zero else 2 * torch.pi * torch.arange(k) / k
        n = max(label_dim, 1)
        self.phi = nn.Parameter(phi0.unsqueeze(0).expand(n, -1).clone())  # (n, k)

    def forward(self, labels=None):
        if self.label_dim > 0 and labels is not None:
            class_idx = labels.argmax(dim=-1)   # (B,)
            return self.phi[class_idx]           # (B, k)
        return self.phi[0]                       # (k,)


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

def get_sequential_nesting_masks(L: int, device=None):
    """
    Nesting masks for sequential nesting (Neural SVD / NestedLoRA paper §3).

    Paper definitions (sequential):
        m_ell     = 1                      scalar weight per level ell — trivially 1
        M_{i,ell} = 1[i <= ell]            matrix mask — upper triangular in (row=i, col=ell)

    Indexing convention used throughout this codebase:
        row = i    (eigenfunction index, 0-based)
        col = ell  (nesting level,       0-based)

    torch.triu(ones(L,L))[i,ell] = 1 iff i <= ell, which is exactly M_{i,ell}. ✓

    Returns
    -------
    vector_mask : (L,)   m_ell = 1 for all ell
    matrix_mask : (L,L)  M_{i,ell} = 1[i <= ell], upper triangular (row=i, col=ell)
    """
    # m_ell = 1  →  all-ones vector, one entry per nesting level ell
    vector_mask = torch.ones(L, device=device)
    # M_{i,ell} = 1[i <= ell]  →  upper triangle: triu[i,ell] = 1 where row i <= col ell
    matrix_mask = torch.triu(torch.ones(L, L, device=device))
    return vector_mask, matrix_mask

def split_uv(x):
    k = x.shape[1] // 2
    return x[:, :k], x[:, k:]


def compute_complex_lambda_from_stacked(psi_half):
    u, v = split_uv(psi_half)
    B = psi_half.shape[0]
    lam_re = (u.T @ u + v.T @ v) / B
    lam_im = (u.T @ v - v.T @ u) / B
    return lam_re, lam_im


class ComplexNestedMetricOpLoss(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        psi,           # (B, 2k)      full batch, stacked [u, v]
        Lpsi,          # (B, 2k)      full batch, stacked [Lu, Lv]
        psi_1,         # (B_half, 2k) first half batch, stacked [u^(1), v^(1)]
        psi_2,         # (B_half, 2k) second half batch, stacked [u^(2), v^(2)]
        phi,           # (B, k)       per-sample phases (same row for same class)
        vector_mask,   # (k,)
        matrix_mask,   # (k, k)
        operator_scale, # float: scale factor for operator term
    ):
        device = psi.device
        vector_mask = vector_mask.to(device)
        matrix_mask = matrix_mask.to(device)

        B = psi.shape[0]
        B_half = psi_1.shape[0]
        k = psi.shape[1] // 2

        u, v = split_uv(psi)
        Lu, Lv = split_uv(Lpsi)

        # Gram matrices from the two half batches
        lam_1_re, lam_1_im = compute_complex_lambda_from_stacked(psi_1)
        lam_2_re, lam_2_im = compute_complex_lambda_from_stacked(psi_2)

        # phase factors: phi is (B, k)
        cos_phi = torch.cos(phi)   # (B, k)
        sin_phi = torch.sin(phi)   # (B, k)
        # batch-average cos(phi_i - phi_j) for the metric term
        cos_dphi = (
            cos_phi[:, :, None] * cos_phi[:, None, :]
            + sin_phi[:, :, None] * sin_phi[:, None, :]
        ).mean(dim=0)  # (k, k)

        # ----- operator term -----
        # per-sample: Re(e^{+i phi_{b,i}} * conj(psi_{b,i}) * Lpsi_{b,i})
        # = cos(phi)*Re(conj(psi)*Lpsi) - sin(phi)*Im(conj(psi)*Lpsi)
        contrib_re = u * Lu + v * Lv   # (B, k)  = Re(conj(psi)*Lpsi)
        contrib_im = u * Lv - v * Lu   # (B, k)  = Im(<psi, Lpsi>)

        loss_operator = -2.0 * (cos_phi * contrib_re - sin_phi * contrib_im).sum() / (B * operator_scale)

        # ----- metric term -----
        D = lam_2_re * lam_1_re + lam_2_im * lam_1_im
        loss_metric = (cos_dphi * D).sum()

        loss = loss_operator + loss_metric

        ctx.save_for_backward(
            psi, Lpsi, psi_1, psi_2,
            phi, vector_mask, matrix_mask,
            lam_1_re, lam_1_im,
            lam_2_re, lam_2_im,
        )
        ctx.B = B
        ctx.B_half = B_half
        ctx.B_half_2 = psi_2.shape[0]
        ctx.k = k
        ctx.operator_scale = operator_scale
        return loss
    
    @staticmethod
    def backward(ctx, grad_output):
        (
            psi, Lpsi, psi_1, psi_2,
            phi, vector_mask, matrix_mask,
            lam_1_re, lam_1_im,
            lam_2_re, lam_2_im,
        ) = ctx.saved_tensors

        B = ctx.B
        B_half = ctx.B_half
        B_half_2 = ctx.B_half_2
        operator_scale = ctx.operator_scale
        g = grad_output

        u, v = split_uv(psi)
        Lu, Lv = split_uv(Lpsi)

        u1, v1 = split_uv(psi_1)
        u2, v2 = split_uv(psi_2)

        # phi is (B, k)
        cos_phi = torch.cos(phi)   # (B, k)
        sin_phi = torch.sin(phi)   # (B, k)
        cos_dphi = (
            cos_phi[:, :, None] * cos_phi[:, None, :]
            + sin_phi[:, :, None] * sin_phi[:, None, :]
        ).mean(dim=0)  # (k, k)

        # -------------------------------------------------
        # operator gradient wrt psi
        # -------------------------------------------------
        fac_op = g * (-2.0 / (B * operator_scale))

        grad_u = fac_op * (
            vector_mask[None, :] * (cos_phi * Lu - sin_phi * Lv)
        )
        grad_v = fac_op * (
            vector_mask[None, :] * (cos_phi * Lv + sin_phi * Lu)
        )

        grad_psi = torch.cat([grad_u, grad_v], dim=1)

        # -------------------------------------------------
        # metric gradient on half batches
        # -------------------------------------------------
        fac_met_1 = g * (2.0 / B_half)
        fac_met_2 = g * (2.0 / B_half_2)
        metric_mask = matrix_mask * cos_dphi

        grad_u1 = fac_met_1 * (
            torch.einsum('lm,lm,bl->bm', metric_mask, lam_2_re, u1)
            - torch.einsum('lm,lm,bl->bm', metric_mask, lam_2_im, v1)
        )
        grad_v1 = fac_met_1 * (
            torch.einsum('lm,lm,bl->bm', metric_mask, lam_2_re, v1)
            + torch.einsum('lm,lm,bl->bm', metric_mask, lam_2_im, u1)
        )

        grad_u2 = fac_met_2 * (
            torch.einsum('lm,lm,bl->bm', metric_mask, lam_1_re, u2)
            - torch.einsum('lm,lm,bl->bm', metric_mask, lam_1_im, v2)
        )
        grad_v2 = fac_met_2 * (
            torch.einsum('lm,lm,bl->bm', metric_mask, lam_1_re, v2)
            + torch.einsum('lm,lm,bl->bm', metric_mask, lam_1_im, u2)
        )

        grad_psi_1 = torch.cat([grad_u1, grad_v1], dim=1)
        grad_psi_2 = torch.cat([grad_u2, grad_v2], dim=1)

        # -------------------------------------------------
        # phi gradient (B, k)
        # -------------------------------------------------
        # operator contribution: d/d(phi_{b,i}) of per-sample operator term
        contrib_re = u * Lu + v * Lv   # (B, k)
        contrib_im = u * Lv - v * Lu   # (B, k)

        grad_phi_op = g * (2.0 / (B * operator_scale)) * vector_mask[None, :] * (
            sin_phi * contrib_re + cos_phi * contrib_im
        )  # (B, k)

        # metric contribution: d/d(phi_{b,i}) of batch-averaged cos_dphi
        # = -(1/B) * [sum_m base_D[i,m]*sin(phi_{b,i}-phi_{b,m}) + sum_l base_D[l,i]*sin(phi_{b,i}-phi_{b,l})]
        sin_dphi_b = (
            sin_phi[:, :, None] * cos_phi[:, None, :]
            - cos_phi[:, :, None] * sin_phi[:, None, :]
        )  # (B, k, k),  sin_dphi_b[b,i,j] = sin(phi_{b,i} - phi_{b,j})

        D_full = lam_2_re * lam_1_re + lam_2_im * lam_1_im   # (k, k)
        grad_phi_metric = -(g * 2.0 / B) * ((matrix_mask * D_full)[None, :, :] * sin_dphi_b).sum(dim=2)

        grad_phi = grad_phi_op + grad_phi_metric  # (B, k)

        return (
            grad_psi,    # psi
            None,        # Lpsi
            grad_psi_1,  # psi_1
            grad_psi_2,  # psi_2
            grad_phi,    # phi
            None,        # vector_mask
            None,        # matrix_mask
            None,        # operator_scale
        )

#----------------------------------------------------------------------------
# Loss from equation (4) of Jon's writeup

@persistence.persistent_class
class KoopmanLoss:
    def __init__(self,
                 P_mean=-1.2, P_std=1.2, sigma_data=0.5,
                 sigma_min=1e-3, sigma_max=80, t_epsilon=1e-4,
                 operator_scale=100.0,
                 toy_mode=False, toy_m=1.0):
        self.P_mean = P_mean
        self.P_std = P_std
        self.sigma_data = sigma_data
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.t_epsilon = t_epsilon
        self.operator_scale = operator_scale
        self.toy_mode = toy_mode
        self.toy_m = toy_m

        # torch.func.jvp exists in torch>=2.0
        try:
            from torch.func import jvp as _jvp
            self._jvp = _jvp
        except Exception:
            self._jvp = None

    def __call__(self, psi_net, phase_net, cfm_net, images, labels=None, augment_pipe=None):
        assert self._jvp is not None, "Need torch.func.jvp (PyTorch 2.x)."

        if self.toy_mode:
            # ----------------------------------------------------------------
            # (A+B) QPSK toy path: analytic data + analytic vector field
            # ----------------------------------------------------------------
            B      = images.shape[0]
            device = images.device
            y      = sample_qpsk_toy(B, m=self.toy_m, device=device)        # (B, 2)

            rnd_normal = torch.randn([B, 1], device=device)
            sigma = (rnd_normal * self.P_std + self.P_mean).exp().clamp(self.sigma_min, self.sigma_max)
            t     = torch.atan(sigma / self.sigma_data)
            t     = t.clamp(min=self.t_epsilon, max=0.5 * torch.pi - self.t_epsilon)
            alpha = torch.cos(t)
            s     = torch.sin(t)
            eps   = torch.randn_like(y)
            x     = alpha * y + s * self.sigma_data * eps                    # (B, 2)

            xdot  = qpsk_closed_form_dxdt(x, t, sigma_data=self.sigma_data, m=self.toy_m)
            tdot  = torch.ones_like(t)
            labels         = None
            augment_labels = None
        else:
            # ----------------------------------------------------------------
            # (A) Sample (x,t) on the TrigFlow curve
            # ----------------------------------------------------------------
            y, augment_labels = augment_pipe(images) if augment_pipe is not None else (images, None)

            rnd_normal = torch.randn([y.shape[0], 1, 1, 1], device=y.device)
            sigma = (rnd_normal * self.P_std + self.P_mean).exp().clamp(self.sigma_min, self.sigma_max)
            t     = torch.atan(sigma / self.sigma_data)
            t     = t.clamp(min=self.t_epsilon, max=0.5 * torch.pi - self.t_epsilon)
            alpha = torch.cos(t)
            s     = torch.sin(t)
            eps   = torch.randn_like(y)
            x     = alpha * y + s * self.sigma_data * eps

            # ----------------------------------------------------------------
            # (B) Vector field for time-augmented dynamics: (xdot, 1)
            # ----------------------------------------------------------------
            with torch.no_grad():
                xdot = cfm_dxdt_from_net(cfm_net, x, t, labels=labels, augment_labels=augment_labels, sigma_data=self.sigma_data)

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

        training_stats.report('Loss/psi_rms',  (psi_re**2  + psi_im**2).mean().sqrt())
        training_stats.report('Loss/Lpsi_rms', (Lpsi_re**2 + Lpsi_im**2).mean().sqrt())

        phi = phase_net(labels)          # (B, k) or (k,) when unconditional
        if phi.dim() == 1:
            phi = phi.unsqueeze(0).expand(B, -1)   # (1, k) -> (B, k), grads sum over batch
        cos_phi = torch.cos(phi)         # (B, k)
        sin_phi = torch.sin(phi)         # (B, k)

        # ------------------------------------------------------------
        # Operator term: full batch  (paper: use all of f, Tf)
        # Metric term:   split halves for independent Gram estimates
        #   ψ_A (first half)  — Gram estimate 1  (f1 in NeuralSVD)
        #   ψ_B (second half) — Gram estimate 2  (f2 in NeuralSVD)
        # ------------------------------------------------------------

        half = B // 2

        # Full batch, stacked as [u, v] = [real, imag]
        psi  = torch.cat([psi_re,  psi_im],  dim=1)   # (B, 2k)
        Lpsi = torch.cat([Lpsi_re, Lpsi_im], dim=1)   # (B, 2k)

        # Half batches for metric term, also stacked as [u, v]
        psi_1 = torch.cat([psi_re[:half], psi_im[:half]], dim=1)   # (B/2, 2k)
        psi_2 = torch.cat([psi_re[half:], psi_im[half:]], dim=1)   # (B/2, 2k)

        # For logging only, if you still want them outside
        ip_re = (psi_re * Lpsi_re + psi_im * Lpsi_im).sum(0) / B
        ip_im = (-psi_im * Lpsi_re + psi_re * Lpsi_im).sum(0) / B

        # Sequential nesting masks
        vector_mask, matrix_mask = get_sequential_nesting_masks(k, device=psi_re.device)

        loss_svd = ComplexNestedMetricOpLoss.apply(
            psi,                    # (B, 2k)
            Lpsi,                   # (B, 2k)
            psi_1,                  # (B/2, 2k)
            psi_2,                  # (B/2, 2k)
            phi,                    # (B, k)
            vector_mask,            # (k,)
            matrix_mask,            # (k, k)
            self.operator_scale,    # float
        )

        # Logging: detached estimates (operator from full batch, metric = total - operator).
        ip_re_log = ip_re.detach()
        ip_im_log = ip_im.detach()
        with torch.no_grad():
            # operator term uses per-sample phi (B,k) and per-sample contributions (B,k)
            contrib_re_log = psi_re.detach() * Lpsi_re.detach() + psi_im.detach() * Lpsi_im.detach()
            contrib_im_log = psi_re.detach() * Lpsi_im.detach() - psi_im.detach() * Lpsi_re.detach()
            loss_operator_log = -2.0 * (
                cos_phi.detach() * contrib_re_log - sin_phi.detach() * contrib_im_log
            ).sum() / (B * self.operator_scale)
            loss_metric_log   = loss_svd.detach() - loss_operator_log

        # Decompose Lpsi into time and spatial parts — the only genuinely new computation.
        with torch.no_grad():
            _, dtpsi_hat_log  = self._jvp(f, (x, t), (torch.zeros_like(x), tdot))
            dxpsi_hat_log     = Lpsi_hat - dtpsi_hat_log               # Lpsi = dtpsi + dxpsi
            dtpsi_re_log, dtpsi_im_log = split_reim(dtpsi_hat_log)
            dxpsi_re_log, dxpsi_im_log = split_reim(dxpsi_hat_log)

        training_stats.report('Loss/loss_operator', loss_operator_log)
        training_stats.report('Loss/loss_metric',   loss_metric_log)
        training_stats.report('Loss/dtpsi_rms',     (dtpsi_re_log**2 + dtpsi_im_log**2).mean().sqrt())
        training_stats.report('Loss/dxpsi_rms',     (dxpsi_re_log**2 + dxpsi_im_log**2).mean().sqrt())
        training_stats.report('Loss/xdot_rms',      (xdot**2).mean().sqrt())
        training_stats.report('Loss/abs_ip_mean',   (ip_re_log**2 + ip_im_log**2).sqrt().mean())

        # ------------------------------------------------------------
        # (G) Eigenvalue magnitude stats  (use full non-detached psi)
        # λ_i = e^{iφ_i} * ||ψ_i||²  →  |λ_i| = ||ψ_i||²
        # ------------------------------------------------------------
        psi_sq      = (psi_re**2 + psi_im**2).mean(dim=0)      # (k,)
        lam_re_vals = cos_phi.mean(dim=0) * psi_sq             # (k,)
        lam_im_vals = sin_phi.mean(dim=0) * psi_sq             # (k,)
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