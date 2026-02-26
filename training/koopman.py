# training/koopman.py
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.func import jvp

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
# Reimplementation of the ADM architecture from the paper
# "Diffusion Models Beat GANS on Image Synthesis". Equivalent to the
# original implementation by Dhariwal and Nichol, available at
# https://github.com/openai/guided-diffusion

@persistence.persistent_class
class DhariwalUNet(torch.nn.Module):
    def __init__(self,
        img_resolution,                     # Image resolution at input/output.
        in_channels,                        # Number of color channels at input.
        out_channels,                       # Number of color channels at output.
        label_dim           = 0,            # Number of class labels, 0 = unconditional.
        augment_dim         = 0,            # Augmentation label dimensionality, 0 = no augmentation.

        model_channels      = 192,          # Base multiplier for the number of channels.
        channel_mult        = [1,2,3,4],    # Per-resolution multipliers for the number of channels.
        channel_mult_emb    = 4,            # Multiplier for the dimensionality of the embedding vector.
        num_blocks          = 3,            # Number of residual blocks per resolution.
        attn_resolutions    = [32,16,8],    # List of resolutions with self-attention.
        dropout             = 0.10,         # List of resolutions with self-attention.
        label_dropout       = 0,            # Dropout probability of class labels for classifier-free guidance.
    ):
        super().__init__()
        self.label_dropout = label_dropout
        emb_channels = model_channels * channel_mult_emb
        init = dict(init_mode='kaiming_uniform', init_weight=np.sqrt(1/3), init_bias=np.sqrt(1/3))
        init_zero = dict(init_mode='kaiming_uniform', init_weight=0, init_bias=0)
        block_kwargs = dict(emb_channels=emb_channels, channels_per_head=64, dropout=dropout, init=init, init_zero=init_zero)

        # Mapping.
        self.map_noise = PositionalEmbedding(num_channels=model_channels)
        self.map_augment = Linear(in_features=augment_dim, out_features=model_channels, bias=False, **init_zero) if augment_dim else None
        self.map_layer0 = Linear(in_features=model_channels, out_features=emb_channels, **init)
        self.map_layer1 = Linear(in_features=emb_channels, out_features=emb_channels, **init)
        self.map_label = Linear(in_features=label_dim, out_features=emb_channels, bias=False, init_mode='kaiming_normal', init_weight=np.sqrt(label_dim)) if label_dim else None

        # Encoder.
        self.enc = torch.nn.ModuleDict()
        cout = in_channels
        for level, mult in enumerate(channel_mult):
            res = img_resolution >> level
            if level == 0:
                cin = cout
                cout = model_channels * mult
                self.enc[f'{res}x{res}_conv'] = Conv2d(in_channels=cin, out_channels=cout, kernel=3, **init)
            else:
                self.enc[f'{res}x{res}_down'] = UNetBlock(in_channels=cout, out_channels=cout, down=True, **block_kwargs)
            for idx in range(num_blocks):
                cin = cout
                cout = model_channels * mult
                self.enc[f'{res}x{res}_block{idx}'] = UNetBlock(in_channels=cin, out_channels=cout, attention=(res in attn_resolutions), **block_kwargs)
        skips = [block.out_channels for block in self.enc.values()]

        # Decoder.
        self.dec = torch.nn.ModuleDict()
        for level, mult in reversed(list(enumerate(channel_mult))):
            res = img_resolution >> level
            if level == len(channel_mult) - 1:
                self.dec[f'{res}x{res}_in0'] = UNetBlock(in_channels=cout, out_channels=cout, attention=True, **block_kwargs)
                self.dec[f'{res}x{res}_in1'] = UNetBlock(in_channels=cout, out_channels=cout, **block_kwargs)
            else:
                self.dec[f'{res}x{res}_up'] = UNetBlock(in_channels=cout, out_channels=cout, up=True, **block_kwargs)
            for idx in range(num_blocks + 1):
                cin = cout + skips.pop()
                cout = model_channels * mult
                self.dec[f'{res}x{res}_block{idx}'] = UNetBlock(in_channels=cin, out_channels=cout, attention=(res in attn_resolutions), **block_kwargs)
        self.out_norm = GroupNorm(num_channels=cout)
        self.out_conv = Conv2d(in_channels=cout, out_channels=out_channels, kernel=3, **init_zero)

    def forward(self, x, noise_labels, class_labels, augment_labels=None):
        # Mapping.
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

        # Encoder.
        skips = []
        for block in self.enc.values():
            x = block(x, emb) if isinstance(block, UNetBlock) else block(x)
            skips.append(x)

        # Decoder.
        for block in self.dec.values():
            if x.shape[1] != block.in_channels:
                x = torch.cat([x, skips.pop()], dim=1)
            x = block(x, emb)
        x = self.out_conv(silu(self.out_norm(x)))
        return x

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
            x = block(x, emb) if isinstance(block, UNetBlock) else block(x)

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
        dropout=0.10,
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

    def forward(self):
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


#----------------------------------------------------------------------------
# Loss from equation (4) of Jon's writeup

@persistence.persistent_class
class KoopmanLoss:
    def __init__(self,
                 P_mean=-1.2, P_std=1.2, sigma_data=0.5,
                 sigma_min=1e-3, sigma_max=80, t_epsilon=1e-4,
                 anti_collapse=1e-3):
        self.P_mean = P_mean
        self.P_std = P_std
        self.sigma_data = sigma_data
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.t_epsilon = t_epsilon
        self.anti_collapse = anti_collapse

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
        def f(x_in, t_in):
            # labels & augment_labels treated as constants (no grads through them)
            return psi_net(x_in, t_in, class_labels=labels, augment_labels=augment_labels)  # (B,2k)

        def _jvp_f(x_in, t_in, xdot, tdot):
            x_in = x_in.unsqueeze(0)
            t_in = t_in.unsqueeze(0)
            xdot = xdot.unsqueeze(0)
            tdot = tdot.unsqueeze(0)

            psi_hat, Lpsi_hat = self._jvp(f, (x_in, t_in), (xdot, tdot)
                                          
            return psi_hat.squeeze(0), Lpsi_hat.squeeze(0)

        psi_hat, Lpsi_hat = torch.func.vmap(_jvp_f)(x, t, xdot, tdot)  # both (B,2k)

        # ------------------------------------------------------------
        # (D) Convert to complex (re/im) blocks: psi = psi_re + i psi_im
        # ------------------------------------------------------------
        psi_re, psi_im     = split_reim(psi_hat)     # (B,k), (B,k)
        Lpsi_re, Lpsi_im   = split_reim(Lpsi_hat)    # (B,k), (B,k)

        B, k = psi_re.shape

        # ------------------------------------------------------------
        # (E) Term 1:  -2 Σ_i Re( e^{i φ_i} <psi_i, L psi_i> )
        # ------------------------------------------------------------
        ip_re, ip_im = complex_inner_batch(psi_re, psi_im, Lpsi_re, Lpsi_im)  # (k,) each
        cos_phi, sin_phi = phase_net()  # (k,), (k,)

        # e^{iφ}(ip_re + i ip_im) real-part:
        # Re( (cos + i sin)(ip_re + i ip_im) ) = cos*ip_re - sin*ip_im
        real_sum = cos_phi * ip_re - sin_phi * ip_im    # (k,)
        term1 = -2.0 * real_sum.sum()

        # ------------------------------------------------------------
        # (F) Term 2: Σ_{i,j} cos(φ_i - φ_j) |<psi_i,psi_j>|^2
        # ------------------------------------------------------------
        G_re, G_im = complex_gram(psi_re, psi_im)          # (k,k), (k,k)
        absG2 = G_re**2 + G_im**2                          # |G_ij|^2

        # cos(φ_i - φ_j) = cosφ_i cosφ_j + sinφ_i sinφ_j
        cos_dphi = cos_phi[:, None] * cos_phi[None, :] + sin_phi[:, None] * sin_phi[None, :]
        term2 = (cos_dphi * absG2).sum()

        loss = term1 + term2

        # ------------------------------------------------------------
        # (G) Anti-collapse (optional): prevents psi -> 0 everywhere
        # ------------------------------------------------------------
        if self.anti_collapse > 0:
            var = psi_re.var(dim=0).mean() + psi_im.var(dim=0).mean()
            loss = loss + self.anti_collapse * (1.0 / (var + 1e-6))

        return loss





