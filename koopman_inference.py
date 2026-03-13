"""
koopman_inference.py — One-step Koopman spectral sampling.

Algorithm
---------
1. Estimate coefficients  c_i = E[conj(ψ_i(x0, t0)) * x0]
   and eigenvalues        λ_i = e^{iφ_i} * E[|ψ_i(x0, t0)|²]
   by Monte Carlo over x0 ~ N(0, I).

2. For a fresh test noise x_test ~ N(0, I), reconstruct:
       x1_hat = Re( Σ_{i=1}^{test_k}  e^{λ_i * T}  *  c_i  *  ψ_i(x_test, t0) )

3. Save a grid of generated images.

Usage
-----
    python koopman_inference.py \
        --pkl=training-runs-koopman/00000-.../koopman-snapshot-002500.pkl \
        --outdir=koopman-inference-out \
        --num-images=64 --test-k=64
"""

import os
import sys
import math
import pickle
import argparse

import numpy as np
import torch
import PIL.Image
import torchvision.utils

# Ensure repo root is on path so pickled classes resolve correctly.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from torch.nn.functional import silu


# ---------------------------------------------------------------------------
# Manual forward — works around the isinstance(block, UNetBlock) pickle bug
# where blocks loaded from a CFM teacher snapshot have a different class
# identity than UNetBlock in the current process.
# ---------------------------------------------------------------------------

@torch.no_grad()
def _psi_forward(model, x, t, class_labels=None, augment_labels=None):
    x = x.to(torch.float32)
    t = t.to(torch.float32).reshape(-1, 1, 1, 1)

    if model.label_dim == 0:
        class_labels = None
    elif class_labels is None:
        class_labels = torch.zeros([x.shape[0], model.label_dim], device=x.device)
    else:
        class_labels = class_labels.to(torch.float32).reshape(-1, model.label_dim)

    c_in         = 1.0 / model.sigma_data
    noise_labels = t.flatten()
    enc          = model.encoder

    emb = enc.map_noise(noise_labels)
    if enc.map_augment is not None and augment_labels is not None:
        emb = emb + enc.map_augment(augment_labels)
    emb = silu(enc.map_layer0(emb))
    emb = enc.map_layer1(emb)
    if enc.map_label is not None:
        emb = emb + enc.map_label(class_labels)
    emb = silu(emb)

    h = (c_in * x).to(torch.float32)
    for block in enc.enc.values():
        h = block(h, emb) if hasattr(block, 'emb_channels') else block(h)

    h = model.pool(h).flatten(1)
    return model.head(h)   # (B, 2k)


# ---------------------------------------------------------------------------
# Snapshot I/O
# ---------------------------------------------------------------------------

def load_snapshot(pkl_path: str, device: torch.device):
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    psi_net   = data['psi_ema'].to(device).eval().requires_grad_(False)
    phase_net = data['phase_ema'].to(device).eval().requires_grad_(False)
    # Pull image resolution out of stored dataset_kwargs if available.
    meta = {}
    dkw = data.get('dataset_kwargs', {})
    if 'resolution' in dkw:
        meta['img_resolution'] = dkw['resolution']
    return psi_net, phase_net, meta


# ---------------------------------------------------------------------------
# Coefficient + eigenvalue estimation
# ---------------------------------------------------------------------------

@torch.no_grad()
def estimate_coefficients(psi_net, phase_net, *,
                           img_shape, label_dim,
                           num_samples, batch_size, t0, device):
    """
    Returns
    -------
    lam_re, lam_im  : (k,)       real / imag parts of λ_i
    c_re,   c_im    : (k,C,H,W)  real / imag parts of c_i
    """
    C, H, W = img_shape
    k = psi_net.k

    psi_sq_acc = torch.zeros(k,          device=device)
    c_re_acc   = torch.zeros(k, C, H, W, device=device)
    c_im_acc   = torch.zeros(k, C, H, W, device=device)
    n_done = 0

    while n_done < num_samples:
        bs    = min(batch_size, num_samples - n_done)
        x0    = torch.randn(bs, C, H, W, device=device)
        t_b   = torch.full([bs], t0, device=device)
        # Use uniform random one-hot class labels so coefficients are
        # averaged over the full class distribution.
        if label_dim > 0:
            cls_idx = torch.randint(label_dim, (bs,), device=device)
            labels  = torch.zeros(bs, label_dim, device=device)
            labels.scatter_(1, cls_idx.unsqueeze(1), 1.0)
        else:
            labels = None

        psi_hat = _psi_forward(psi_net, x0, t_b, class_labels=labels)  # (bs, 2k)
        psi_re  = psi_hat[:, :k]   # (bs, k)
        psi_im  = psi_hat[:, k:]   # (bs, k)

        # Accumulate |ψ_i|² for eigenvalue magnitude
        psi_sq_acc += (psi_re**2 + psi_im**2).sum(dim=0)

        # c_i = E[conj(ψ_i) * x0]
        #   Re(conj(ψ_i) * x0) =  ψ_re_i * x0
        #   Im(conj(ψ_i) * x0) = -ψ_im_i * x0
        c_re_acc += torch.einsum('bi,bchw->ichw',  psi_re, x0)
        c_im_acc += torch.einsum('bi,bchw->ichw', -psi_im, x0)

        n_done += bs

    psi_sq_mean = psi_sq_acc / n_done   # (k,)
    c_re        = c_re_acc   / n_done   # (k, C, H, W)
    c_im        = c_im_acc   / n_done   # (k, C, H, W)

    # λ_i = e^{iφ_i} * E[|ψ_i|²]
    dummy = torch.zeros(1, device=device)
    cos_phi, sin_phi = phase_net(dummy)          # (k,)
    lam_re = cos_phi * psi_sq_mean               # (k,)
    lam_im = sin_phi * psi_sq_mean               # (k,)

    return lam_re, lam_im, c_re, c_im


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

@torch.no_grad()
def generate(psi_net, *, lam_re, lam_im, c_re, c_im,
             img_shape, label_dim, num_images, batch_size,
             test_k, t0, final_t, class_idx, device):
    """
    x1_hat = Re( Σ_{i<test_k}  e^{λ_i * T}  *  c_i  *  ψ_i(x_test) )

    Parameters
    ----------
    class_idx : int or None
        Fixed class index (0-based) to condition on.  None = random per image.
    """
    C, H, W = img_shape
    k = psi_net.k
    T = final_t

    # e^{λ_i * T} = e^{Re(λ_i)*T} * [cos(Im(λ_i)*T) + i*sin(Im(λ_i)*T)]
    exp_scale = torch.exp(lam_re[:test_k] * T)               # (test_k,)
    exp_re    = exp_scale * torch.cos(lam_im[:test_k] * T)   # (test_k,)
    exp_im    = exp_scale * torch.sin(lam_im[:test_k] * T)   # (test_k,)

    c_re_t = c_re[:test_k]   # (test_k, C, H, W)
    c_im_t = c_im[:test_k]   # (test_k, C, H, W)

    all_imgs = []
    for start in range(0, num_images, batch_size):
        bs     = min(batch_size, num_images - start)
        x_test = torch.randn(bs, C, H, W, device=device)
        t_b    = torch.full([bs], t0, device=device)

        if label_dim > 0:
            if class_idx is None:
                ci = torch.randint(label_dim, (bs,), device=device)
            else:
                ci = torch.full((bs,), class_idx, dtype=torch.long, device=device)
            labels = torch.zeros(bs, label_dim, device=device)
            labels.scatter_(1, ci.unsqueeze(1), 1.0)
        else:
            labels = None

        psi_hat  = _psi_forward(psi_net, x_test, t_b, class_labels=labels)  # (bs, 2k)
        psi_re_t = psi_hat[:, :test_k]        # (bs, test_k) — real block
        psi_im_t = psi_hat[:, k:k + test_k]   # (bs, test_k) — imag block (starts at k)

        # Expand: A_i = e^{λ_i*T} * ψ_i(x)
        #   A_re = exp_re[i]*ψ_re[i] - exp_im[i]*ψ_im[i]   (bs, test_k)
        #   A_im = exp_re[i]*ψ_im[i] + exp_im[i]*ψ_re[i]   (bs, test_k)
        A_re = exp_re[None, :] * psi_re_t - exp_im[None, :] * psi_im_t
        A_im = exp_re[None, :] * psi_im_t + exp_im[None, :] * psi_re_t

        # x1_hat = Re( Σ_i A_i * c_i ) = Σ_i (A_re_i * c_re_i - A_im_i * c_im_i)
        x1 = (torch.einsum('bi,ichw->bchw', A_re, c_re_t)
            - torch.einsum('bi,ichw->bchw', A_im, c_im_t))

        all_imgs.append(x1.cpu())

    return torch.cat(all_imgs, dim=0)   # (num_images, C, H, W)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Koopman one-step spectral inference')
    parser.add_argument('--pkl',               required=True,
                        help='Path to koopman-snapshot-*.pkl')
    parser.add_argument('--outdir',            default='koopman-inference-out')
    parser.add_argument('--num-images',        type=int,   default=64,
                        help='Number of images to generate')
    parser.add_argument('--num-coeff-samples', type=int,   default=4096,
                        help='Monte Carlo samples for coefficient estimation')
    parser.add_argument('--batch',             type=int,   default=64)
    parser.add_argument('--test-k',            type=int,   default=None,
                        help='Eigenfunctions to use (default: all k in checkpoint)')
    parser.add_argument('--final-t',           type=float, default=math.pi / 2,
                        help='Integration time T (default π/2, the full TrigFlow trajectory)')
    parser.add_argument('--t0',                type=float, default=1e-4,
                        help='Initial noise time (should match training t_epsilon)')
    parser.add_argument('--img-resolution',    type=int,   default=32)
    parser.add_argument('--img-channels',      type=int,   default=3)
    parser.add_argument('--class-idx',         type=int,   default=None,
                        help='Fixed class index to condition on (None = random)')
    parser.add_argument('--nrow',              type=int,   default=8,
                        help='Images per row in output grid')
    parser.add_argument('--seed',              type=int,   default=0)
    parser.add_argument('--device',            type=str,   default='cuda')
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)
    os.makedirs(args.outdir, exist_ok=True)

    # Load -----------------------------------------------------------------
    print(f'Loading {args.pkl} ...')
    psi_net, phase_net, meta = load_snapshot(args.pkl, device)
    k         = psi_net.k
    label_dim = psi_net.label_dim

    img_resolution = meta.get('img_resolution', args.img_resolution)
    img_shape      = (args.img_channels, img_resolution, img_resolution)

    test_k = args.test_k if args.test_k is not None else k
    assert 1 <= test_k <= k, f'--test-k must be in [1, {k}]'

    print(f'  k={k}, test_k={test_k}, img_shape={img_shape}, label_dim={label_dim}')
    print(f'  T={args.final_t:.4f}  (π/2 ≈ {math.pi/2:.4f})')

    # Estimate coefficients ------------------------------------------------
    print(f'Estimating coefficients over {args.num_coeff_samples} noise samples ...')
    lam_re, lam_im, c_re, c_im = estimate_coefficients(
        psi_net, phase_net,
        img_shape=img_shape, label_dim=label_dim,
        num_samples=args.num_coeff_samples,
        batch_size=args.batch,
        t0=args.t0,
        device=device,
    )

    # Diagnostics
    lam_mag   = (lam_re**2 + lam_im**2).sqrt()
    exp_scale = torch.exp(lam_re * args.final_t)
    c_mag     = (c_re**2 + c_im**2).sum(dim=[1, 2, 3]).sqrt()
    print(f'  |λ_i|:           min={lam_mag.min():.4f}  max={lam_mag.max():.4f}  mean={lam_mag.mean():.4f}')
    print(f'  e^(Re(λ_i)*T):   min={exp_scale.min():.4f}  max={exp_scale.max():.4f}')
    print(f'  |c_i|:           min={c_mag.min():.4f}  max={c_mag.max():.4f}  mean={c_mag.mean():.4f}')

    if exp_scale.max() > 1e4:
        print('  WARNING: some e^(Re(λ_i)*T) values are very large — '
              'generated images may be dominated by a few eigenfunctions.')

    # Generate -------------------------------------------------------------
    print(f'Generating {args.num_images} images (test_k={test_k}) ...')
    images = generate(
        psi_net,
        lam_re=lam_re, lam_im=lam_im, c_re=c_re, c_im=c_im,
        img_shape=img_shape, label_dim=label_dim,
        num_images=args.num_images, batch_size=args.batch,
        test_k=test_k, t0=args.t0, final_t=args.final_t,
        class_idx=args.class_idx,
        device=device,
    )

    # Save grid ------------------------------------------------------------
    snap_kimg = os.path.basename(args.pkl).replace('koopman-snapshot-', '').replace('.pkl', '')
    stem      = f'koopman_k{test_k}_kimg{snap_kimg}_seed{args.seed}'
    out_png   = os.path.join(args.outdir, f'{stem}.png')
    images_u8 = ((images.clamp(-1, 1) + 1) * 127.5).to(torch.uint8)
    grid      = torchvision.utils.make_grid(images_u8, nrow=args.nrow, padding=2)
    PIL.Image.fromarray(grid.permute(1, 2, 0).numpy()).save(out_png)
    print(f'Saved grid  → {out_png}')

    # Save raw tensors for further analysis
    out_pt = os.path.join(args.outdir, f'{stem}_raw.pt')
    torch.save(images, out_pt)
    print(f'Saved raw   → {out_pt}')


if __name__ == '__main__':
    main()
