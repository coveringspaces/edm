# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This is a research codebase extending the EDM (Elucidating the Design Space of Diffusion-Based Generative Models, NeurIPS 2022) framework. The original EDM code has been extended with:
1. **CFM (Conditional Flow Matching)** via TrigFlow formulation — added `CFMPrecond` and `CFMLoss`
2. **Koopman Flow Matching** — a second training stage that learns Koopman eigenfunctions on top of a frozen CFM vector field

## Environment Setup

```bash
conda env create -f environment.yml -n edm
conda activate edm
```

Note: `environment.yml` pins PyTorch 1.12.1, but the Koopman code uses `torch.func.jvp` / `vmap` which requires PyTorch 2.x. The actual installed environment likely diverges from the yml file.

## Common Commands

### Training (EDM/CFM)
```bash
# Single GPU
python train.py --outdir=training-runs --data=datasets/cifar10-32x32.zip --cond=1 --arch=ddpmpp --precond=cfm

# Multi-GPU (torchrun)
torchrun --standalone --nproc_per_node=2 train.py --outdir=training-runs \
    --data=datasets/cifar10-32x32.zip --cond=1 --arch=ddpmpp --precond=cfm --fp16=1

# Resume from checkpoint
python train.py --outdir=training-runs --data=datasets/cifar10-32x32.zip \
    --resume=training-runs/XXXXX-.../training-state-NNNNNN.pt ...
```

### Training (Koopman — second stage)
```bash
# Requires a frozen CFM snapshot (.pkl) from the first stage
torchrun --standalone --nproc_per_node=2 train_koopman.py \
    --outdir=training-runs-koopman \
    --data=datasets/cifar10-32x32.zip \
    --cond=1 \
    --cfm-pkl=training-runs/XXXXX-.../network-snapshot-NNNNNN.pkl \
    --k=64 --fp16=1
```

### Generation
```bash
python generate.py --outdir=out --seeds=0-63 --batch=64 \
    --network=training-runs/XXXXX-.../network-snapshot-NNNNNN.pkl

# Multi-GPU generation
torchrun --standalone --nproc_per_node=2 generate.py --outdir=out --seeds=0-999 --batch=64 \
    --network=<pkl>
```

### FID Evaluation
```bash
# Step 1: generate 50k images
torchrun --standalone --nproc_per_node=1 generate.py --outdir=fid-tmp --seeds=0-49999 --subdirs \
    --network=<pkl>

# Step 2: calculate FID
torchrun --standalone --nproc_per_node=1 fid.py calc --images=fid-tmp \
    --ref=fid-refs/cifar10-32x32.npz
```

### Dataset Preparation
```bash
python dataset_tool.py --source=downloads/cifar10/cifar-10-python.tar.gz \
    --dest=datasets/cifar10-32x32.zip
python fid.py ref --data=datasets/cifar10-32x32.zip --dest=fid-refs/cifar10-32x32.npz
```

### Inspecting Koopman Snapshots
```bash
python scripts/probe_koopman_snapshot.py  # hardcoded snapshot path, edit before running
python scripts/plot_koopman_loss.py
```

## Architecture

### `training/networks.py`
All model architectures and preconditioning wrappers:
- `SongUNet` — DDPM++ / NCSN++ backbone used with `--arch=ddpmpp` or `--arch=ncsnpp`
- `DhariwalUNet` — ADM backbone used with `--arch=adm`
- `VPPrecond`, `VEPrecond`, `EDMPrecond` — original EDM preconditioning wrappers
- `CFMPrecond` — TrigFlow-based CFM preconditioning; uses `t ∈ [0, π/2]` where `σ = tan(t) * σ_data`; outputs `x0_hat` (not velocity); optionally returns `logvar` for uncertainty weighting
- `MPFourier`, `MPConv` — magnitude-preserving layers used for the logvar head in CFMPrecond

### `training/loss.py`
- `EDMLoss`, `VPLoss`, `VELoss` — original losses
- `CFMLoss` — TrigFlow CFM loss: samples `σ ~ lognormal`, converts to `t = atan(σ/σ_data)`, constructs `x = cos(t)*y + sin(t)*σ_data*ε`, trains with uncertainty weighting `(1/exp(logvar)) * ||v_pred - v_star||² + logvar`

### `training/koopman.py`
Second-stage Koopman Flow Matching:
- `DhariwalEncoderOnly` — strips the decoder from `DhariwalUNet`, keeps mapping + encoder
- `KoopmanEigenNet` — encoder-only network outputting `(B, 2k)` real values representing `k` complex eigenfunctions `ψ_i(x,t)`
- `KoopmanPhases` — learnable phase parameters `φ_i` defining the *phase* of each eigenvalue; the full eigenvalue is `λ_i = e^{iφ_i} · ‖ψ_{θ,i}‖²`, so the magnitude `|λ_i| = ‖ψ_{θ,i}‖²` is determined by the network output, not fixed at 1
- `KoopmanLoss` — uses `torch.func.jvp` to compute directional derivatives of ψ along the CFM vector field; enforces `dψ/dt = iφ * ψ` (eigenfunction equation) plus an anti-collapse regularizer
  - **Eigenvalue magnitude**: NOT constrained to 1. Forcing `‖ψ_{θ,i}‖ = 1` would force `|λ_i| = 1`, which assumes the Koopman operator is unitary — not generally true for a learned CFM vector field. Per-component normalization is therefore NOT applied.
  - **Loss minimum**: `ψ=0` gives loss=0 but is a saddle, not the global minimum. The true minimum is achieved when the `ψ_i` are eigenfunctions of the CFM Koopman operator with eigenvalues `λ_i = e^{iφ_i} · ‖ψ_{θ,i}‖²`.

### `training/training_loop.py` / `training/training_loop_koopman.py`
Both loops follow the same pattern: DDP wrapping, EMA tracking, gradient accumulation, periodic snapshot saves (`network-snapshot-*.pkl`) and state dumps (`training-state-*.pt` / `koopman-training-state-*.pt`).

### `torch_utils/persistence.py`
`@persistence.persistent_class` decorator embeds source code in pickled objects so snapshots remain loadable even if code changes. **All network classes must use this decorator** to be pickle-compatible.

### `dnnlib/util.py`
`construct_class_by_name(class_name=..., **kwargs)` — used throughout to instantiate classes by dotted string name from config dicts.

## Key Design Patterns

- Training configs are `dnnlib.EasyDict` dicts serialized to `training_options.json` in each run directory
- Multi-GPU training uses `torchrun` + `torch.distributed`; all logging uses `dist.print0()` (rank-0 only)
- Run directories auto-numbered: `training-runs/NNNNN-<desc>/` and `training-runs-koopman/NNNNN-<desc>/`
- The CFM teacher network is loaded from a `.pkl` snapshot and kept frozen (no grad) during Koopman training
- `PYTHONPATH` must include the repo root when using models from pickles in external scripts
