"""Train Koopman eigenfunctions/phases on top of a frozen CFM vector field."""

import os
import re
import json
import click
import torch
import dnnlib
from torch_utils import distributed as dist

from training import training_loop_koopman

import warnings
warnings.filterwarnings('ignore', 'Grad strides do not match bucket view strides')

# ----------------------------------------------------------------------------

def parse_int_list(s):
    if isinstance(s, list):
        return s
    ranges = []
    range_re = re.compile(r'^(\d+)-(\d+)$')
    for p in s.split(','):
        m = range_re.match(p)
        if m:
            ranges.extend(range(int(m.group(1)), int(m.group(2)) + 1))
        else:
            ranges.append(int(p))
    return ranges

# ----------------------------------------------------------------------------

@click.command()

# Main options.
@click.option('--outdir',       help='Where to save the results', metavar='DIR',           type=str, required=True)
@click.option('--data',         help='Path to dataset', metavar='ZIP|DIR',                 type=str, required=True)
@click.option('--cond',         help='Use class labels', metavar='BOOL',                   type=bool, default=False, show_default=True)
@click.option('--xflip',        help='Enable dataset x-flips', metavar='BOOL',             type=bool, default=False, show_default=True)

# Frozen teacher CFM.
@click.option('--cfm-pkl',      help='CFM network snapshot pickle (teacher)', metavar='PKL|URL', type=str, required=True)

# Koopman model choices.
@click.option('--k',            help='Number of Koopman eigenfunctions', metavar='INT',    type=click.IntRange(min=1), default=32, show_default=True)
@click.option('--arch',         help='Psi-network backbone', metavar='adm-enc',            type=click.Choice(['adm-enc']), default='adm-enc', show_default=True)
@click.option('--fp16',         help='Use FP16 in psi_net encoder', metavar='BOOL',        type=bool, default=False, show_default=True)

# Hyperparameters.
@click.option('--duration',     help='Training duration', metavar='MIMG',                  type=click.FloatRange(min=0, min_open=True), default=50, show_default=True)
@click.option('--batch',        help='Total batch size', metavar='INT',                    type=click.IntRange(min=1), default=512, show_default=True)
@click.option('--batch-gpu',    help='Limit batch size per GPU', metavar='INT',            type=click.IntRange(min=1))
@click.option('--lr',           help='Learning rate', metavar='FLOAT',                     type=click.FloatRange(min=0, min_open=True), default=1e-5, show_default=True)
@click.option('--lr-rampup',    help='LR ramp-up', metavar='MIMG',                         type=click.FloatRange(min=0), default=1.0, show_default=True)
@click.option('--ema',          help='EMA half-life', metavar='MIMG',                      type=click.FloatRange(min=0), default=0.5, show_default=True)

# ADM-encoder-ish knobs for psi_net (mirrors your KoopmanEigenNet init).
@click.option('--cbase',        help='Model channels (ADM base)', metavar='INT',           type=int, default=192, show_default=True)
@click.option('--cres',         help='Channel mult list', metavar='LIST',                  type=parse_int_list, default='1,2,3,4', show_default=True)
@click.option('--dropout',      help='Dropout probability', metavar='FLOAT',               type=click.FloatRange(min=0, max=1), default=0.10, show_default=True)
@click.option('--operator-scale',      help='Scale of CFM vector field (divides term1)', metavar='FLOAT', type=click.FloatRange(min=0, min_open=True), default=100.0, show_default=True)
@click.option('--normalize-psi-loss', help='Normalize psi per-mode before loss (diagnostic)', is_flag=True, default=False)
@click.option('--grad-clip',    help='Max gradient norm for clipping', metavar='FLOAT',            type=click.FloatRange(min=0, min_open=True), default=100.0, show_default=True)

# Augment (optional; same as EDM train.py pattern).
@click.option('--augment',      help='Augment probability', metavar='FLOAT',               type=click.FloatRange(min=0, max=1), default=0.12, show_default=True)
@click.option('--cache',        help='Cache dataset in CPU memory', metavar='BOOL',        type=bool, default=True, show_default=True)
@click.option('--workers',      help='DataLoader workers', metavar='INT',                  type=click.IntRange(min=1), default=1, show_default=True)

# I/O / logging.
@click.option('--desc',         help='String to include in result dir name', metavar='STR', type=str)
@click.option('--nosubdir',     help='Do not create subdir for results', is_flag=True)
@click.option('--tick',         help='How often to print progress', metavar='KIMG',        type=click.IntRange(min=1), default=50, show_default=True)
@click.option('--snap',         help='How often to save snapshots', metavar='TICKS',       type=click.IntRange(min=1), default=50, show_default=True)
@click.option('--dump',         help='How often to dump state', metavar='TICKS',           type=click.IntRange(min=1), default=500, show_default=True)

# Resume.
@click.option('--resume',       help='Resume from previous koopman-training-state-*.pt', metavar='PT', type=str)

# W&B.
@click.option('--wandb-project', help='Weights & Biases project name', metavar='STR', type=str, default=None)
@click.option('--wandb-name',    help='Weights & Biases run name', metavar='STR', type=str, default=None)

# Seed.
@click.option('--seed',         help='Random seed [default: random]', metavar='INT',       type=int)
@click.option('-n', '--dry-run', help='Print options and exit', is_flag=True)

def main(**kwargs):
    opts = dnnlib.EasyDict(kwargs)

    torch.multiprocessing.set_start_method('spawn', force=True)
    dist.init()

    # ----------------------------------------------------------------------
    # Initialize config dict (mirrors EDM train.py).
    # ----------------------------------------------------------------------
    c = dnnlib.EasyDict()

    # Dataset / loader.
    c.dataset_kwargs = dnnlib.EasyDict(
        class_name='training.dataset.ImageFolderDataset',
        path=opts.data,
        use_labels=opts.cond,
        xflip=opts.xflip,
        cache=opts.cache,
    )
    c.data_loader_kwargs = dnnlib.EasyDict(pin_memory=True, num_workers=opts.workers, prefetch_factor=2)

    # Validate dataset (also captures name/resolution/size).
    try:
        dataset_obj = dnnlib.util.construct_class_by_name(**c.dataset_kwargs)
        dataset_name = dataset_obj.name
        c.dataset_kwargs.resolution = dataset_obj.resolution
        c.dataset_kwargs.max_size = len(dataset_obj)
        if opts.cond and not dataset_obj.has_labels:
            raise click.ClickException('--cond=True requires labels specified in dataset.json')
        del dataset_obj
    except IOError as err:
        raise click.ClickException(f'--data: {err}')

    # ----------------------------------------------------------------------
    # Frozen CFM net (teacher).
    # IMPORTANT: cfm_network_kwargs must construct the SAME class you used
    # in CFM training (e.g., training.networks.CFMPrecond).
    # ----------------------------------------------------------------------
    # c.cfm_network_kwargs = dnnlib.EasyDict()
    # # You can keep the arch choice simple at first:
    # # assume your teacher was trained with CFMPrecond wrapping DhariwalUNet.
    # # If your teacher is something else, you’ll change these two lines.
    # c.cfm_network_kwargs.class_name = 'training.networks.CFMPrecond'
    # c.cfm_network_kwargs.model_type = 'DhariwalUNet'
    # c.cfm_network_kwargs.model_channels = 192
    # c.cfm_network_kwargs.channel_mult = [1, 2, 3, 4]
    c.cfm_resume_pkl = opts.cfm_pkl

    # ----------------------------------------------------------------------
    # Psi network (trainable KoopmanEigenNet) + phase net.
    # ----------------------------------------------------------------------
    c.psi_network_kwargs = dnnlib.EasyDict(
        class_name='training.koopman.KoopmanEigenNet',
        k=opts.k,
        use_fp16=opts.fp16,
        model_channels=opts.cbase,
        channel_mult=opts.cres,
        dropout=opts.dropout,
        # keep the rest defaults unless you want CLI flags for them:
        channel_mult_emb=4,
        num_blocks=3,
        attn_resolutions=[32, 16, 8],
        label_dropout=0,
    )
    c.phase_kwargs = dnnlib.EasyDict(
        class_name='training.koopman.KoopmanPhases',
        k=opts.k,
        init_zero=False,
    )

    # Koopman loss.
    c.koopman_loss_kwargs = dnnlib.EasyDict(
        class_name='training.koopman.KoopmanLoss',
        P_mean=-1.2,
        P_std=1.2,
        sigma_data=0.5,
        sigma_min=1e-3,
        sigma_max=80,
        t_epsilon=1e-4,
        operator_scale=opts.operator_scale,

    )

    # Optimizer.
    c.optimizer_kwargs = dnnlib.EasyDict(class_name='torch.optim.Adam', lr=opts.lr, betas=[0.9, 0.999], eps=1e-8)

    # Augment (optional).
    if opts.augment and opts.augment > 0:
        c.augment_kwargs = dnnlib.EasyDict(class_name='training.augment.AugmentPipe', p=opts.augment)
        c.augment_kwargs.update(xflip=1e8, yflip=1, scale=1, rotate_frac=1, aniso=1, translate_frac=1)
        # psi_net needs augment_dim if you actually pass augment labels into the encoder
        # (your KoopmanEigenNet passes augment_labels through to Dhariwal blocks)
        c.psi_network_kwargs.augment_dim = 9
        # c.cfm_network_kwargs.augment_dim = 9  # teacher net also expects augment labels if used

    # Training options.
    c.grad_clip_norm = opts.grad_clip
    c.total_kimg = max(int(opts.duration * 1000), 1)
    c.lr_rampup_kimg = int(opts.lr_rampup * 1000)
    c.ema_halflife_kimg = int(opts.ema * 1000)
    c.batch_size = opts.batch
    c.batch_gpu = opts.batch_gpu
    c.kimg_per_tick = opts.tick
    c.snapshot_ticks = opts.snap
    c.state_dump_ticks = opts.dump
    c.wandb_project = opts.wandb_project
    c.wandb_name = opts.wandb_name

    # Seed.
    if opts.seed is not None:
        c.seed = opts.seed
    else:
        seed = torch.randint(1 << 31, size=[], device=torch.device('cuda'))
        torch.distributed.broadcast(seed, src=0)
        c.seed = int(seed)

    # Resume.
    if opts.resume is not None:
        match = re.fullmatch(r'koopman-training-state-(\d+).pt', os.path.basename(opts.resume))
        if not match or not os.path.isfile(opts.resume):
            raise click.ClickException('--resume must point to koopman-training-state-*.pt from a previous run')
        c.resume_kimg = int(match.group(1))
        c.resume_state_dump = opts.resume

    # ----------------------------------------------------------------------
    # Run dir naming (same style as EDM).
    # ----------------------------------------------------------------------
    cond_str = 'cond' if c.dataset_kwargs.use_labels else 'uncond'
    dtype_str = 'fp16' if opts.fp16 else 'fp32'
    desc = f'{dataset_name:s}-{cond_str:s}-koopman-k{opts.k:d}-gpus{dist.get_world_size():d}-batch{c.batch_size:d}-{dtype_str:s}'
    if opts.desc is not None:
        desc += f'-{opts.desc}'

    if dist.get_rank() != 0:
        c.run_dir = None
    elif opts.nosubdir:
        c.run_dir = opts.outdir
    else:
        prev_run_dirs = []
        if os.path.isdir(opts.outdir):
            prev_run_dirs = [x for x in os.listdir(opts.outdir) if os.path.isdir(os.path.join(opts.outdir, x))]
        prev_run_ids = [re.match(r'^\d+', x) for x in prev_run_dirs]
        prev_run_ids = [int(x.group()) for x in prev_run_ids if x is not None]
        cur_run_id = max(prev_run_ids, default=-1) + 1
        c.run_dir = os.path.join(opts.outdir, f'{cur_run_id:05d}-{desc}')
        assert not os.path.exists(c.run_dir)

    # Print options.
    dist.print0()
    dist.print0('Koopman training options:')
    dist.print0(json.dumps(c, indent=2))
    dist.print0()
    dist.print0(f'Output directory:        {c.run_dir}')
    dist.print0(f'Dataset path:            {c.dataset_kwargs.path}')
    dist.print0(f'CFM teacher snapshot:    {c.cfm_resume_pkl}')
    dist.print0(f'Class-conditional:       {c.dataset_kwargs.use_labels}')
    dist.print0(f'k:                       {opts.k}')
    dist.print0(f'Number of GPUs:          {dist.get_world_size()}')
    dist.print0(f'Batch size:              {c.batch_size}')
    dist.print0(f'FP16 psi encoder:        {opts.fp16}')
    dist.print0()

    if opts.dry_run:
        dist.print0('Dry run; exiting.')
        return

    # Create output directory + log.
    dist.print0('Creating output directory...')
    if dist.get_rank() == 0:
        os.makedirs(c.run_dir, exist_ok=True)
        with open(os.path.join(c.run_dir, 'training_options.json'), 'wt') as f:
            json.dump(c, f, indent=2)
        dnnlib.util.Logger(file_name=os.path.join(c.run_dir, 'log.txt'), file_mode='a', should_flush=True)

    # Train.
    training_loop_koopman.koopman_training_loop(**c)

# ----------------------------------------------------------------------------

if __name__ == "__main__":
    main()