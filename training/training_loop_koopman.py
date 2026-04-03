import os
import time
import copy
import json
import pickle
import psutil
import numpy as np
import torch
import dnnlib

from torch_utils import distributed as dist
from torch_utils import training_stats
from torch_utils import misc

# ----------------------------------------------------------------------------

def koopman_training_loop(
    run_dir             = '.',      # Output directory.
    dataset_kwargs      = {},       # Options for training set.
    data_loader_kwargs  = {},       # Options for torch.utils.data.DataLoader.

    # --- Koopman pieces (trainable) ---
    psi_network_kwargs  = {},       # construct_class_by_name(...) for KoopmanEigenNet
    phase_kwargs        = {},       # construct_class_by_name(...) for KoopmanPhases
    koopman_loss_kwargs = {},       # construct_class_by_name(...) for KoopmanLoss
    optimizer_kwargs    = {},       # torch.optim.* kwargs for psi+phase params

    # --- Frozen CFM vector field (NOT trainable) ---
    cfm_resume_pkl      = None,     # load EMA weights from EDM snapshot into cfm net

    augment_kwargs      = None,     # training.augment.AugmentPipe, None = disable.
    seed                = 0,
    batch_size          = 512,
    batch_gpu           = None,
    total_kimg          = 50000,    # Koopman stage duration (kimg)
    lr_rampup_kimg      = 1000,     # smaller ramp for this stage is often fine

    # Optional EMA over psi_net (phases usually don’t need EMA, but harmless)
    ema_halflife_kimg   = 500,
    ema_rampup_ratio    = 0.05,

    loss_scaling        = 1,
    grad_clip_norm      = 10.0,     # max gradient norm for clipping
    kimg_per_tick       = 50,
    snapshot_ticks      = 50,
    state_dump_ticks    = 500,

    resume_state_dump   = None,
    resume_kimg         = 0,
    cudnn_benchmark     = True,
    device              = torch.device('cuda'),
    wandb_project       = None,     # W&B project name, None = disabled
    wandb_name          = None,     # W&B run name (optional)
):
    # ------------------------------------------------------------------------
    # Init.
    # ------------------------------------------------------------------------
    start_time = time.time()

    if wandb_project is not None and dist.get_rank() == 0:
        import wandb
        wandb.init(project=wandb_project, name=wandb_name, resume='allow')
    np.random.seed((seed * dist.get_world_size() + dist.get_rank()) % (1 << 31))
    torch.manual_seed(np.random.randint(1 << 31))

    torch.backends.cudnn.benchmark = cudnn_benchmark
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False

    # Select batch size per GPU.
    batch_gpu_total = batch_size // dist.get_world_size()
    if batch_gpu is None or batch_gpu > batch_gpu_total:
        batch_gpu = batch_gpu_total
    num_accumulation_rounds = batch_gpu_total // batch_gpu
    assert batch_size == batch_gpu * num_accumulation_rounds * dist.get_world_size()

    # ------------------------------------------------------------------------
    # Load dataset.
    # ------------------------------------------------------------------------
    dist.print0('Loading dataset...')
    dataset_obj = dnnlib.util.construct_class_by_name(**dataset_kwargs)
    dataset_sampler = misc.InfiniteSampler(
        dataset=dataset_obj,
        rank=dist.get_rank(),
        num_replicas=dist.get_world_size(),
        seed=seed
    )
    dataset_iterator = iter(torch.utils.data.DataLoader(
        dataset=dataset_obj,
        sampler=dataset_sampler,
        batch_size=batch_gpu,
        **data_loader_kwargs
    ))

    interface_kwargs = dict(
    img_resolution=dataset_obj.resolution,
    img_channels=dataset_obj.num_channels,
    label_dim=dataset_obj.label_dim,
    )

    # ------------------------------------------------------------------------
    # Load frozen CFM net (vector field provider) directly from snapshot.
    # ------------------------------------------------------------------------
    dist.print0('Loading frozen CFM network from snapshot...')

    assert cfm_resume_pkl is not None, "Need --cfm-pkl to provide a frozen teacher."

    if dist.get_rank() != 0:
        torch.distributed.barrier()  # rank 0 goes first

    with dnnlib.util.open_url(cfm_resume_pkl, verbose=(dist.get_rank() == 0)) as f:
        data = pickle.load(f)

    if dist.get_rank() == 0:
        torch.distributed.barrier()  # other ranks follow

    # Use the EMA model from the snapshot (already has correct architecture).
    cfm_net = data['ema'].to(device).eval().requires_grad_(False)

    # Optional: if you want to reuse augment pipe from teacher snapshot when present:
    # if augment_pipe is None and data.get('augment_pipe', None) is not None:
    #     augment_pipe = data['augment_pipe'].to(device).eval()

    del data

    # ------------------------------------------------------------------------
    # Construct Koopman nets.
    # ------------------------------------------------------------------------
    dist.print0('Constructing Koopman networks (psi + phases)...')
    psi_net = dnnlib.util.construct_class_by_name(**psi_network_kwargs, **interface_kwargs)
    psi_net.train().requires_grad_(True).to(device)

    # # Manual weight scaling to prevent JVP explosion
    # with torch.no_grad():
    #     for name, param in psi_net.named_parameters():
    #         # Check for standard names in Dhariwal/ADM-style blocks
    #         if any(x in name for x in ['qkv', 'proj', 'head', 'out']):
    #             param.data.mul_(0.01)
    #             dist.print0(f'Scaled down: {name}')

    phase_net = dnnlib.util.construct_class_by_name(**phase_kwargs, label_dim=interface_kwargs['label_dim'])
    phase_net.train().requires_grad_(True).to(device)

    # Optional: print summary for psi_net.
    if dist.get_rank() == 0:
        with torch.no_grad():
            img_channels = dataset_obj.num_channels
            img_resolution = dataset_obj.resolution

            images = torch.zeros([batch_gpu, img_channels, img_resolution, img_resolution], device=device)

            t = torch.ones([batch_gpu], device=device) * 0.5
            labels = torch.zeros([batch_gpu, psi_net.label_dim], device=device)
            misc.print_module_summary(psi_net, [images, t, labels], max_nesting=2)

    # ------------------------------------------------------------------------
    # Optimizer / loss / augment / DDP / EMA.
    # ------------------------------------------------------------------------
    dist.print0('Setting up optimizer & loss...')
    koop_loss = dnnlib.util.construct_class_by_name(**koopman_loss_kwargs)  # your KoopmanLoss
    augment_pipe = dnnlib.util.construct_class_by_name(**augment_kwargs) if augment_kwargs is not None else None

    # Trainable params = psi_net + phase_net
    train_params = list(psi_net.parameters()) + list(phase_net.parameters())
    optimizer = dnnlib.util.construct_class_by_name(params=train_params, **optimizer_kwargs)

    # No DDP wrappers: torch.func.jvp is incompatible with DDP.
    # Gradient synchronisation is done manually via all_reduce after backward.

    ema_psi = copy.deepcopy(psi_net).eval().requires_grad_(False)
    ema_phase = copy.deepcopy(phase_net).eval().requires_grad_(False)

    # 1. Ensure all parameters are synced, including unused ones
    if dist.get_world_size() > 1:
        dist.print0('Synchronizing initial parameters across ranks...')
        for param in psi_net.parameters():
            torch.distributed.broadcast(param.data, src=0)
        for param in phase_net.parameters():
            torch.distributed.broadcast(param.data, src=0)
        
        # 2. Sync EMA buffers to prevent save-time drift
        for p_ema in ema_psi.parameters():
            torch.distributed.broadcast(p_ema.data, src=0)
        for p_ema in ema_phase.parameters():
            torch.distributed.broadcast(p_ema.data, src=0)

    dist.print0('Initialization complete. Starting training loop...')

    # Resume (Koopman stage) from training-state dump, if any.
    if resume_state_dump:
        dist.print0(f'Loading Koopman training state from "{resume_state_dump}"...')
        data = torch.load(resume_state_dump, map_location=torch.device('cpu'))
        misc.copy_params_and_buffers(src_module=data['psi_net'], dst_module=psi_net, require_all=True)
        misc.copy_params_and_buffers(src_module=data['phase_net'], dst_module=phase_net, require_all=True)
        optimizer.load_state_dict(data['optimizer_state'])
        del data

    # ------------------------------------------------------------------------
    # Train.
    # ------------------------------------------------------------------------
    dist.print0(f'Training Koopman for {total_kimg} kimg...')
    dist.print0()

    cur_nimg = resume_kimg * 1000
    cur_tick = 0
    tick_start_nimg = cur_nimg
    tick_start_time = time.time()
    maintenance_time = tick_start_time - start_time

    dist.update_progress(cur_nimg // 1000, total_kimg)
    stats_jsonl = None

    def _allreduce_and_step():
        """All-reduce grads, apply LR ramp, NaN-guard, clip, then step."""
        if dist.get_world_size() > 1:
            for param in train_params:
                if param.grad is not None:
                    torch.distributed.all_reduce(param.grad)
                    param.grad.div_(dist.get_world_size())
        for g in optimizer.param_groups:
            base_lr = optimizer_kwargs.get('lr', g.get('lr', 1e-4))
            g['lr'] = base_lr * min(cur_nimg / max(lr_rampup_kimg * 1000, 1e-8), 1.0)
        for p in train_params:
            if p.grad is not None:
                torch.nan_to_num(p.grad, nan=0, posinf=1e5, neginf=-1e5, out=p.grad)
        grad_norm = torch.sqrt(sum(p.grad.norm()**2 for p in train_params if p.grad is not None))
        training_stats.report('Grads/grad_norm', grad_norm)
        torch.nn.utils.clip_grad_norm_(train_params, max_norm=grad_clip_norm)
        clipped_norm = torch.sqrt(sum(p.grad.norm()**2 for p in train_params if p.grad is not None))
        training_stats.report('Grads/grad_norm_clipped', clipped_norm)
        optimizer.step()

    def _ema_and_count():
        """EMA update + cur_nimg increment for one optimizer step worth of batch_size images."""
        nonlocal cur_nimg
        ema_halflife_nimg = ema_halflife_kimg * 1000
        if ema_rampup_ratio is not None:
            ema_halflife_nimg = min(ema_halflife_nimg, cur_nimg * ema_rampup_ratio)
        ema_beta = 0.5 ** (batch_size / max(ema_halflife_nimg, 1e-8))
        with torch.no_grad():
            for p_ema, p_net in zip(ema_psi.parameters(), psi_net.parameters()):
                p_ema.copy_(p_net.detach().lerp(p_ema, ema_beta))
            for p_ema, p_net in zip(ema_phase.parameters(), phase_net.parameters()):
                p_ema.copy_(p_net.detach().lerp(p_ema, ema_beta))
        cur_nimg += batch_size

    while True:
        # One forward pass, one loss, one backward, one optimizer step.
        # Joint and sequential nesting both follow this path; they differ only by the
        # struct mask inside KoopmanLoss (ones for joint, upper-triangular for sequential).
        optimizer.zero_grad(set_to_none=True)
        for round_idx in range(num_accumulation_rounds):
            images, labels = next(dataset_iterator)
            images = images.to(device).to(torch.float32) / 127.5 - 1
            labels = labels.to(device)
            loss = koop_loss(
                psi_net=psi_net,
                phase_net=phase_net,
                cfm_net=cfm_net,
                images=images,
                labels=labels,
                augment_pipe=augment_pipe,
            )
            training_stats.report('Loss/koopman', loss)
            loss.mul(loss_scaling / batch_gpu_total).backward()
        _allreduce_and_step()
        _ema_and_count()
        done = (cur_nimg >= total_kimg * 1000)
        if (not done) and (cur_tick != 0) and (cur_nimg < tick_start_nimg + kimg_per_tick * 1000):
            continue

        tick_end_time = time.time()
        fields = []
        fields += [f"tick {training_stats.report0('Progress/tick', cur_tick):<5d}"]
        fields += [f"kimg {training_stats.report0('Progress/kimg', cur_nimg / 1e3):<9.1f}"]
        fields += [f"time {dnnlib.util.format_time(training_stats.report0('Timing/total_sec', tick_end_time - start_time)):<12s}"]
        fields += [f"sec/tick {training_stats.report0('Timing/sec_per_tick', tick_end_time - tick_start_time):<7.1f}"]
        fields += [f"sec/kimg {training_stats.report0('Timing/sec_per_kimg', (tick_end_time - tick_start_time) / (cur_nimg - tick_start_nimg) * 1e3):<7.2f}"]
        fields += [f"maintenance {training_stats.report0('Timing/maintenance_sec', maintenance_time):<6.1f}"]
        fields += [f"cpumem {training_stats.report0('Resources/cpu_mem_gb', psutil.Process(os.getpid()).memory_info().rss / 2**30):<6.2f}"]
        fields += [f"gpumem {training_stats.report0('Resources/peak_gpu_mem_gb', torch.cuda.max_memory_allocated(device) / 2**30):<6.2f}"]
        fields += [f"reserved {training_stats.report0('Resources/peak_gpu_mem_reserved_gb', torch.cuda.max_memory_reserved(device) / 2**30):<6.2f}"]
        torch.cuda.reset_peak_memory_stats()
        dist.print0(' '.join(fields))

        if (not done) and dist.should_stop():
            done = True
            dist.print0()
            dist.print0('Aborting...')

        # Snapshot.
        if (snapshot_ticks is not None) and (done or cur_tick % snapshot_ticks == 0):
            # store EMA nets (recommended for inference)
            data = dict(
                psi_ema=copy.deepcopy(ema_psi).eval().requires_grad_(False),
                phase_ema=copy.deepcopy(ema_phase).eval().requires_grad_(False),
                # Also store non-EMA if you want:
                psi_net=copy.deepcopy(psi_net).eval().requires_grad_(False),
                phase_net=copy.deepcopy(phase_net).eval().requires_grad_(False),
                # record how we got the vector field:
                cfm_resume_pkl=cfm_resume_pkl,
                dataset_kwargs=dict(dataset_kwargs),
                psi_network_kwargs=dict(psi_network_kwargs),
                phase_kwargs=dict(phase_kwargs),
                koopman_loss_kwargs=dict(koopman_loss_kwargs),
            )
            # Check parameter consistency across ranks, then move to CPU for pickling.
            for k, v in list(data.items()):
                if isinstance(v, torch.nn.Module):
                    if dist.get_world_size() > 1:
                        misc.check_ddp_consistency(v)
                    data[k] = v.cpu()
                del v

            if dist.get_rank() == 0:
                snap_path = os.path.join(run_dir, f'koopman-snapshot-{cur_nimg//1000:06d}.pkl')
                with open(snap_path, 'wb') as f:
                    pickle.dump(data, f)

            del data

        # State dump (for resuming optimizer).
        if (state_dump_ticks is not None) and (done or cur_tick % state_dump_ticks == 0) and cur_tick != 0 and dist.get_rank() == 0:
            torch.save(
                dict(
                    psi_net=psi_net,
                    phase_net=phase_net,
                    optimizer_state=optimizer.state_dict(),
                ),
                os.path.join(run_dir, f'koopman-training-state-{cur_nimg//1000:06d}.pt')
            )

        training_stats.default_collector.update()
        if dist.get_rank() == 0:
            if stats_jsonl is None:
                stats_jsonl = open(os.path.join(run_dir, 'koopman-stats.jsonl'), 'at')
            stats_jsonl.write(json.dumps(dict(training_stats.default_collector.as_dict(), timestamp=time.time())) + '\n')
            stats_jsonl.flush()

            if wandb_project is not None:
                import wandb
                log_dict = {name: training_stats.default_collector.mean(name)
                            for name in training_stats.default_collector.names()}
                if hasattr(koop_loss, '_last_lam_mag'):
                    log_dict['Eigenvalues/lam_mag_hist'] = wandb.Histogram(
                        koop_loss._last_lam_mag.cpu().numpy()
                    )
                wandb.log(log_dict, step=cur_nimg)

        dist.update_progress(cur_nimg // 1000, total_kimg)

        cur_tick += 1
        tick_start_nimg = cur_nimg
        tick_start_time = time.time()
        maintenance_time = tick_start_time - tick_end_time

        if done:
            break

    if wandb_project is not None and dist.get_rank() == 0:
        import wandb
        wandb.finish()

    dist.print0()
    dist.print0('Exiting Koopman training...')

# ----------------------------------------------------------------------------