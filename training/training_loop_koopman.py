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
    kimg_per_tick       = 50,
    snapshot_ticks      = 50,
    state_dump_ticks    = 500,

    resume_state_dump   = None,
    resume_kimg         = 0,
    cudnn_benchmark     = True,
    device              = torch.device('cuda'),
):
    # ------------------------------------------------------------------------
    # Init.
    # ------------------------------------------------------------------------
    start_time = time.time()
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

    phase_net = dnnlib.util.construct_class_by_name(**phase_kwargs)  # expects k inside kwargs or class default
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

    ddp_psi = torch.nn.parallel.DistributedDataParallel(psi_net, device_ids=[device], broadcast_buffers=False)
    # phase_net is tiny; DDP is optional. Keeping it DDP avoids rank drift.
    ddp_phase = torch.nn.parallel.DistributedDataParallel(phase_net, device_ids=[device], broadcast_buffers=False)

    ema_psi = copy.deepcopy(psi_net).eval().requires_grad_(False)
    ema_phase = copy.deepcopy(phase_net).eval().requires_grad_(False)

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

    while True:
        optimizer.zero_grad(set_to_none=True)

        for round_idx in range(num_accumulation_rounds):
            with misc.ddp_sync(ddp_psi, (round_idx == num_accumulation_rounds - 1)):
                with misc.ddp_sync(ddp_phase, (round_idx == num_accumulation_rounds - 1)):
                    images, labels = next(dataset_iterator)
                    images = images.to(device).to(torch.float32) / 127.5 - 1
                    labels = labels.to(device)

                    # IMPORTANT: CFM is frozen; KoopmanLoss already uses no_grad() around xdot.
                    loss = koop_loss(
                        psi_net=ddp_psi,
                        phase_net=ddp_phase,
                        cfm_net=cfm_net,
                        images=images,
                        labels=labels,
                        augment_pipe=augment_pipe,
                    )

                    training_stats.report('Loss/koopman', loss)
                    loss.mul(loss_scaling / batch_gpu_total).backward()

        # LR ramp.
        for g in optimizer.param_groups:
            base_lr = optimizer_kwargs.get('lr', g.get('lr', 1e-4))
            g['lr'] = base_lr * min(cur_nimg / max(lr_rampup_kimg * 1000, 1e-8), 1.0)

        # NaN guards.
        for p in train_params:
            if p.grad is not None:
                torch.nan_to_num(p.grad, nan=0, posinf=1e5, neginf=-1e5, out=p.grad)

        optimizer.step()

        # EMA update.
        ema_halflife_nimg = ema_halflife_kimg * 1000
        if ema_rampup_ratio is not None:
            ema_halflife_nimg = min(ema_halflife_nimg, cur_nimg * ema_rampup_ratio)
        ema_beta = 0.5 ** (batch_size / max(ema_halflife_nimg, 1e-8))

        with torch.no_grad():
            for p_ema, p_net in zip(ema_psi.parameters(), psi_net.parameters()):
                p_ema.copy_(p_net.detach().lerp(p_ema, ema_beta))
            for p_ema, p_net in zip(ema_phase.parameters(), phase_net.parameters()):
                p_ema.copy_(p_net.detach().lerp(p_ema, ema_beta))

        # Tick / logging.
        cur_nimg += batch_size
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
                psi_ema=copy.deepcopy(ema_psi).eval().requires_grad_(False).cpu(),
                phase_ema=copy.deepcopy(ema_phase).eval().requires_grad_(False).cpu(),
                # Also store non-EMA if you want:
                psi_net=copy.deepcopy(psi_net).eval().requires_grad_(False).cpu(),
                phase_net=copy.deepcopy(phase_net).eval().requires_grad_(False).cpu(),
                # record how we got the vector field:
                cfm_resume_pkl=cfm_resume_pkl,
                dataset_kwargs=dict(dataset_kwargs),
                psi_network_kwargs=dict(psi_network_kwargs),
                phase_kwargs=dict(phase_kwargs),
                koopman_loss_kwargs=dict(koopman_loss_kwargs),
            )
            # DDP consistency checks on modules we saved.
            misc.check_ddp_consistency(data['psi_ema'])
            misc.check_ddp_consistency(data['psi_net'])
            # phase nets are tiny but still modules.
            misc.check_ddp_consistency(data['phase_ema'])
            misc.check_ddp_consistency(data['phase_net'])

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

        dist.update_progress(cur_nimg // 1000, total_kimg)

        cur_tick += 1
        tick_start_nimg = cur_nimg
        tick_start_time = time.time()
        maintenance_time = tick_start_time - tick_end_time

        if done:
            break

    dist.print0()
    dist.print0('Exiting Koopman training...')

# ----------------------------------------------------------------------------