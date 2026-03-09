"""
StateVLA Two-Phase Training Script (Single-GPU & DDP)

Phase 1: Temporal JEPA (representation learning)
    python train.py --config conf/config.yaml --phase 1

Phase 2: Flow Matching (policy learning)
    python train.py --config conf/config.yaml --phase 2 --phase1_checkpoint checkpoints/phase1/checkpoint_best.pt

Resume training:
    python train.py --config conf/config.yaml --phase 2 --phase1_checkpoint ... --checkpoint checkpoints/phase2_xxx/checkpoint_latest.pt

Multi-GPU (DDP):
    torchrun --nproc_per_node=2 train.py --config conf/config.yaml --phase 1
"""

import os
import math
import argparse
import logging
import random
from datetime import datetime

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler
import yaml
from tqdm import tqdm

import sys
_project_root = os.path.dirname(os.path.abspath(__file__))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from statevla_model import StateVLA, StateVLATrainer
from dataloader import create_dataloader, collate_fn

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


# ==================== DDP Utilities ====================

def is_ddp():
    return dist.is_initialized()

def get_rank():
    return dist.get_rank() if is_ddp() else 0

def get_world_size():
    return dist.get_world_size() if is_ddp() else 1

def is_main_process():
    return get_rank() == 0

def setup_ddp():
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend='nccl', rank=rank, world_size=world_size)
        if rank == 0:
            log.info(f"DDP initialized: {world_size} GPUs")
        return local_rank
    return 0

def cleanup_ddp():
    if is_ddp():
        dist.destroy_process_group()


# ==================== Core Functions ====================

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_config(config_path: str) -> dict:
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def save_checkpoint(
    model,
    optimizer,
    scheduler,
    epoch: int,
    loss: float,
    config: dict,
    checkpoint_dir: str,
    is_best: bool = False,
    save_epoch_checkpoint: bool = False,
    best_train_loss: float = float('inf'),
    val_loss: float = None,
    best_val_loss: float = float('inf'),
):
    """Save model checkpoint with scheduler state (only on rank 0)."""
    if not is_main_process():
        return

    os.makedirs(checkpoint_dir, exist_ok=True)

    model_to_save = model.module if isinstance(model, DDP) else model

    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model_to_save.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler is not None else None,
        'loss': loss,
        'best_train_loss': best_train_loss,
        'val_loss': val_loss,
        'best_val_loss': best_val_loss,
        'config': config,
    }

    path = os.path.join(checkpoint_dir, 'checkpoint_latest.pt')
    torch.save(checkpoint, path)

    if is_best:
        path = os.path.join(checkpoint_dir, 'checkpoint_best.pt')
        torch.save(checkpoint, path)
        log.info(f"New best checkpoint at epoch {epoch} (val_loss: {val_loss:.4f})")

    if save_epoch_checkpoint:
        path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch}.pt')
        torch.save(checkpoint, path)
        log.info(f"Saved epoch checkpoint at epoch {epoch}")

    if not is_best and not save_epoch_checkpoint:
        log.info(f"Saved checkpoint at epoch {epoch}")


# ==================== Train/Val Epoch Functions ====================

def train_epoch_phase1(trainer, dataloader, optimizer, device, epoch, config, global_step=0, total_steps=0):
    """Train Phase 1: Temporal JEPA."""
    trainer.train()
    total_loss = total_mse = total_var = total_cov = 0
    num_batches = 0
    current_step = global_step

    pbar = tqdm(dataloader, desc=f"[Phase 1] Train Epoch {epoch}", disable=not is_main_process())

    for batch in pbar:
        obs = {k: v.to(device) for k, v in batch['obs'].items()}
        actions = batch['actions'].to(device)

        next_obs = batch.get('next_obs')
        if next_obs is None:
            continue
        next_obs = {k: v.to(device) for k, v in next_obs.items()}

        a_t = actions[:, 0, :]

        optimizer.zero_grad()
        trainer_module = trainer.module if isinstance(trainer, DDP) else trainer
        outputs = trainer_module(
            obs_dict=obs,
            next_obs_dict=next_obs,
            action=a_t,
            step=current_step,
            total_steps=total_steps,
        )

        loss = outputs['loss']
        loss.backward()

        if config['training'].get('gradient_clip', 0) > 0:
            torch.nn.utils.clip_grad_norm_(trainer.parameters(), config['training']['gradient_clip'])

        optimizer.step()
        trainer_module.update_target_encoder(current_step, total_steps)
        current_step += 1

        total_loss += loss.item()
        total_mse += outputs['jepa_mse'].item()
        total_var += outputs['jepa_variance'].item()
        total_cov += outputs['jepa_covariance'].item()
        num_batches += 1

        if is_main_process():
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'mse': f"{outputs['jepa_mse'].item():.4f}",
                'var': f"{outputs['jepa_variance'].item():.4f}",
            })

    n = max(num_batches, 1)
    return {
        'loss': total_loss / n,
        'jepa_mse': total_mse / n,
        'jepa_variance': total_var / n,
        'jepa_covariance': total_cov / n,
        'global_step': current_step,
    }


@torch.no_grad()
def val_epoch_phase1(trainer, dataloader, device):
    """Validate Phase 1: compute loss without gradient updates."""
    trainer.eval()
    total_loss = 0
    num_batches = 0

    for batch in tqdm(dataloader, desc="[Phase 1] Val", disable=not is_main_process()):
        obs = {k: v.to(device) for k, v in batch['obs'].items()}
        actions = batch['actions'].to(device)

        next_obs = batch.get('next_obs')
        if next_obs is None:
            continue
        next_obs = {k: v.to(device) for k, v in next_obs.items()}

        a_t = actions[:, 0, :]
        trainer_module = trainer.module if isinstance(trainer, DDP) else trainer
        outputs = trainer_module(
            obs_dict=obs,
            next_obs_dict=next_obs,
            action=a_t,
            step=0,
            total_steps=1,
        )

        total_loss += outputs['loss'].item()
        num_batches += 1

    val_loss = total_loss / max(num_batches, 1)

    # Aggregate across DDP ranks
    if is_ddp():
        val_loss_tensor = torch.tensor(val_loss, device=device)
        dist.all_reduce(val_loss_tensor, op=dist.ReduceOp.AVG)
        val_loss = val_loss_tensor.item()

    return val_loss


def train_epoch_phase2(trainer, dataloader, optimizer, device, epoch, config, global_step=0, total_steps=0):
    """Train Phase 2: Flow Matching."""
    trainer.train()
    total_loss = total_pos_rot = total_gripper = total_goal = 0
    num_batches = 0
    current_step = global_step

    pbar = tqdm(dataloader, desc=f"[Phase 2] Train Epoch {epoch}", disable=not is_main_process())

    for batch in pbar:
        obs = {k: v.to(device) for k, v in batch['obs'].items()}
        actions = batch['actions'].to(device)

        goal_obs = batch.get('goal_obs')
        if goal_obs is not None:
            goal_obs = {k: v.to(device) for k, v in goal_obs.items()}

        optimizer.zero_grad()
        trainer_module = trainer.module if isinstance(trainer, DDP) else trainer
        outputs = trainer_module(obs_dict=obs, gt_actions=actions, goal_obs_dict=goal_obs)

        loss = outputs['loss']
        loss.backward()

        if config['training'].get('gradient_clip', 0) > 0:
            torch.nn.utils.clip_grad_norm_(trainer.parameters(), config['training']['gradient_clip'])

        optimizer.step()
        current_step += 1

        total_loss += loss.item()
        total_pos_rot += outputs['pos_rot_loss'].item()
        total_gripper += outputs['gripper_loss'].item()
        total_goal += outputs.get('goal_loss', torch.tensor(0.0)).item()
        num_batches += 1

        if is_main_process():
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'flow7d': f"{outputs['pos_rot_loss'].item():.4f}",
                'goal': f"{outputs.get('goal_loss', torch.tensor(0.0)).item():.4f}",
            })

    n = max(num_batches, 1)
    return {
        'loss': total_loss / n,
        'pos_rot_loss': total_pos_rot / n,
        'gripper_loss': total_gripper / n,
        'goal_loss': total_goal / n,
        'global_step': current_step,
    }


@torch.no_grad()
def val_epoch_phase2(trainer, dataloader, device):
    """Validate Phase 2: compute flow matching loss without gradient updates."""
    trainer.eval()
    total_loss = 0
    num_batches = 0

    for batch in tqdm(dataloader, desc="[Phase 2] Val", disable=not is_main_process()):
        obs = {k: v.to(device) for k, v in batch['obs'].items()}
        actions = batch['actions'].to(device)

        goal_obs = batch.get('goal_obs')
        if goal_obs is not None:
            goal_obs = {k: v.to(device) for k, v in goal_obs.items()}

        trainer_module = trainer.module if isinstance(trainer, DDP) else trainer
        outputs = trainer_module(obs_dict=obs, gt_actions=actions, goal_obs_dict=goal_obs)

        total_loss += outputs['loss'].item()
        num_batches += 1

    val_loss = total_loss / max(num_batches, 1)

    # Aggregate across DDP ranks
    if is_ddp():
        val_loss_tensor = torch.tensor(val_loss, device=device)
        dist.all_reduce(val_loss_tensor, op=dist.ReduceOp.AVG)
        val_loss = val_loss_tensor.item()

    return val_loss


# ==================== Main ====================

def main():
    parser = argparse.ArgumentParser(description='Train StateVLA (Two-Phase)')
    parser.add_argument('--config', type=str, default='conf/config.yaml')
    parser.add_argument('--phase', type=int, default=1, choices=[1, 2])
    parser.add_argument('--phase1_checkpoint', type=str, default=None,
                        help='Phase 1 checkpoint for Phase 2 training')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Resume from checkpoint (same phase)')
    parser.add_argument('--data_directory', type=str, default=None)
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--reset_optimizer', action='store_true',
                        help='Reset optimizer state (useful when resuming with different LR)')
    args = parser.parse_args()

    local_rank = setup_ddp()
    config = load_config(args.config)

    if args.data_directory:
        config['data']['data_directory'] = args.data_directory
    if args.device:
        config['device'] = args.device
    if args.batch_size:
        config['training']['batch_size'] = args.batch_size

    if is_ddp():
        device = f'cuda:{local_rank}'
    else:
        device = config.get('device', 'cuda')
        if device == 'cuda' and not torch.cuda.is_available():
            log.warning("CUDA not available, using CPU")
            device = 'cpu'

    set_seed(config.get('seed', 42) + get_rank())

    phase = args.phase
    training_config = config['training']
    phase_config = training_config.get(f'phase{phase}', {})

    if is_main_process():
        log.info(f"=== Phase {phase} Training ===")
        log.info(f"Device: {device}")

    # ── Dataset ──────────────────────────────────────────────────────────────
    if is_main_process():
        log.info("Loading dataset...")

    _, dataset = create_dataloader(
        data_directory=config['data']['data_directory'],
        batch_size=1,           # placeholder; we'll make our own loaders below
        action_dim=config['model']['action_dim'],
        action_seq_len=config['model']['action_seq_len'],
        demos_per_task=config['data']['demos_per_task'],
        max_len_data=config['data']['max_len_data'],
        image_size=config['model'].get('image_size', 224),
        camera_names=config['cameras']['names'],
        language_embedding_path=config['data'].get('language_embedding_path', None),
    )

    # Train/val split
    val_ratio = training_config.get('val_ratio', 0.1)
    n_total = len(dataset)
    n_val = int(n_total * val_ratio)
    n_train = n_total - n_val
    all_indices = list(range(n_total))
    train_dataset = Subset(dataset, all_indices[:n_train])
    val_dataset = Subset(dataset, all_indices[n_train:])

    if is_main_process():
        log.info(f"Dataset: {n_total} total → {n_train} train / {n_val} val")

    batch_size = config['training']['batch_size']
    num_workers = 4

    if is_ddp():
        train_sampler = DistributedSampler(train_dataset, num_replicas=get_world_size(),
                                           rank=get_rank(), shuffle=True)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler,
                                  num_workers=num_workers, pin_memory=True,
                                  drop_last=True, collate_fn=collate_fn)
        val_sampler = DistributedSampler(val_dataset, num_replicas=get_world_size(),
                                         rank=get_rank(), shuffle=False)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, sampler=val_sampler,
                                num_workers=num_workers, pin_memory=True,
                                drop_last=False, collate_fn=collate_fn)
    else:
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                  num_workers=num_workers, pin_memory=True,
                                  drop_last=True, collate_fn=collate_fn)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                                num_workers=num_workers, pin_memory=True,
                                drop_last=False, collate_fn=collate_fn)

    # ── Model ─────────────────────────────────────────────────────────────────
    if is_main_process():
        log.info(f"Creating StateVLA model (Phase {phase})...")

    model_config = config['model']
    init_device = 'cpu' if is_ddp() else device

    model = StateVLA(
        camera_names=config['cameras']['names'],
        image_size=model_config.get('image_size', 224),
        patch_size=model_config.get('patch_size', 16),
        embed_dim=model_config.get('embed_dim', 256),
        lang_emb_dim=model_config.get('lang_emb_dim', 512),
        robot_state_dim=model_config.get('robot_state_dim', 9),
        use_pretrained_vision=model_config.get('use_pretrained_vision', False),
        use_pretrained_language=model_config.get('use_pretrained_language', False),
        vision_model_name=model_config.get('vision_model_name', 'google/siglip-base-patch16-224'),
        language_model_name=model_config.get('language_model_name', 'ViT-B/32'),
        freeze_vision=model_config.get('freeze_vision', True),
        freeze_language=model_config.get('freeze_language', True),
        encoder_depth=model_config.get('encoder_depth', 12),
        d_state=model_config.get('d_state', 16),
        d_conv=model_config.get('d_conv', 4),
        expand=model_config.get('expand', 2),
        predictor_embed_dim=model_config.get('predictor_embed_dim', 192),
        predictor_depth=model_config.get('predictor_depth', 6),
        mask_ratio=model_config.get('mask_ratio', 0.5),
        masking_strategy=model_config.get('masking_strategy', 'modality_aware'),
        state_dim=model_config.get('state_dim', 256),
        action_dim=model_config.get('action_dim', 7),
        action_seq_len=model_config.get('action_seq_len', 10),
        policy_layers=model_config.get('policy_layers', 3),
        policy_embed_dim=model_config.get('policy_embed_dim', 256),
        temporal_hidden_dim=phase_config.get('temporal_predictor_hidden_dim', 512),
        goal_predictor_hidden_dim=model_config.get('goal_predictor_hidden_dim', 512),
        training_phase=phase,
        device=init_device,
    )

    trainer = StateVLATrainer(
        model=model,
        jepa_loss_weight=training_config.get('jepa_loss_weight', 1.0),
        action_loss_weight=training_config.get('action_loss_weight', 1.0),
        variance_weight=training_config.get('variance_weight', 1.0),
        covariance_weight=training_config.get('covariance_weight', 0.04),
        ema_momentum=training_config.get('ema_momentum', 0.996),
        ema_momentum_schedule=training_config.get('ema_momentum_schedule', 'cosine'),
        world_model_loss_weight=training_config.get('world_model_loss_weight', 0.0),
        goal_loss_weight=training_config.get('goal_loss_weight', 0.1),
    )
    trainer = trainer.to(device)

    # Action normalization stats (computed from full dataset - no val split)
    action_stats = dataset.get_action_stats()
    trainer.model.set_action_stats(
        action_stats['min'].to(device),
        action_stats['max'].to(device),
    )

    # ── Phase 2: load Phase 1 encoder ────────────────────────────────────────
    if phase == 2:
        phase1_ckpt_path = args.phase1_checkpoint or phase_config.get('phase1_checkpoint')
        if phase1_ckpt_path is None and args.checkpoint is None:
            log.error("Phase 2 requires --phase1_checkpoint (or --checkpoint to resume)")
            cleanup_ddp()
            return

        if phase1_ckpt_path is not None:
            if is_main_process():
                log.info(f"Loading Phase 1 checkpoint: {phase1_ckpt_path}")
            phase1_ckpt = torch.load(phase1_ckpt_path, map_location=device)

            current_state = trainer.state_dict()
            filtered = {k: v for k, v in phase1_ckpt['model_state_dict'].items()
                        if not (k in current_state and current_state[k].shape != v.shape)}
            trainer.load_state_dict(filtered, strict=False)
            trainer.model.freeze_encoder()
            if is_main_process():
                log.info("Phase 1 encoder loaded and frozen")
        else:
            # Freeze encoder before optimizer creation so param groups match saved checkpoint
            trainer.model.freeze_encoder()
            if is_main_process():
                log.info("Resuming Phase 2 — encoder frozen, weights will be loaded from Phase 2 checkpoint")

    total_params = sum(p.numel() for p in trainer.parameters())
    trainable_params = sum(p.numel() for p in trainer.parameters() if p.requires_grad)
    if is_main_process():
        log.info(f"Parameters: {total_params/1e6:.2f}M total, {trainable_params/1e6:.2f}M trainable")

    # ── DDP wrap ──────────────────────────────────────────────────────────────
    if is_ddp():
        trainer = DDP(trainer, device_ids=[local_rank], find_unused_parameters=True)

    # ── Optimizer & Scheduler ─────────────────────────────────────────────────
    num_epochs = phase_config.get('num_epochs', training_config['num_epochs'])
    lr = phase_config.get('learning_rate', training_config['learning_rate'])

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, trainer.parameters()),
        lr=lr,
        weight_decay=training_config['weight_decay'],
    )

    scheduler = None
    if training_config.get('use_lr_scheduler', False):
        scheduler_epochs = phase_config.get('scheduler_epochs', num_epochs)
        min_lr = training_config.get('min_lr', 1e-6)
        lr_ratio = min_lr / lr  # eta_min / lr_init

        def cosine_no_cycle(epoch):
            if epoch >= scheduler_epochs:
                return lr_ratio
            return lr_ratio + 0.5 * (1.0 - lr_ratio) * (
                1.0 + math.cos(math.pi * epoch / scheduler_epochs)
            )

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=cosine_no_cycle)

    # ── Resume from checkpoint ────────────────────────────────────────────────
    start_epoch = 0
    best_train_loss = float('inf')
    best_val_loss = float('inf')

    if args.checkpoint:
        if is_main_process():
            log.info(f"Resuming from: {args.checkpoint}")
        ckpt = torch.load(args.checkpoint, map_location=device)

        model_to_load = trainer.module if isinstance(trainer, DDP) else trainer
        current_state = model_to_load.state_dict()
        filtered = {k: v for k, v in ckpt['model_state_dict'].items()
                    if not (k in current_state and current_state[k].shape != v.shape)}
        model_to_load.load_state_dict(filtered, strict=False)
        if args.reset_optimizer:
            if is_main_process():
                log.info("Optimizer state reset (--reset_optimizer): starting fresh with new LR")
        else:
            try:
                optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            except (ValueError, RuntimeError) as e:
                if is_main_process():
                    log.warning(f"Optimizer state not loaded (architecture changed): {e}")
                    log.warning("Starting optimizer from scratch — LR will be restored via scheduler")

        # Restore scheduler state (correct LR position after resume)
        if scheduler is not None:
            loaded = False
            saved_sched = ckpt.get('scheduler_state_dict')
            if saved_sched is not None:
                # If T_max changed, skip loading to start a fresh cosine schedule
                saved_T_max = saved_sched.get('T_max')
                current_T_max = getattr(scheduler, 'T_max', None)
                if saved_T_max is not None and saved_T_max != current_T_max:
                    if is_main_process():
                        log.info(f"Scheduler T_max changed ({saved_T_max} → {current_T_max}) — starting fresh LR schedule")
                else:
                    try:
                        scheduler.load_state_dict(saved_sched)
                        loaded = True
                        if is_main_process():
                            log.info("Scheduler state restored from checkpoint")
                    except (KeyError, ValueError):
                        if is_main_process():
                            log.info("Scheduler type changed — starting fresh LR schedule")
            if not loaded:
                if is_main_process():
                    log.info(f"Fresh LR schedule: {scheduler.get_last_lr()} over {getattr(scheduler, 'T_max', '?')} epochs")

        start_epoch = ckpt['epoch'] + 1
        best_train_loss = ckpt.get('best_train_loss', float('inf'))
        best_val_loss = ckpt.get('best_val_loss', float('inf'))

        if phase == 2:
            model_to_load.model.freeze_encoder()

        if is_main_process():
            log.info(f"Resumed from epoch {start_epoch}, best_train_loss={best_train_loss:.4f}, best_val_loss={best_val_loss:.4f}")
            if scheduler is not None:
                log.info(f"Current LR: {scheduler.get_last_lr()[0]:.2e}")

    # ── Checkpoint directory ──────────────────────────────────────────────────
    checkpoint_dir = training_config.get('checkpoint_dir', 'checkpoints')
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    checkpoint_dir = os.path.join(checkpoint_dir, f'phase{phase}_{timestamp}')

    # ── Training loop ─────────────────────────────────────────────────────────
    steps_per_epoch = len(train_loader)
    total_steps = num_epochs * steps_per_epoch
    global_step = start_epoch * steps_per_epoch
    save_interval = training_config.get('save_interval', 200)
    val_interval = training_config.get('val_interval', 10)

    train_fn = train_epoch_phase1 if phase == 1 else train_epoch_phase2
    val_fn = val_epoch_phase1 if phase == 1 else val_epoch_phase2

    if is_main_process():
        log.info(f"Epochs: {num_epochs}, steps/epoch: {steps_per_epoch}, total: {total_steps}")
        log.info(f"Validation every {val_interval} epochs, epoch checkpoint every {save_interval} epochs")

    for epoch in range(start_epoch, num_epochs):
        if is_ddp():
            train_loader.sampler.set_epoch(epoch)

        # Train
        train_metrics = train_fn(
            trainer=trainer,
            dataloader=train_loader,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            config=config,
            global_step=global_step,
            total_steps=total_steps,
        )
        global_step = train_metrics.get('global_step', global_step + steps_per_epoch)

        if scheduler is not None:
            scheduler.step()

        # Validate (all ranks must participate for DDP all_reduce)
        val_loss = None
        if (epoch + 1) % val_interval == 0:
            val_loss = val_fn(trainer, val_loader, device)

        # ── Logging & Checkpoint (main process only) ──────────────────────────
        if is_main_process():
            current_lr = scheduler.get_last_lr()[0] if scheduler else lr
            if phase == 1:
                log_msg = (
                    f"Epoch {epoch} | train_loss={train_metrics['loss']:.4f} "
                    f"mse={train_metrics['jepa_mse']:.4f} "
                    f"var={train_metrics['jepa_variance']:.4f} "
                    f"lr={current_lr:.2e}"
                )
            else:
                log_msg = (
                    f"Epoch {epoch} | train_loss={train_metrics['loss']:.4f} "
                    f"flow7d={train_metrics['pos_rot_loss']:.4f} "
                    f"goal={train_metrics.get('goal_loss', 0.0):.4f} "
                    f"lr={current_lr:.2e}"
                )
            if val_loss is not None:
                log_msg += f" | val_loss={val_loss:.4f}"
            log.info(log_msg)

            save_epoch_ckpt = (epoch + 1) % save_interval == 0
            # Best checkpoint based on val_loss (only updated on val epochs)
            is_best = val_loss is not None and val_loss < best_val_loss
            if is_best:
                best_val_loss = val_loss
            # Track train loss separately
            if train_metrics['loss'] < best_train_loss:
                best_train_loss = train_metrics['loss']
            save_checkpoint(
                model=trainer, optimizer=optimizer, scheduler=scheduler,
                epoch=epoch, loss=train_metrics['loss'], config=config,
                checkpoint_dir=checkpoint_dir,
                is_best=is_best,
                save_epoch_checkpoint=save_epoch_ckpt,
                best_train_loss=best_train_loss,
                val_loss=val_loss,
                best_val_loss=best_val_loss,
            )

        if is_ddp():
            dist.barrier()

    if is_main_process():
        log.info(f"Phase {phase} training complete!")
        log.info(f"Checkpoints saved to: {checkpoint_dir}")
        if phase == 1:
            log.info(
                f"\nNext step:\n"
                f"  python train.py --config {args.config} --phase 2 "
                f"--phase1_checkpoint {checkpoint_dir}/checkpoint_latest.pt"
            )

    cleanup_ddp()


if __name__ == '__main__':
    main()
