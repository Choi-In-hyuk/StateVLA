"""
StateVLA Two-Phase Training Script (Single-GPU & DDP)

Phase 1: Temporal JEPA (representation learning)
    python train.py --config conf/config.yaml --phase 1

Phase 2: Flow Matching (policy learning)
    python train.py --config conf/config.yaml --phase 2 --phase1_checkpoint checkpoints/phase1/checkpoint_best.pt

Multi-GPU (DDP):
    torchrun --nproc_per_node=2 train.py --config conf/config.yaml --phase 1
"""

import os
import argparse
import logging
import random
from datetime import datetime

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import yaml
from tqdm import tqdm

import sys
_project_root = os.path.dirname(os.path.abspath(__file__))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from statevla_model import StateVLA, StateVLATrainer
from dataloader import create_dataloader

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


# ==================== DDP Utilities ====================

def is_ddp():
    """Check if running in DDP mode."""
    return dist.is_initialized()


def get_rank():
    """Get current process rank (0 if not DDP)."""
    return dist.get_rank() if is_ddp() else 0


def get_world_size():
    """Get total number of processes (1 if not DDP)."""
    return dist.get_world_size() if is_ddp() else 1


def is_main_process():
    """Check if this is the main process (rank 0)."""
    return get_rank() == 0


def setup_ddp():
    """Initialize DDP if launched via torchrun."""
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
    """Clean up DDP."""
    if is_ddp():
        dist.destroy_process_group()


# ==================== Core Functions ====================

def set_seed(seed: int):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def save_checkpoint(
    model,
    optimizer,
    epoch: int,
    loss: float,
    config: dict,
    checkpoint_dir: str,
    is_best: bool = False,
    save_epoch_checkpoint: bool = False
):
    """Save model checkpoint (only on rank 0)."""
    if not is_main_process():
        return

    os.makedirs(checkpoint_dir, exist_ok=True)

    # Unwrap DDP model for saving
    model_to_save = model.module if isinstance(model, DDP) else model

    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model_to_save.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'config': config
    }

    # Save latest checkpoint (overwrite)
    path = os.path.join(checkpoint_dir, 'checkpoint_latest.pt')
    torch.save(checkpoint, path)

    # Save epoch checkpoint only if requested (to save disk space)
    if save_epoch_checkpoint:
        path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch}.pt')
        torch.save(checkpoint, path)

    # Save best checkpoint (overwrite)
    if is_best:
        path = os.path.join(checkpoint_dir, 'checkpoint_best.pt')
        torch.save(checkpoint, path)
        log.info(f"New best checkpoint at epoch {epoch} (loss: {loss:.4f})")
    else:
        log.info(f"Saved checkpoint at epoch {epoch}")


def train_epoch_phase1(
    trainer,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: str,
    epoch: int,
    config: dict,
    global_step: int = 0,
    total_steps: int = 0
):
    """Train Phase 1: Temporal JEPA."""
    trainer.train()
    total_loss = 0
    total_mse = 0
    total_var = 0
    total_cov = 0
    num_batches = 0
    current_step = global_step

    # Only show progress bar on main process
    pbar = tqdm(dataloader, desc=f"[Phase 1] Epoch {epoch}", disable=not is_main_process())

    for batch in pbar:
        # Move data to device
        obs = {k: v.to(device) for k, v in batch['obs'].items()}
        actions = batch['actions'].to(device)

        # Get next observation
        next_obs = batch.get('next_obs')
        if next_obs is None:
            continue
        next_obs = {k: v.to(device) for k, v in next_obs.items()}

        # Extract single action at time t (first action in sequence)
        a_t = actions[:, 0, :]  # [B, action_dim]

        # Forward pass
        optimizer.zero_grad()

        # Access underlying trainer for DDP-wrapped models
        trainer_module = trainer.module if isinstance(trainer, DDP) else trainer
        outputs = trainer_module(
            obs_dict=obs,
            next_obs_dict=next_obs,
            action=a_t,
            step=current_step,
            total_steps=total_steps,
        )

        loss = outputs['loss']

        # Backward pass
        loss.backward()

        # Gradient clipping
        if config['training'].get('gradient_clip', 0) > 0:
            torch.nn.utils.clip_grad_norm_(
                trainer.parameters(),
                config['training']['gradient_clip']
            )

        optimizer.step()

        # Update target encoder (EMA) - use unwrapped model
        trainer_module.update_target_encoder(current_step, total_steps)

        current_step += 1

        # Logging
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

    avg_loss = total_loss / max(num_batches, 1)
    avg_mse = total_mse / max(num_batches, 1)
    avg_var = total_var / max(num_batches, 1)
    avg_cov = total_cov / max(num_batches, 1)

    return {
        'loss': avg_loss,
        'jepa_mse': avg_mse,
        'jepa_variance': avg_var,
        'jepa_covariance': avg_cov,
        'global_step': current_step
    }


def train_epoch_phase2(
    trainer,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: str,
    epoch: int,
    config: dict,
    global_step: int = 0,
    total_steps: int = 0
):
    """Train Phase 2: Flow Matching."""
    trainer.train()
    total_loss = 0
    total_pos_rot_loss = 0
    total_gripper_loss = 0
    num_batches = 0
    current_step = global_step

    pbar = tqdm(dataloader, desc=f"[Phase 2] Epoch {epoch}", disable=not is_main_process())

    for batch in pbar:
        # Move data to device
        obs = {k: v.to(device) for k, v in batch['obs'].items()}
        actions = batch['actions'].to(device)

        # Forward pass
        optimizer.zero_grad()

        trainer_module = trainer.module if isinstance(trainer, DDP) else trainer
        outputs = trainer_module(
            obs_dict=obs,
            gt_actions=actions,
        )

        loss = outputs['loss']

        # Backward pass
        loss.backward()

        # Gradient clipping
        if config['training'].get('gradient_clip', 0) > 0:
            torch.nn.utils.clip_grad_norm_(
                trainer.parameters(),
                config['training']['gradient_clip']
            )

        optimizer.step()
        current_step += 1

        # Logging
        total_loss += loss.item()
        total_pos_rot_loss += outputs['pos_rot_loss'].item()
        total_gripper_loss += outputs['gripper_loss'].item()
        num_batches += 1

        if is_main_process():
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'pos_rot': f"{outputs['pos_rot_loss'].item():.4f}",
                'gripper': f"{outputs['gripper_loss'].item():.4f}",
            })

    avg_loss = total_loss / max(num_batches, 1)
    avg_pos_rot = total_pos_rot_loss / max(num_batches, 1)
    avg_gripper = total_gripper_loss / max(num_batches, 1)

    return {
        'loss': avg_loss,
        'pos_rot_loss': avg_pos_rot,
        'gripper_loss': avg_gripper,
        'global_step': current_step
    }


def main():
    parser = argparse.ArgumentParser(description='Train StateVLA (Two-Phase)')
    parser.add_argument('--config', type=str, default='conf/config.yaml',
                        help='Path to config file')
    parser.add_argument('--phase', type=int, default=1, choices=[1, 2],
                        help='Training phase (1=JEPA, 2=Flow Matching)')
    parser.add_argument('--phase1_checkpoint', type=str, default=None,
                        help='Phase 1 checkpoint for Phase 2 training')
    parser.add_argument('--data_directory', type=str, default=None,
                        help='Override data directory')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Resume from checkpoint (same phase)')
    parser.add_argument('--device', type=str, default=None,
                        help='Override device')
    parser.add_argument('--batch_size', type=int, default=None,
                        help='Override batch size')
    args = parser.parse_args()

    # Setup DDP (no-op if not launched via torchrun)
    local_rank = setup_ddp()

    # Load config
    config = load_config(args.config)

    # Override config with command line arguments
    if args.data_directory:
        config['data']['data_directory'] = args.data_directory
    if args.device:
        config['device'] = args.device
    if args.batch_size:
        config['training']['batch_size'] = args.batch_size

    # Setup device
    if is_ddp():
        device = f'cuda:{local_rank}'
    else:
        device = config.get('device', 'cuda')
        if device == 'cuda' and not torch.cuda.is_available():
            log.warning("CUDA not available, using CPU")
            device = 'cpu'

    # Set seed (offset by rank for different data per GPU)
    set_seed(config.get('seed', 42) + get_rank())

    phase = args.phase
    if is_main_process():
        log.info(f"=== Phase {phase} Training ===")
        log.info(f"Using device: {device}")
        if is_ddp():
            log.info(f"DDP: {get_world_size()} GPUs, batch_size per GPU = {config['training']['batch_size']}")
            log.info(f"Effective batch size = {config['training']['batch_size'] * get_world_size()}")

    # Get phase-specific config
    phase_config = config['training'].get(f'phase{phase}', {})

    # Create dataset and dataloader
    if is_main_process():
        log.info("Loading dataset...")

    dataloader, dataset = create_dataloader(
        data_directory=config['data']['data_directory'],
        batch_size=config['training']['batch_size'],
        action_dim=config['model']['action_dim'],
        action_seq_len=config['model']['action_seq_len'],
        demos_per_task=config['data']['demos_per_task'],
        max_len_data=config['data']['max_len_data'],
        image_size=config['model'].get('image_size', 224),
        camera_names=config['cameras']['names'],
    )

    # Replace dataloader with DDP sampler if needed
    if is_ddp():
        sampler = DistributedSampler(
            dataset,
            num_replicas=get_world_size(),
            rank=get_rank(),
            shuffle=True,
        )
        dataloader = DataLoader(
            dataset,
            batch_size=config['training']['batch_size'],
            sampler=sampler,
            num_workers=dataloader.num_workers,
            pin_memory=True,
            drop_last=True,
        )

    if is_main_process():
        log.info(f"Dataset size: {len(dataset)}")

    # Create model
    if is_main_process():
        log.info(f"Creating StateVLA model (Phase {phase})...")
    model_config = config['model']

    # Create model on CPU first, then move to device
    # (required for DDP: each rank must .to() its own GPU)
    init_device = 'cpu' if is_ddp() else device

    model = StateVLA(
        # Tokenizer config
        camera_names=config['cameras']['names'],
        image_size=model_config.get('image_size', 224),
        patch_size=model_config.get('patch_size', 16),
        embed_dim=model_config.get('embed_dim', 256),
        lang_emb_dim=model_config.get('lang_emb_dim', 512),
        robot_state_dim=model_config.get('robot_state_dim', 8),
        # Pretrained encoder config
        use_pretrained_vision=model_config.get('use_pretrained_vision', False),
        use_pretrained_language=model_config.get('use_pretrained_language', False),
        vision_model_name=model_config.get('vision_model_name', 'google/siglip-base-patch16-224'),
        language_model_name=model_config.get('language_model_name', 'ViT-B/32'),
        freeze_vision=model_config.get('freeze_vision', True),
        freeze_language=model_config.get('freeze_language', True),
        # Encoder config (Mamba)
        encoder_depth=model_config.get('encoder_depth', 12),
        d_state=model_config.get('d_state', 16),
        d_conv=model_config.get('d_conv', 4),
        expand=model_config.get('expand', 2),
        # Legacy predictor config
        predictor_embed_dim=model_config.get('predictor_embed_dim', 192),
        predictor_depth=model_config.get('predictor_depth', 6),
        mask_ratio=model_config.get('mask_ratio', 0.5),
        masking_strategy=model_config.get('masking_strategy', 'modality_aware'),
        # State config
        state_dim=model_config.get('state_dim', 256),
        # Action config
        action_dim=model_config.get('action_dim', 7),
        action_seq_len=model_config.get('action_seq_len', 10),
        # Policy config
        policy_layers=model_config.get('policy_layers', 3),
        policy_embed_dim=model_config.get('policy_embed_dim', 256),
        # Temporal predictor config
        temporal_hidden_dim=phase_config.get('temporal_predictor_hidden_dim', 512),
        # Training phase
        training_phase=phase,
        # Device: CPU for DDP init, actual device for single-GPU
        device=init_device,
    )

    # Create training wrapper
    training_config = config['training']
    trainer = StateVLATrainer(
        model=model,
        jepa_loss_weight=training_config.get('jepa_loss_weight', 1.0),
        action_loss_weight=training_config.get('action_loss_weight', 1.0),
        variance_weight=training_config.get('variance_weight', 1.0),
        covariance_weight=training_config.get('covariance_weight', 0.04),
        ema_momentum=training_config.get('ema_momentum', 0.996),
        ema_momentum_schedule=training_config.get('ema_momentum_schedule', 'cosine'),
    )

    trainer = trainer.to(device)

    # Set action normalization stats from dataset
    action_stats = dataset.get_action_stats()
    trainer.model.set_action_stats(
        action_stats['mean'].to(device),
        action_stats['std'].to(device)
    )

    # Phase 2: Load Phase 1 checkpoint and freeze encoder
    if phase == 2:
        phase1_ckpt_path = args.phase1_checkpoint or phase_config.get('phase1_checkpoint')
        if phase1_ckpt_path is None:
            log.error("Phase 2 requires --phase1_checkpoint or phase2.phase1_checkpoint in config")
            cleanup_ddp()
            return

        if is_main_process():
            log.info(f"Loading Phase 1 checkpoint: {phase1_ckpt_path}")
        phase1_ckpt = torch.load(phase1_ckpt_path, map_location=device)
        trainer.load_state_dict(phase1_ckpt['model_state_dict'], strict=False)
        if is_main_process():
            log.info("Phase 1 encoder weights loaded successfully")

        # Freeze encoder
        trainer.model.freeze_encoder()
        if is_main_process():
            log.info("Encoder frozen for Phase 2 training")

    # Count parameters
    total_params = sum(p.numel() for p in trainer.parameters())
    trainable_params = sum(p.numel() for p in trainer.parameters() if p.requires_grad)
    if is_main_process():
        log.info(f"Total parameters: {total_params / 1e6:.2f}M")
        log.info(f"Trainable parameters: {trainable_params / 1e6:.2f}M")

    # Wrap with DDP
    if is_ddp():
        trainer = DDP(trainer, device_ids=[local_rank], find_unused_parameters=True)
        if is_main_process():
            log.info("Model wrapped with DDP")

    # Create optimizer (only for trainable parameters)
    lr = phase_config.get('learning_rate', training_config['learning_rate'])
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, trainer.parameters()),
        lr=lr,
        weight_decay=training_config['weight_decay']
    )

    # Learning rate scheduler
    num_epochs = phase_config.get('num_epochs', training_config['num_epochs'])
    scheduler = None
    if training_config.get('use_lr_scheduler', False):
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=num_epochs,
            eta_min=training_config.get('min_lr', 1e-6)
        )

    # Resume from checkpoint (same phase)
    start_epoch = 0
    best_loss = float('inf')

    if args.checkpoint:
        if is_main_process():
            log.info(f"Resuming from checkpoint: {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint, map_location=device)

        # Load into unwrapped model
        model_to_load = trainer.module if isinstance(trainer, DDP) else trainer
        model_to_load.load_state_dict(checkpoint['model_state_dict'], strict=False)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_loss = checkpoint.get('loss', float('inf'))

        if is_main_process():
            log.info(f"Resumed from epoch {start_epoch}")

        # Re-freeze encoder if Phase 2
        if phase == 2:
            model_to_load.model.freeze_encoder()

    # Setup checkpoint directory
    checkpoint_dir = training_config.get('checkpoint_dir', 'checkpoints')
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    checkpoint_dir = os.path.join(checkpoint_dir, f'phase{phase}_{timestamp}')

    # Training loop
    if is_main_process():
        log.info(f"Starting Phase {phase} training...")
    steps_per_epoch = len(dataloader)
    total_steps = num_epochs * steps_per_epoch
    global_step = start_epoch * steps_per_epoch

    if is_main_process():
        log.info(f"Epochs: {num_epochs}, Steps/epoch: {steps_per_epoch}, Total steps: {total_steps}")

    # Select phase-specific training function
    train_fn = train_epoch_phase1 if phase == 1 else train_epoch_phase2

    for epoch in range(start_epoch, num_epochs):
        # Set epoch for DDP sampler (ensures proper shuffling)
        if is_ddp():
            dataloader.sampler.set_epoch(epoch)

        # Train
        train_metrics = train_fn(
            trainer=trainer,
            dataloader=dataloader,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            config=config,
            global_step=global_step,
            total_steps=total_steps
        )

        # Update global step
        global_step = train_metrics.get('global_step', global_step + steps_per_epoch)

        # Update learning rate
        if scheduler is not None:
            scheduler.step()

        # Log metrics (main process only)
        if is_main_process():
            if phase == 1:
                log.info(
                    f"Epoch {epoch}: loss={train_metrics['loss']:.4f}, "
                    f"mse={train_metrics['jepa_mse']:.4f}, "
                    f"var={train_metrics['jepa_variance']:.4f}, "
                    f"cov={train_metrics['jepa_covariance']:.4f}"
                )
            else:
                log.info(
                    f"Epoch {epoch}: loss={train_metrics['loss']:.4f}, "
                    f"pos_rot={train_metrics['pos_rot_loss']:.4f}, "
                    f"gripper={train_metrics['gripper_loss']:.4f}"
                )

            # Save checkpoint (main process only)
            is_best = train_metrics['loss'] < best_loss
            if is_best:
                best_loss = train_metrics['loss']

            if (epoch + 1) % training_config.get('save_interval', 100) == 0 or is_best:
                save_checkpoint(
                    model=trainer,
                    optimizer=optimizer,
                    epoch=epoch,
                    loss=train_metrics['loss'],
                    config=config,
                    checkpoint_dir=checkpoint_dir,
                    is_best=is_best
                )

        # Synchronize all processes before next epoch
        if is_ddp():
            dist.barrier()

    if is_main_process():
        log.info(f"Phase {phase} training complete!")
        log.info(f"Best loss: {best_loss:.4f}")
        log.info(f"Checkpoints saved to: {checkpoint_dir}")

        if phase == 1:
            log.info(
                f"\nNext step: Run Phase 2 with:\n"
                f"  python train.py --config {args.config} --phase 2 "
                f"--phase1_checkpoint {checkpoint_dir}/checkpoint_best.pt"
            )

    cleanup_ddp()


if __name__ == '__main__':
    main()
