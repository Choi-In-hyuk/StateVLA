"""
StateVLA JEPA Training Script

Usage:
    python train.py --config conf/config.yaml
    python train.py --config conf/config.yaml --data_directory /path/to/data
"""

import os
import argparse
import logging
import random
from datetime import datetime

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

import sys
_project_root = os.path.dirname(os.path.abspath(__file__))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from statevla_model import StateVLA, StateVLATrainer
from dataloader import create_dataloader

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


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
    """Save model checkpoint."""
    os.makedirs(checkpoint_dir, exist_ok=True)

    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
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


def train_epoch(
    trainer: StateVLATrainer,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: str,
    epoch: int,
    config: dict,
    global_step: int = 0,
    total_steps: int = 0
):
    """Train for one epoch."""
    trainer.train()
    total_loss = 0
    total_action_loss = 0
    total_jepa_loss = 0
    num_batches = 0
    current_step = global_step

    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")

    for batch in pbar:
        # Move data to device
        obs = {k: v.to(device) for k, v in batch['obs'].items()}
        actions = batch['actions'].to(device)

        # Forward pass
        optimizer.zero_grad()
        outputs = trainer(
            obs_dict=obs,
            gt_actions=actions,
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

        # Update target encoder (EMA)
        trainer.update_target_encoder(current_step, total_steps)

        current_step += 1

        # Logging
        total_loss += loss.item()
        total_action_loss += outputs['action_loss'].item()
        total_jepa_loss += outputs['jepa_loss'].item()
        num_batches += 1

        pbar.set_postfix({
            'loss': f"{loss.item():.4f}",
            'act': f"{outputs['action_loss'].item():.4f}",
            'jepa': f"{outputs['jepa_loss'].item():.4f}"
        })

    avg_loss = total_loss / num_batches
    avg_action_loss = total_action_loss / num_batches
    avg_jepa_loss = total_jepa_loss / num_batches

    return {
        'loss': avg_loss,
        'action_loss': avg_action_loss,
        'jepa_loss': avg_jepa_loss,
        'global_step': current_step
    }


def main():
    parser = argparse.ArgumentParser(description='Train StateVLA with JEPA')
    parser.add_argument('--config', type=str, default='conf/config.yaml',
                        help='Path to config file')
    parser.add_argument('--data_directory', type=str, default=None,
                        help='Override data directory')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Resume from checkpoint')
    parser.add_argument('--device', type=str, default=None,
                        help='Override device')
    parser.add_argument('--batch_size', type=int, default=None,
                        help='Override batch size')
    args = parser.parse_args()

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
    device = config.get('device', 'cuda')
    if device == 'cuda' and not torch.cuda.is_available():
        log.warning("CUDA not available, using CPU")
        device = 'cpu'

    # Set seed
    set_seed(config.get('seed', 42))

    log.info(f"Using device: {device}")
    log.info(f"Config: {config}")

    # Create dataset and dataloader
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

    log.info(f"Dataset size: {len(dataset)}")

    # Create model
    log.info("Creating StateVLA model with JEPA...")
    model_config = config['model']

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
        # Predictor config
        predictor_embed_dim=model_config.get('predictor_embed_dim', 192),
        predictor_depth=model_config.get('predictor_depth', 6),
        # Masking config
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
        # Device
        device=device,
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
    log.info(f"Action normalization - mean: {action_stats['mean'].numpy()}")
    log.info(f"Action normalization - std: {action_stats['std'].numpy()}")

    # Count parameters
    total_params = sum(p.numel() for p in trainer.parameters())
    trainable_params = sum(p.numel() for p in trainer.parameters() if p.requires_grad)
    log.info(f"Total parameters: {total_params / 1e6:.2f}M")
    log.info(f"Trainable parameters: {trainable_params / 1e6:.2f}M")

    # Create optimizer
    optimizer = torch.optim.AdamW(
        trainer.parameters(),
        lr=training_config['learning_rate'],
        weight_decay=training_config['weight_decay']
    )

    # Learning rate scheduler
    scheduler = None
    if training_config.get('use_lr_scheduler', False):
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=training_config['num_epochs'],
            eta_min=training_config.get('min_lr', 1e-6)
        )

    # Resume from checkpoint
    start_epoch = 0
    best_loss = float('inf')

    if args.checkpoint:
        log.info(f"Loading checkpoint from {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint, map_location=device)
        trainer.load_state_dict(checkpoint['model_state_dict'], strict=False)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_loss = checkpoint.get('loss', float('inf'))
        log.info(f"Resumed from epoch {start_epoch}")

    # Setup checkpoint directory
    checkpoint_dir = training_config.get('checkpoint_dir', 'checkpoints')
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    checkpoint_dir = os.path.join(checkpoint_dir, f'jepa_{timestamp}')

    # Training loop
    log.info("Starting JEPA training...")
    num_epochs = training_config['num_epochs']

    # Calculate total steps for EMA momentum schedule
    steps_per_epoch = len(dataloader)
    total_steps = num_epochs * steps_per_epoch
    global_step = start_epoch * steps_per_epoch

    log.info(f"Total training steps: {total_steps}")

    for epoch in range(start_epoch, num_epochs):
        # Train
        train_metrics = train_epoch(
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

        log.info(
            f"Epoch {epoch}: loss={train_metrics['loss']:.4f}, "
            f"action_loss={train_metrics['action_loss']:.4f}, "
            f"jepa_loss={train_metrics['jepa_loss']:.4f}"
        )

        # Save checkpoint
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

    log.info("Training complete!")


if __name__ == '__main__':
    main()
