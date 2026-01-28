"""
StateVLA Training Script

Usage:
    python train.py --config conf/config.yaml
    python train.py --config conf/config.yaml --data_directory /path/to/data
"""

# CRITICAL: Disable Flash Attention BEFORE any imports
import os
os.environ['TRANSFORMERS_NO_FLASH_ATTN'] = '1'
os.environ['DISABLE_FLASH_ATTN'] = '1'

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

# Monkey-patch transformers to disable flash attention checking
try:
    from transformers import modeling_utils
    original_check = modeling_utils.PreTrainedModel._check_and_adjust_attn_implementation

    def patched_check(self, config, torch_dtype=None, device_map=None, hard_check_only=False, check_device_map=True, is_init_check=False):
        # If config is already a string (like "eager"), just return it
        if isinstance(config, str):
            return config
        # Force eager attention on config objects
        if hasattr(config, '_attn_implementation'):
            config._attn_implementation = "eager"
        if hasattr(config, 'vision_config') and config.vision_config is not None:
            config.vision_config._attn_implementation = "eager"
        if hasattr(config, 'text_config') and config.text_config is not None:
            config.text_config._attn_implementation = "eager"
        return "eager"

    modeling_utils.PreTrainedModel._check_and_adjust_attn_implementation = patched_check
    print("Successfully patched transformers to disable flash attention")
except Exception as e:
    print(f"Warning: Could not patch transformers: {e}")

# Use absolute imports
from model_factory import create_statevla_model
from train_policy import create_training_model
from dataloader import StateVLADataset, create_dataloader

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
    is_best: bool = False
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

    # Save latest checkpoint
    path = os.path.join(checkpoint_dir, 'checkpoint_latest.pt')
    torch.save(checkpoint, path)

    # Save epoch checkpoint
    path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch}.pt')
    torch.save(checkpoint, path)

    # Save best checkpoint
    if is_best:
        path = os.path.join(checkpoint_dir, 'checkpoint_best.pt')
        torch.save(checkpoint, path)

    log.info(f"Saved checkpoint at epoch {epoch}")


def train_epoch(
    model,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: str,
    epoch: int,
    config: dict,
    ema=None
):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    total_action_loss = 0
    total_state_loss = 0
    num_batches = 0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")

    for batch in pbar:
        # Reset state for each batch since batches contain samples from different trajectories
        model.reset_state()
        # Move data to device
        obs = {k: v.to(device) for k, v in batch['obs'].items()}
        actions = batch['actions'].to(device)
        prev_action = batch['prev_action'].to(device)

        next_obs = None
        if batch['next_obs'] is not None:
            next_obs = {k: v.to(device) for k, v in batch['next_obs'].items()}

        # Forward pass
        optimizer.zero_grad()
        outputs = model.forward(
            obs_dict=obs,
            prev_action=prev_action,
            gt_actions=actions,
            next_obs_dict=next_obs
        )

        loss = outputs['loss']

        # Backward pass
        loss.backward()

        # Gradient clipping
        if config['training'].get('gradient_clip', 0) > 0:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                config['training']['gradient_clip']
            )

        optimizer.step()

        # Update EMA
        if ema is not None:
            ema.update()

        # Logging
        total_loss += loss.item()
        total_action_loss += outputs['action_loss'].item()
        total_state_loss += outputs['state_loss'].item()
        num_batches += 1

        pbar.set_postfix({
            'loss': f"{loss.item():.4f}",
            'act_loss': f"{outputs['action_loss'].item():.4f}",
            'state_loss': f"{outputs['state_loss'].item():.4f}"
        })

    avg_loss = total_loss / num_batches
    avg_action_loss = total_action_loss / num_batches
    avg_state_loss = total_state_loss / num_batches

    return {
        'loss': avg_loss,
        'action_loss': avg_action_loss,
        'state_loss': avg_state_loss
    }


def main():
    parser = argparse.ArgumentParser(description='Train StateVLA')
    parser.add_argument('--config', type=str, default='conf/config.yaml',
                        help='Path to config file')
    parser.add_argument('--data_directory', type=str, default=None,
                        help='Override data directory')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Resume from checkpoint')
    parser.add_argument('--device', type=str, default=None,
                        help='Override device')
    parser.add_argument('--cross_attention', action='store_true',
                        help='Use cross-attention fusion instead of MLP')
    parser.add_argument('--state_encoder_type', type=str, default=None,
                        choices=['mlp', 'cross_attention'],
                        help='State encoder type (mlp or cross_attention)')
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

    # Handle state encoder type override
    if args.cross_attention:
        config['model']['state_encoder_type'] = 'cross_attention'
    if args.state_encoder_type:
        config['model']['state_encoder_type'] = args.state_encoder_type

    # Handle batch size override
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
        image_size=config['cameras'].get('image_size', 224),
        camera_names=config['cameras']['names']
    )

    log.info(f"Dataset size: {len(dataset)}")

    # Create model
    log.info("Creating model...")
    model = create_statevla_model(
        camera_names=config['cameras']['names'],
        latent_dim=config['model']['latent_dim'],
        lang_emb_dim=config['model']['lang_emb_dim'],
        state_encoder_type=config['model'].get('state_encoder_type', 'mlp'),
        state_dim=config['model']['state_dim'],
        action_dim=config['model']['action_dim'],
        action_seq_len=config['model']['action_seq_len'],
        state_predictor_layers=config['model']['state_predictor_layers'],
        policy_layers=config['model']['policy_layers'],
        policy_embed_dim=config['model']['policy_embed_dim'],
        use_correction=config['model']['use_correction'],
        # Vision encoder config
        image_encoder_type=config['model'].get('image_encoder_type', 'resnet'),
        eagle2_model_name=config['model'].get('eagle2_model_name', 'nvidia/Eagle2-1B'),
        freeze_vision_encoder=config['model'].get('freeze_vision_encoder', True),
        # Language encoder config
        language_encoder_type=config['model'].get('language_encoder_type', 'clip'),
        qwen_model_name=config['model'].get('qwen_model_name', 'Qwen/Qwen-7B-Chat'),
        use_language_encoder=config['model']['use_language_encoder'],
        freeze_language_encoder=config['model']['freeze_language_encoder'],
        clip_model_name=config['model'].get('clip_model_name', 'ViT-B/32'),
        dropout=config['model']['dropout'],
        device=device
    )

    # Create training model
    training_model, ema = create_training_model(
        model=model,
        action_loss_weight=config['training']['action_loss_weight'],
        state_loss_weight=config['training']['state_loss_weight'],
        learning_rate=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay'],
        sampling_steps=config['training']['sampling_steps'],
        use_ema=config['training']['enable_ema'],
        ema_decay=config['training']['ema_decay_rate']
    )

    training_model = training_model.to(device)

    # Count parameters
    total_params = sum(p.numel() for p in training_model.parameters())
    trainable_params = sum(p.numel() for p in training_model.parameters() if p.requires_grad)
    log.info(f"Total parameters: {total_params / 1e6:.2f}M")
    log.info(f"Trainable parameters: {trainable_params / 1e6:.2f}M")

    # Create optimizer
    optimizer = torch.optim.AdamW(
        training_model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )

    # Resume from checkpoint
    start_epoch = 0
    best_loss = float('inf')

    if args.checkpoint:
        log.info(f"Loading checkpoint from {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint, map_location=device)

        # Filter out prev_state buffers from checkpoint
        state_dict = checkpoint['model_state_dict']
        state_dict = {k: v for k, v in state_dict.items()
                      if not (k.endswith('.prev_state') or k.endswith('.prev_pred_state'))}

        training_model.load_state_dict(state_dict, strict=False)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_loss = checkpoint.get('loss', float('inf'))
        log.info(f"Resumed from epoch {start_epoch}")

    # Setup checkpoint directory
    checkpoint_dir = config['training'].get('checkpoint_dir', 'checkpoints')
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    checkpoint_dir = os.path.join(checkpoint_dir, f'run_{timestamp}')

    # Setup wandb (optional)
    if config['training'].get('use_wandb', False):
        try:
            import wandb
            wandb.init(
                project=config['training'].get('wandb_project', 'statevla'),
                entity=config['training'].get('wandb_entity'),
                config=config
            )
        except ImportError:
            log.warning("wandb not installed, skipping logging")

    # Training loop
    log.info("Starting training...")
    num_epochs = config['training']['num_epochs']

    for epoch in range(start_epoch, num_epochs):
        # Train
        train_metrics = train_epoch(
            model=training_model,
            dataloader=dataloader,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            config=config,
            ema=ema
        )

        log.info(
            f"Epoch {epoch}: loss={train_metrics['loss']:.4f}, "
            f"action_loss={train_metrics['action_loss']:.4f}, "
            f"state_loss={train_metrics['state_loss']:.4f}"
        )

        # Log to wandb
        if config['training'].get('use_wandb', False):
            try:
                import wandb
                wandb.log({
                    'epoch': epoch,
                    'loss': train_metrics['loss'],
                    'action_loss': train_metrics['action_loss'],
                    'state_loss': train_metrics['state_loss']
                })
            except:
                pass

        # Save checkpoint
        is_best = train_metrics['loss'] < best_loss
        if is_best:
            best_loss = train_metrics['loss']

        if (epoch + 1) % config['training'].get('save_interval', 100) == 0 or is_best:
            save_checkpoint(
                model=training_model,
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
