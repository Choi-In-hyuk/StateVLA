"""
StateVLA JEPA Evaluation Script

Usage:
    python eval.py --checkpoint checkpoints/libero_object/jepa_xxx/checkpoint_best.pt
    python eval.py --checkpoint checkpoints/checkpoint_best.pt --config conf/config.yaml
"""

import os
import argparse
import logging
import sys

# Add project root to path
_project_root = os.path.dirname(os.path.abspath(__file__))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import numpy as np
import torch
import yaml
from tqdm import tqdm

from statevla_model import StateVLA, StateVLATrainer
from dataloader import create_dataloader

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


def evaluate(
    model: StateVLA,
    dataloader,
    device: str,
    num_samples: int = None,
    sample_steps: int = 4
):
    """
    Evaluate model on dataset.

    Args:
        model: StateVLA model (not trainer)
        dataloader: Data loader
        device: Device to use
        num_samples: Number of samples to evaluate (None = all)
        sample_steps: Number of denoising steps for action generation

    Returns:
        Dictionary with evaluation metrics
    """
    model.eval()

    total_mse = 0
    total_samples = 0
    all_errors = []

    # Per-dimension errors for analysis
    per_dim_errors = []

    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Evaluating")

        for batch_idx, batch in enumerate(pbar):
            if num_samples and batch_idx * dataloader.batch_size >= num_samples:
                break

            # Move to device
            obs = {k: v.to(device) for k, v in batch['obs'].items()}
            actions_gt = batch['actions'].to(device)  # [B, seq_len, action_dim]

            # Predict actions
            pred_actions = model.predict(obs, sample_steps=sample_steps)

            # Binarize gripper action (dim 6): continuous -> binary (-1 or 1)
            pred_actions[:, :, 6] = torch.where(
                pred_actions[:, :, 6] > 0,
                torch.ones_like(pred_actions[:, :, 6]),
                -torch.ones_like(pred_actions[:, :, 6])
            )

            # Compute MSE per sample
            mse_per_sample = ((pred_actions - actions_gt) ** 2).mean(dim=[1, 2])
            all_errors.extend(mse_per_sample.cpu().numpy().tolist())

            # Per-dimension MSE
            mse_per_dim = ((pred_actions - actions_gt) ** 2).mean(dim=[0, 1])
            per_dim_errors.append(mse_per_dim.cpu().numpy())

            total_mse += mse_per_sample.sum().item()
            total_samples += actions_gt.shape[0]

            pbar.set_postfix({'mse': f"{total_mse / total_samples:.6f}"})

    avg_mse = total_mse / total_samples
    std_error = np.std(all_errors)
    per_dim_avg = np.mean(per_dim_errors, axis=0)

    return {
        'mse': avg_mse,
        'rmse': np.sqrt(avg_mse),
        'std': std_error,
        'per_dim_mse': per_dim_avg,
        'num_samples': total_samples
    }


def main():
    parser = argparse.ArgumentParser(description='Evaluate StateVLA JEPA')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--config', type=str, default=None,
                        help='Path to config file (optional, uses checkpoint config if not provided)')
    parser.add_argument('--data_directory', type=str, default=None,
                        help='Path to evaluation data (overrides config)')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use')
    parser.add_argument('--num_samples', type=int, default=None,
                        help='Number of samples to evaluate (None = all)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for evaluation')
    parser.add_argument('--sample_steps', type=int, default=4,
                        help='Number of denoising steps for action generation')
    args = parser.parse_args()

    # Setup device
    device = args.device
    if device == 'cuda' and not torch.cuda.is_available():
        log.warning("CUDA not available, using CPU")
        device = 'cpu'

    # Load checkpoint
    log.info(f"Loading checkpoint from {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)

    # Get config
    if args.config:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
    elif 'config' in checkpoint:
        config = checkpoint['config']
    else:
        raise ValueError("No config found. Please provide --config argument.")

    # Override data directory if provided
    if args.data_directory:
        config['data']['data_directory'] = args.data_directory

    # Create model
    log.info("Creating StateVLA model...")
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

    # Load model weights
    state_dict = checkpoint['model_state_dict']

    # Filter out trainer-specific keys if needed
    model_state_dict = {}
    for k, v in state_dict.items():
        # Remove 'model.' prefix if it exists (from StateVLATrainer wrapper)
        if k.startswith('model.'):
            model_state_dict[k[6:]] = v
        else:
            model_state_dict[k] = v

    model.load_state_dict(model_state_dict, strict=False)
    model = model.to(device)

    epoch = checkpoint.get('epoch', 'unknown')
    train_loss = checkpoint.get('loss', 'unknown')
    log.info(f"Loaded checkpoint from epoch {epoch} (train loss: {train_loss})")

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    log.info(f"Total parameters: {total_params / 1e6:.2f}M")

    # Create dataloader
    log.info("Loading dataset...")
    dataloader, dataset = create_dataloader(
        data_directory=config['data']['data_directory'],
        batch_size=args.batch_size,
        shuffle=False,
        action_dim=config['model']['action_dim'],
        action_seq_len=config['model']['action_seq_len'],
        demos_per_task=config['data'].get('demos_per_task', 50),
        max_len_data=config['data'].get('max_len_data', 260),
        image_size=config['model'].get('image_size', 224),
        camera_names=config['cameras']['names'],
    )

    log.info(f"Dataset size: {len(dataset)}")

    # Evaluate
    log.info("Starting evaluation...")
    metrics = evaluate(
        model=model,
        dataloader=dataloader,
        device=device,
        num_samples=args.num_samples,
        sample_steps=args.sample_steps
    )

    # Print results
    log.info("=" * 60)
    log.info("Evaluation Results:")
    log.info(f"  MSE:  {metrics['mse']:.6f} ± {metrics['std']:.6f}")
    log.info(f"  RMSE: {metrics['rmse']:.6f}")
    log.info(f"  Samples: {metrics['num_samples']}")
    log.info("-" * 60)
    log.info("Per-dimension MSE:")
    dim_names = ['x', 'y', 'z', 'rx', 'ry', 'rz', 'gripper']
    for i, (name, err) in enumerate(zip(dim_names, metrics['per_dim_mse'])):
        log.info(f"  {name}: {err:.6f}")
    log.info("=" * 60)


if __name__ == '__main__':
    main()
