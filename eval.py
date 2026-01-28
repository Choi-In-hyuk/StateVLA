"""
StateVLA Evaluation Script

Usage:
    python eval.py --checkpoint checkpoints/checkpoint_best.pt --data_directory /path/to/data
"""

# CRITICAL: Disable Flash Attention BEFORE any imports
import os
os.environ['TRANSFORMERS_NO_FLASH_ATTN'] = '1'
os.environ['DISABLE_FLASH_ATTN'] = '1'

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

from model_factory import create_statevla_model
from train_policy import create_training_model
from dataloader import create_dataloader

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


def evaluate(
    model,
    dataloader,
    device: str,
    num_samples: int = None
):
    """
    Evaluate model on dataset.

    Returns:
        Dictionary with evaluation metrics
    """
    model.eval()
    model.reset_state()

    total_mse = 0
    total_samples = 0
    all_errors = []

    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Evaluating")

        for batch_idx, batch in enumerate(pbar):
            if num_samples and batch_idx * dataloader.batch_size >= num_samples:
                break

            # Reset state for each batch since batches contain samples from different trajectories
            model.reset_state()

            # Move to device
            obs = {k: v.to(device) for k, v in batch['obs'].items()}
            actions = batch['actions'].to(device)
            prev_action = batch['prev_action'].to(device)

            # Predict actions
            pred_actions = model.predict(obs, prev_action)

            # Compute MSE
            mse = ((pred_actions - actions) ** 2).mean(dim=[1, 2])
            all_errors.extend(mse.cpu().numpy().tolist())

            total_mse += mse.sum().item()
            total_samples += actions.shape[0]

            pbar.set_postfix({'mse': f"{total_mse / total_samples:.6f}"})

    avg_mse = total_mse / total_samples
    std_error = np.std(all_errors)

    return {
        'mse': avg_mse,
        'std': std_error,
        'num_samples': total_samples
    }


def main():
    parser = argparse.ArgumentParser(description='Evaluate StateVLA')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--config', type=str, default=None,
                        help='Path to config file (optional, uses checkpoint config if not provided)')
    parser.add_argument('--data_directory', type=str, default=None,
                        help='Path to evaluation data')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use')
    parser.add_argument('--num_samples', type=int, default=None,
                        help='Number of samples to evaluate (None = all)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for evaluation')
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

    # Create training model wrapper for inference
    training_model, _ = create_training_model(
        model=model,
        action_loss_weight=config['training'].get('action_loss_weight', 1.0),
        state_loss_weight=config['training'].get('state_loss_weight', 0.1),
        learning_rate=config['training'].get('learning_rate', 1e-4),
        weight_decay=config['training'].get('weight_decay', 0.05),
        sampling_steps=config['training'].get('sampling_steps', 4),
        use_ema=False
    )

    # Load weights (ignore prev_state buffers as they are runtime-only)
    state_dict = checkpoint['model_state_dict']
    # Remove prev_state and prev_pred_state buffers from checkpoint
    state_dict = {k: v for k, v in state_dict.items()
                  if not (k.endswith('.prev_state') or k.endswith('.prev_pred_state'))}
    training_model.load_state_dict(state_dict, strict=False)
    training_model = training_model.to(device)

    log.info(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")

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
        image_size=config['cameras'].get('image_size', 224),
        camera_names=config['cameras']['names']
    )

    log.info(f"Dataset size: {len(dataset)}")

    # Evaluate
    log.info("Starting evaluation...")
    metrics = evaluate(
        model=training_model,
        dataloader=dataloader,
        device=device,
        num_samples=args.num_samples
    )

    log.info("=" * 50)
    log.info("Evaluation Results:")
    log.info(f"  MSE: {metrics['mse']:.6f} ± {metrics['std']:.6f}")
    log.info(f"  Samples: {metrics['num_samples']}")
    log.info("=" * 50)


if __name__ == '__main__':
    main()
