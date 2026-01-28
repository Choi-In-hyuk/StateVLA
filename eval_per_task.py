"""
StateVLA Evaluation Script - Per Task Analysis

Evaluates model performance on each LIBERO task separately.

Usage:
    python eval_per_task.py --checkpoint checkpoints/checkpoint_best.pt --data_directory /path/to/data
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
        if isinstance(config, str):
            return config
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


def evaluate_per_task(
    model,
    data_directory: str,
    config: dict,
    device: str,
    samples_per_task: int = 50,
    batch_size: int = 32
):
    """
    Evaluate model on each task separately.

    Args:
        model: Trained model
        data_directory: Path to data directory
        config: Configuration dict
        device: Device to use
        samples_per_task: Number of samples to evaluate per task
        batch_size: Batch size for evaluation

    Returns:
        Dictionary with per-task and overall metrics
    """
    import h5py
    import glob

    # Get all task files
    task_files = sorted(glob.glob(os.path.join(data_directory, "*.hdf5")))

    if len(task_files) == 0:
        raise ValueError(f"No HDF5 files found in {data_directory}")

    log.info(f"Found {len(task_files)} tasks")

    task_results = {}
    all_mses = []

    model.eval()

    for task_file in task_files:
        task_name = os.path.basename(task_file).replace('_demo.hdf5', '')
        log.info(f"\n{'='*60}")
        log.info(f"Evaluating task: {task_name}")
        log.info(f"{'='*60}")

        # Create dataloader for this task only
        task_dir = os.path.dirname(task_file)

        # Temporarily move other files
        other_files = [f for f in task_files if f != task_file]
        temp_moved = []

        try:
            # Create temp directory
            temp_dir = os.path.join(task_dir, '.temp_eval')
            os.makedirs(temp_dir, exist_ok=True)

            # Move other task files temporarily
            for other_file in other_files:
                temp_path = os.path.join(temp_dir, os.path.basename(other_file))
                os.rename(other_file, temp_path)
                temp_moved.append((other_file, temp_path))

            # Create dataloader for single task
            dataloader, dataset = create_dataloader(
                data_directory=task_dir,
                batch_size=batch_size,
                shuffle=False,
                action_dim=config['model']['action_dim'],
                action_seq_len=config['model']['action_seq_len'],
                demos_per_task=config['data'].get('demos_per_task', 50),
                max_len_data=config['data'].get('max_len_data', 260),
                image_size=config['cameras'].get('image_size', 224),
                camera_names=config['cameras']['names']
            )

            log.info(f"Dataset size for {task_name}: {len(dataset)}")

            # Evaluate this task
            task_mse = 0
            task_samples = 0
            task_errors = []

            with torch.no_grad():
                pbar = tqdm(dataloader, desc=f"Evaluating {task_name[:30]}")

                for batch_idx, batch in enumerate(pbar):
                    if samples_per_task and task_samples >= samples_per_task:
                        break

                    # Reset state for each batch
                    model.reset_state()

                    # Move to device
                    obs = {k: v.to(device) for k, v in batch['obs'].items()}
                    actions = batch['actions'].to(device)
                    prev_action = batch['prev_action'].to(device)

                    # Predict actions
                    pred_actions = model.predict(obs, prev_action)

                    # Compute MSE
                    mse = ((pred_actions - actions) ** 2).mean(dim=[1, 2])
                    task_errors.extend(mse.cpu().numpy().tolist())

                    task_mse += mse.sum().item()
                    task_samples += actions.shape[0]

                    pbar.set_postfix({'mse': f"{task_mse / task_samples:.6f}"})

            # Store results
            avg_mse = task_mse / task_samples if task_samples > 0 else float('inf')
            std_mse = np.std(task_errors) if len(task_errors) > 0 else 0.0

            task_results[task_name] = {
                'mse': avg_mse,
                'std': std_mse,
                'samples': task_samples
            }

            all_mses.extend(task_errors)

            log.info(f"{task_name}: MSE = {avg_mse:.6f} ± {std_mse:.6f} ({task_samples} samples)")

        finally:
            # Restore moved files
            for original_path, temp_path in temp_moved:
                os.rename(temp_path, original_path)

            # Remove temp directory
            if os.path.exists(temp_dir):
                os.rmdir(temp_dir)

    # Compute overall statistics
    overall_mse = np.mean(all_mses)
    overall_std = np.std(all_mses)

    return {
        'per_task': task_results,
        'overall_mse': overall_mse,
        'overall_std': overall_std,
        'total_samples': len(all_mses)
    }


def main():
    parser = argparse.ArgumentParser(description='Evaluate StateVLA per task')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--config', type=str, default=None,
                        help='Path to config file')
    parser.add_argument('--data_directory', type=str, default=None,
                        help='Path to evaluation data')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use')
    parser.add_argument('--samples_per_task', type=int, default=None,
                        help='Number of samples to evaluate per task (None = all)')
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

    # Load weights (ignore prev_state buffers)
    state_dict = checkpoint['model_state_dict']
    state_dict = {k: v for k, v in state_dict.items()
                  if not (k.endswith('.prev_state') or k.endswith('.prev_pred_state'))}
    training_model.load_state_dict(state_dict, strict=False)
    training_model = training_model.to(device)

    log.info(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")

    # Evaluate
    log.info("Starting per-task evaluation...")
    results = evaluate_per_task(
        model=training_model,
        data_directory=config['data']['data_directory'],
        config=config,
        device=device,
        samples_per_task=args.samples_per_task,
        batch_size=args.batch_size
    )

    # Print results
    print("\n" + "="*80)
    print("EVALUATION RESULTS")
    print("="*80)

    print("\nPer-Task Results:")
    print("-"*80)
    for task_name, metrics in results['per_task'].items():
        task_short = task_name.replace('pick_up_the_', '').replace('_and_place_it_in_the_basket', '')
        print(f"{task_short:30s}: MSE = {metrics['mse']:.6f} ± {metrics['std']:.6f} ({metrics['samples']} samples)")

    print("\n" + "-"*80)
    print(f"Overall MSE: {results['overall_mse']:.6f} ± {results['overall_std']:.6f}")
    print(f"Total samples: {results['total_samples']}")
    print("="*80)


if __name__ == '__main__':
    main()
