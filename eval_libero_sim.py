"""
StateVLA LIBERO Simulation Evaluation Script

Evaluates a trained StateVLA policy in LIBERO simulation benchmark tasks.

Usage:
    python eval_libero_sim.py \
        --checkpoint checkpoints/libero_object/run_XXX/checkpoint_best.pt \
        --config conf/config_libero_object.yaml \
        --task_suite_name libero_object \
        --num_trials_per_task 50
"""

# CRITICAL: Disable Flash Attention BEFORE any imports
import os
os.environ['TRANSFORMERS_NO_FLASH_ATTN'] = '1'
os.environ['DISABLE_FLASH_ATTN'] = '1'
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Fix robosuite log file permission issue
# Monkey-patch logging.FileHandler before robosuite import
import logging
import tempfile

_original_file_handler_init = logging.FileHandler.__init__

def _patched_file_handler_init(self, filename, mode='a', encoding=None, delay=False, errors=None):
    # Redirect /tmp/robosuite.log to a writable location
    if filename == '/tmp/robosuite.log':
        temp_dir = tempfile.gettempdir()
        filename = os.path.join(temp_dir, f'robosuite_{os.getpid()}.log')
    _original_file_handler_init(self, filename, mode, encoding, delay, errors)

logging.FileHandler.__init__ = _patched_file_handler_init

import argparse
import json
import logging
import sys
from collections import deque
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
import tqdm
import yaml
from PIL import Image

# Add project root to path
_project_root = os.path.dirname(os.path.abspath(__file__))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

# Add LIBERO to path
libero_path = os.path.join(_project_root, "LIBERO")
if libero_path not in sys.path:
    sys.path.insert(0, libero_path)

import torchvision.transforms as transforms

# Monkey-patch transformers
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
except Exception as e:
    print(f"Warning: Could not patch transformers: {e}")

from libero.libero import benchmark
from libero.libero.envs import OffScreenRenderEnv

from model_factory import create_statevla_model
from train_policy import create_training_model

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


# Task suite constants
TASK_SUITES = {
    "libero_spatial": "libero_spatial",
    "libero_object": "libero_object",
    "libero_goal": "libero_goal",
    "libero_10": "libero_10",
}

TASK_MAX_STEPS = {
    "libero_spatial": 220,
    "libero_object": 280,
    "libero_goal": 300,
    "libero_10": 520,
}


def load_model(checkpoint_path: str, config_path: str, device: str):
    """Load StateVLA model from checkpoint."""
    logger.info(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Load config - prefer checkpoint config over file config
    if 'config' in checkpoint:
        config = checkpoint['config']
        logger.info("Using config from checkpoint")
    else:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        logger.info("Using config from file")

    logger.info("Creating StateVLA model...")
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
        qwen_model_name=config['model'].get('qwen_model_name', 'Qwen/Qwen2-7B-Instruct'),
        use_language_encoder=config['model']['use_language_encoder'],
        freeze_language_encoder=config['model']['freeze_language_encoder'],
        clip_model_name=config['model'].get('clip_model_name', 'ViT-B/32'),
        dropout=config['model']['dropout'],
        device=device
    )

    # Create training model wrapper
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
    training_model.eval()

    logger.info(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")

    return training_model, config


def get_libero_env(task, resolution=256):
    """Create LIBERO environment."""
    from libero.libero import get_libero_path

    bddl_file = os.path.join(
        get_libero_path("bddl_files"),
        task.problem_folder,
        task.bddl_file
    )

    env_args = {
        "bddl_file_name": bddl_file,
        "camera_heights": resolution,
        "camera_widths": resolution,
    }

    env = OffScreenRenderEnv(**env_args)
    task_description = task.language

    return env, task_description


def prepare_observation(obs, image_size=384, device='cuda'):
    """
    Prepare observation for StateVLA model.

    Args:
        obs: LIBERO observation dict
        image_size: Size to resize images to
        device: Device to put tensors on

    Returns:
        obs_dict: Dictionary with processed observations
    """
    # Get images from LIBERO observation
    agentview_img = obs['agentview_image']  # [H, W, 3]
    eye_in_hand_img = obs['robot0_eye_in_hand_image']  # [H, W, 3]

    # Convert BGR to RGB if needed (LIBERO uses RGB)
    # Images are already in RGB format [H, W, 3] with values in [0, 255]

    # Resize images
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),  # Converts to [C, H, W] and scales to [0, 1]
    ])

    agentview_tensor = transform(agentview_img).unsqueeze(0)  # [1, 3, H, W]
    eye_in_hand_tensor = transform(eye_in_hand_img).unsqueeze(0)  # [1, 3, H, W]

    # Create observation dict for StateVLA (model expects keys with '_image' suffix)
    obs_dict = {
        'agentview_image': agentview_tensor.to(device),
        'eye_in_hand_image': eye_in_hand_tensor.to(device),
    }

    return obs_dict


def quat2axisangle(quat):
    """Convert quaternion to axis-angle representation."""
    # quat: [x, y, z, w]
    theta = 2 * np.arccos(np.clip(quat[3], -1, 1))
    if theta < 1e-6:
        return np.zeros(3)
    axis = quat[:3] / np.sin(theta / 2)
    return axis * theta


def run_episode(
    model,
    env,
    task_description: str,
    config: dict,
    initial_state,
    device: str,
    num_steps_wait: int = 10,
    max_steps: int = 280,
    action_seq_len: int = 10,
):
    """
    Run a single episode.

    Returns:
        success: Whether the task was completed successfully
        replay_images: List of images for video replay
    """
    # Reset environment
    env.reset()

    # Set initial state
    if initial_state is not None:
        obs = env.set_init_state(initial_state)
    else:
        obs = env.get_observation()

    # Reset model state
    model.reset_state()

    # Initialize previous action (zero vector)
    prev_action = torch.zeros(1, config['model']['action_dim'], device=device)

    # Action queue for open-loop execution
    action_queue = deque(maxlen=action_seq_len)

    # Setup
    t = 0
    replay_images = []
    success = False

    try:
        while t < max_steps + num_steps_wait:
            # Wait for objects to stabilize
            if t < num_steps_wait:
                dummy_action = np.zeros(7)
                obs, reward, done, info = env.step(dummy_action)
                t += 1
                continue

            # Prepare observation for model
            obs_dict = prepare_observation(
                obs,
                image_size=config['cameras'].get('image_size', 384),
                device=device
            )

            # Add language instruction
            obs_dict['lang'] = task_description

            # Save image for replay
            replay_images.append(obs['agentview_image'])

            # Query model if action queue is empty
            if len(action_queue) == 0:
                with torch.no_grad():
                    # Predict action sequence
                    actions = model.predict(obs_dict, prev_action)  # [1, action_seq_len, action_dim]

                    # Add actions to queue
                    actions_np = actions[0].cpu().numpy()  # [action_seq_len, action_dim]
                    for i in range(actions_np.shape[0]):
                        action_queue.append(actions_np[i])

            # Get next action from queue
            action = action_queue.popleft()

            # Update previous action
            prev_action = torch.from_numpy(action).unsqueeze(0).float().to(device)

            # Execute action
            obs, reward, done, info = env.step(action.tolist())

            if done:
                success = True
                break

            t += 1

    except Exception as e:
        logger.error(f"Episode error: {e}")
        import traceback
        traceback.print_exc()

    return success, replay_images


def save_video(images, filename, fps=10):
    """Save images as video."""
    if len(images) == 0:
        return

    h, w = images[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(filename, fourcc, fps, (w, h))

    for img in images:
        # Convert RGB to BGR for OpenCV
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        video.write(img_bgr)

    video.release()
    logger.info(f"Saved video to {filename}")


def run_task(
    model,
    task_suite,
    task_id: int,
    config: dict,
    device: str,
    num_trials: int = 50,
    num_steps_wait: int = 10,
    max_steps: int = 280,
    save_videos: bool = False,
    video_dir: Optional[str] = None,
):
    """Run evaluation for a single task."""
    # Get task
    task = task_suite.get_task(task_id)

    # Get initial states
    initial_states = task_suite.get_task_init_states(task_id)

    # Create environment
    env, task_description = get_libero_env(task, resolution=256)

    logger.info(f"\n{'='*80}")
    logger.info(f"Task {task_id}: {task_description}")
    logger.info(f"{'='*80}")

    # Run episodes
    successes = 0
    for episode_idx in tqdm.tqdm(range(num_trials), desc=f"Task {task_id}"):
        initial_state = initial_states[episode_idx % len(initial_states)]

        success, replay_images = run_episode(
            model=model,
            env=env,
            task_description=task_description,
            config=config,
            initial_state=initial_state,
            device=device,
            num_steps_wait=num_steps_wait,
            max_steps=max_steps,
            action_seq_len=config['model']['action_seq_len'],
        )

        if success:
            successes += 1

        # Save video if requested
        if save_videos and video_dir is not None:
            os.makedirs(video_dir, exist_ok=True)
            status = "success" if success else "failure"
            video_path = os.path.join(
                video_dir,
                f"task{task_id}_ep{episode_idx}_{status}.mp4"
            )
            save_video(replay_images, video_path)

        logger.info(f"Episode {episode_idx}: {'SUCCESS' if success else 'FAILURE'}")

    success_rate = successes / num_trials
    logger.info(f"Task {task_id} success rate: {success_rate:.2%} ({successes}/{num_trials})")

    env.close()

    return success_rate, successes, num_trials


def main():
    parser = argparse.ArgumentParser(description='Evaluate StateVLA on LIBERO')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to config file')
    parser.add_argument('--task_suite_name', type=str, default='libero_object',
                        choices=['libero_spatial', 'libero_object', 'libero_goal', 'libero_10'],
                        help='LIBERO task suite')
    parser.add_argument('--num_trials_per_task', type=int, default=50,
                        help='Number of trials per task')
    parser.add_argument('--num_steps_wait', type=int, default=10,
                        help='Number of steps to wait for stabilization')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use')
    parser.add_argument('--save_videos', action='store_true',
                        help='Save rollout videos')
    parser.add_argument('--video_dir', type=str, default='./eval_videos',
                        help='Directory to save videos')
    parser.add_argument('--log_file', type=str, default=None,
                        help='Path to log file')

    args = parser.parse_args()

    # Setup device
    device = args.device
    if device == 'cuda' and not torch.cuda.is_available():
        logger.warning("CUDA not available, using CPU")
        device = 'cpu'

    # Load model
    model, config = load_model(args.checkpoint, args.config, device)

    # Get max steps for task suite
    max_steps = TASK_MAX_STEPS[args.task_suite_name]

    # Load LIBERO task suite
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[TASK_SUITES[args.task_suite_name]]()
    num_tasks = task_suite.n_tasks

    logger.info(f"Evaluating on {args.task_suite_name} with {num_tasks} tasks")

    # Run evaluation
    all_results = []
    total_successes = 0
    total_trials = 0

    for task_id in range(num_tasks):
        success_rate, successes, trials = run_task(
            model=model,
            task_suite=task_suite,
            task_id=task_id,
            config=config,
            device=device,
            num_trials=args.num_trials_per_task,
            num_steps_wait=args.num_steps_wait,
            max_steps=max_steps,
            save_videos=args.save_videos,
            video_dir=args.video_dir,
        )

        all_results.append({
            'task_id': task_id,
            'success_rate': success_rate,
            'successes': successes,
            'trials': trials,
        })

        total_successes += successes
        total_trials += trials

    # Print summary
    overall_success_rate = total_successes / total_trials if total_trials > 0 else 0

    logger.info("\n" + "="*80)
    logger.info("EVALUATION SUMMARY")
    logger.info("="*80)
    for result in all_results:
        logger.info(f"Task {result['task_id']}: {result['success_rate']:.2%} ({result['successes']}/{result['trials']})")
    logger.info("-"*80)
    logger.info(f"Overall: {overall_success_rate:.2%} ({total_successes}/{total_trials})")
    logger.info("="*80)

    # Save results to file
    if args.log_file:
        with open(args.log_file, 'w') as f:
            json.dump({
                'task_suite': args.task_suite_name,
                'overall_success_rate': overall_success_rate,
                'total_successes': total_successes,
                'total_trials': total_trials,
                'per_task_results': all_results,
            }, f, indent=2)
        logger.info(f"Results saved to {args.log_file}")


if __name__ == '__main__':
    main()
