"""
run_libero_eval.py

Evaluates StateVLA in LIBERO simulation benchmark.

Usage:
    python run_libero_eval.py \
        --checkpoint checkpoints/libero_object/jepa_xxx/checkpoint_best.pt \
        --task_suite libero_object \
        --num_trials 50
"""

import os
import sys
import math
import json
import logging
import argparse
from datetime import datetime
from collections import deque
from pathlib import Path

import numpy as np
import torch
import yaml
import imageio
from tqdm import tqdm

# Add project root to path
_project_root = os.path.dirname(os.path.abspath(__file__))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

# Add LIBERO to path
_libero_path = os.path.join(_project_root, "LIBERO")
if _libero_path not in sys.path:
    sys.path.insert(0, _libero_path)

from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv

from statevla_model import StateVLA
from jepa.tokenizer import PretrainedLanguageTokenizer

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

# Task suite configurations
TASK_SUITES = {
    "libero_spatial": {"max_steps": 220},
    "libero_object": {"max_steps": 280},
    "libero_goal": {"max_steps": 300},
    "libero_10": {"max_steps": 520},
    "libero_90": {"max_steps": 400},
}


def quat2axisangle(quat):
    """Convert quaternion to axis-angle format."""
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0

    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        return np.zeros(3)

    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


def get_libero_env(task, resolution=256):
    """Initialize LIBERO environment."""
    task_description = task.language
    task_bddl_file = os.path.join(
        get_libero_path("bddl_files"), task.problem_folder, task.bddl_file
    )
    env_args = {
        "bddl_file_name": task_bddl_file,
        "camera_heights": resolution,
        "camera_widths": resolution,
    }
    env = OffScreenRenderEnv(**env_args)
    env.seed(0)
    return env, task_description


def preprocess_image(img, target_size=224):
    """Preprocess image from LIBERO environment."""
    # Rotate 180 degrees (LIBERO convention)
    img = img[::-1, ::-1].copy()

    # Resize if needed
    if img.shape[0] != target_size or img.shape[1] != target_size:
        from PIL import Image
        pil_img = Image.fromarray(img)
        pil_img = pil_img.resize((target_size, target_size), Image.BILINEAR)
        img = np.array(pil_img)

    return img


def prepare_observation(obs, device, image_size=224):
    """
    Prepare observation dict for StateVLA model.

    Args:
        obs: Raw observation from LIBERO environment
        device: Torch device
        image_size: Target image size

    Returns:
        obs_dict: Dictionary ready for model input
    """
    # Get and preprocess images
    agentview = preprocess_image(obs["agentview_image"], image_size)
    eye_in_hand = preprocess_image(obs["robot0_eye_in_hand_image"], image_size)

    # Convert to tensor [C, H, W] and normalize to [0, 1]
    agentview_tensor = torch.from_numpy(agentview).permute(2, 0, 1).float() / 255.0
    eye_in_hand_tensor = torch.from_numpy(eye_in_hand).permute(2, 0, 1).float() / 255.0

    # Get robot state
    eef_pos = obs["robot0_eef_pos"]  # [3]
    eef_quat = obs["robot0_eef_quat"]  # [4]
    eef_axisangle = quat2axisangle(eef_quat.copy())  # [3]
    gripper_qpos = obs["robot0_gripper_qpos"]  # [2]

    # Combine robot state: [pos(3) + axisangle(3) + gripper(2)] = 8
    # But our model expects 9-dim (joint_states:7 + gripper:2)
    # We'll use: [eef_pos(3) + eef_axisangle(3) + gripper(2) + padding(1)]
    robot_state = np.concatenate([eef_pos, eef_axisangle, gripper_qpos, [0.0]])
    robot_state_tensor = torch.from_numpy(robot_state).float()

    # Build observation dict
    obs_dict = {
        "agentview_image": agentview_tensor.unsqueeze(0).to(device),  # [1, 3, H, W]
        "eye_in_hand_image": eye_in_hand_tensor.unsqueeze(0).to(device),  # [1, 3, H, W]
        "robot_states": robot_state_tensor.unsqueeze(0).to(device),  # [1, 9]
    }

    return obs_dict, agentview  # Return original image for video


def normalize_gripper_action(action, binarize=True):
    """Normalize gripper action from [0, 1] to [-1, 1]."""
    action = action.copy()
    if binarize:
        action[-1] = 1.0 if action[-1] > 0.5 else -1.0
    else:
        action[-1] = action[-1] * 2.0 - 1.0
    return action


def save_rollout_video(images, episode_idx, success, task_description, save_dir):
    """Save episode as MP4 video."""
    os.makedirs(save_dir, exist_ok=True)

    task_name = task_description.lower().replace(" ", "_")[:50]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"episode_{episode_idx}_success_{success}_{task_name}.mp4"
    filepath = os.path.join(save_dir, filename)

    writer = imageio.get_writer(filepath, fps=30)
    for img in images:
        writer.append_data(img)
    writer.close()

    return filepath


class LanguageEncoder:
    """Load pre-computed language embeddings from pickle file."""

    def __init__(self, task_suite: str, device="cuda"):
        self.device = device
        self.embed_dim = 3584

        # Load pre-computed embeddings
        emb_path = os.path.join(
            _project_root, "data", "libero", "language_embeddings", f"{task_suite}.pkl"
        )

        if os.path.exists(emb_path):
            import pickle
            with open(emb_path, 'rb') as f:
                self.embeddings = pickle.load(f)
            log.info(f"Loaded {len(self.embeddings)} language embeddings from {emb_path}")
        else:
            log.warning(f"Embeddings not found at {emb_path}, using random embeddings")
            self.embeddings = {}

    def _normalize_task_name(self, text: str) -> str:
        """Convert task description to key format."""
        # "pick up the butter and place it in the basket" -> "pick_up_the_butter_and_place_it_in_the_basket"
        return text.lower().replace(" ", "_").replace(".", "").replace(",", "")

    def encode(self, text: str) -> torch.Tensor:
        """Get embedding for task description."""
        key = self._normalize_task_name(text)

        if key in self.embeddings:
            emb = self.embeddings[key]
            if isinstance(emb, torch.Tensor):
                return emb.unsqueeze(0).to(self.device)
            else:
                return torch.from_numpy(emb).unsqueeze(0).float().to(self.device)
        else:
            log.warning(f"Embedding not found for: {key}")
            return torch.zeros(1, self.embed_dim, device=self.device)


def run_episode(
    model,
    env,
    task_description,
    lang_embedding,
    device,
    max_steps,
    initial_state,
    num_steps_wait=10,
    action_chunk_size=10,
    sample_steps=4,
    image_size=224,
):
    """
    Run a single episode in the environment.

    Args:
        model: StateVLA model
        env: LIBERO environment
        task_description: Task instruction string
        lang_embedding: Pre-computed language embedding
        device: Torch device
        max_steps: Maximum steps per episode
        num_steps_wait: Steps to wait for objects to stabilize
        action_chunk_size: Number of actions to execute before re-querying
        sample_steps: Denoising steps for action generation
        image_size: Image size for model input

    Returns:
        success: Whether episode succeeded
        images: List of images for video
    """
    model.eval()

    # Reset environment and set initial state
    env.reset()
    obs = env.set_init_state(initial_state)

    # Initialize
    t = 0
    replay_images = []
    action_queue = deque(maxlen=action_chunk_size)
    dummy_action = [0, 0, 0, 0, 0, 0, -1]

    success = False

    try:
        while t < max_steps + num_steps_wait:
            # Wait for objects to stabilize
            if t < num_steps_wait:
                obs, reward, done, info = env.step(dummy_action)
                t += 1
                continue

            # Prepare observation
            obs_dict, img = prepare_observation(obs, device, image_size)
            obs_dict["lang_emb"] = lang_embedding
            replay_images.append(img)

            # Query model if action queue is empty
            if len(action_queue) == 0:
                with torch.no_grad():
                    # Get action sequence from model
                    actions = model.predict(obs_dict, sample_steps=sample_steps)
                    actions = actions[0].cpu().numpy()  # [seq_len, action_dim]

                # Add actions to queue
                for i in range(min(action_chunk_size, len(actions))):
                    action_queue.append(actions[i])

            # Get next action
            action = action_queue.popleft()

            # Process action
            action = normalize_gripper_action(action, binarize=True)

            # Execute action
            obs, reward, done, info = env.step(action.tolist())

            if done:
                success = True
                break

            t += 1

    except Exception as e:
        log.error(f"Episode error: {e}")
        import traceback
        traceback.print_exc()

    return success, replay_images


def load_model(checkpoint_path, config_path, device):
    """Load StateVLA model from checkpoint."""
    log.info(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Get config
    if config_path:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
    elif "config" in checkpoint:
        config = checkpoint["config"]
    else:
        raise ValueError("No config found. Please provide --config argument.")

    # Create model
    model_config = config["model"]
    model = StateVLA(
        camera_names=config["cameras"]["names"],
        image_size=model_config.get("image_size", 224),
        patch_size=model_config.get("patch_size", 16),
        embed_dim=model_config.get("embed_dim", 256),
        lang_emb_dim=model_config.get("lang_emb_dim", 3584),
        robot_state_dim=model_config.get("robot_state_dim", 9),
        use_pretrained_vision=model_config.get("use_pretrained_vision", False),
        use_pretrained_language=model_config.get("use_pretrained_language", False),
        vision_model_name=model_config.get("vision_model_name", "google/siglip-base-patch16-224"),
        language_model_name=model_config.get("language_model_name", "ViT-B/32"),
        freeze_vision=model_config.get("freeze_vision", True),
        freeze_language=model_config.get("freeze_language", True),
        encoder_depth=model_config.get("encoder_depth", 12),
        d_state=model_config.get("d_state", 16),
        d_conv=model_config.get("d_conv", 4),
        expand=model_config.get("expand", 2),
        predictor_embed_dim=model_config.get("predictor_embed_dim", 192),
        predictor_depth=model_config.get("predictor_depth", 6),
        mask_ratio=model_config.get("mask_ratio", 0.5),
        masking_strategy=model_config.get("masking_strategy", "modality_aware"),
        state_dim=model_config.get("state_dim", 256),
        action_dim=model_config.get("action_dim", 7),
        action_seq_len=model_config.get("action_seq_len", 10),
        policy_layers=model_config.get("policy_layers", 3),
        policy_embed_dim=model_config.get("policy_embed_dim", 256),
        device=device,
    )

    # Load weights
    state_dict = checkpoint["model_state_dict"]
    model_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("model."):
            model_state_dict[k[6:]] = v
        else:
            model_state_dict[k] = v

    model.load_state_dict(model_state_dict, strict=False)
    model = model.to(device)
    model.eval()

    log.info(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")

    return model, config


def main():
    parser = argparse.ArgumentParser(description="Evaluate StateVLA on LIBERO")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to config file (optional)")
    parser.add_argument("--task_suite", type=str, default="libero_object",
                        choices=list(TASK_SUITES.keys()),
                        help="LIBERO task suite to evaluate")
    parser.add_argument("--num_trials", type=int, default=50,
                        help="Number of trials per task")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to use")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--save_video", action="store_true",
                        help="Save rollout videos")
    parser.add_argument("--video_dir", type=str, default="rollouts",
                        help="Directory to save videos")
    parser.add_argument("--action_chunk_size", type=int, default=10,
                        help="Number of actions to execute before re-querying")
    parser.add_argument("--sample_steps", type=int, default=4,
                        help="Denoising steps for action generation")
    parser.add_argument("--env_resolution", type=int, default=256,
                        help="Environment image resolution")
    args = parser.parse_args()

    # Set seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Setup device
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        log.warning("CUDA not available, using CPU")
        device = "cpu"

    # Load model
    model, config = load_model(args.checkpoint, args.config, device)
    image_size = config["model"].get("image_size", 224)

    # Initialize language encoder (uses pre-computed embeddings)
    log.info("Initializing language encoder...")
    lang_encoder = LanguageEncoder(task_suite=args.task_suite, device=device)

    # Get task suite
    log.info(f"Loading task suite: {args.task_suite}")
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite]()
    num_tasks = task_suite.n_tasks
    max_steps = TASK_SUITES[args.task_suite]["max_steps"]

    log.info(f"Number of tasks: {num_tasks}")
    log.info(f"Trials per task: {args.num_trials}")
    log.info(f"Max steps per episode: {max_steps}")

    # Setup logging
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = f"eval_logs/{args.task_suite}_{timestamp}"
    os.makedirs(log_dir, exist_ok=True)
    log_file = open(os.path.join(log_dir, "eval.log"), "w")

    # Results storage
    results = {
        "task_suite": args.task_suite,
        "checkpoint": args.checkpoint,
        "num_trials": args.num_trials,
        "tasks": {},
    }

    total_episodes = 0
    total_successes = 0

    # Evaluate each task
    for task_id in range(num_tasks):
        task = task_suite.get_task(task_id)
        initial_states = task_suite.get_task_init_states(task_id)

        # Create environment
        env, task_description = get_libero_env(task, resolution=args.env_resolution)

        log.info(f"\n{'='*60}")
        log.info(f"Task {task_id + 1}/{num_tasks}: {task_description}")
        log.info(f"{'='*60}")
        log_file.write(f"\nTask {task_id + 1}/{num_tasks}: {task_description}\n")

        # Encode task description
        lang_embedding = lang_encoder.encode(task_description)

        task_episodes = 0
        task_successes = 0

        for trial_idx in tqdm(range(args.num_trials), desc=f"Task {task_id + 1}"):
            # Get initial state
            initial_state = initial_states[trial_idx % len(initial_states)]

            # Run episode
            success, replay_images = run_episode(
                model=model,
                env=env,
                task_description=task_description,
                lang_embedding=lang_embedding,
                device=device,
                max_steps=max_steps,
                initial_state=initial_state,
                action_chunk_size=args.action_chunk_size,
                sample_steps=args.sample_steps,
                image_size=image_size,
            )

            task_episodes += 1
            total_episodes += 1

            if success:
                task_successes += 1
                total_successes += 1

            # Save video if requested
            if args.save_video:
                video_dir = os.path.join(args.video_dir, args.task_suite, f"task_{task_id}")
                save_rollout_video(
                    replay_images, trial_idx, success, task_description, video_dir
                )

        # Task results
        task_success_rate = task_successes / task_episodes if task_episodes > 0 else 0
        log.info(f"Task {task_id + 1} success rate: {task_success_rate:.2%} ({task_successes}/{task_episodes})")
        log_file.write(f"Success rate: {task_success_rate:.2%} ({task_successes}/{task_episodes})\n")

        results["tasks"][task_description] = {
            "success_rate": task_success_rate,
            "successes": task_successes,
            "episodes": task_episodes,
        }

        env.close()

    # Final results
    final_success_rate = total_successes / total_episodes if total_episodes > 0 else 0

    log.info("\n" + "=" * 60)
    log.info("FINAL RESULTS")
    log.info("=" * 60)
    log.info(f"Task Suite: {args.task_suite}")
    log.info(f"Total Episodes: {total_episodes}")
    log.info(f"Total Successes: {total_successes}")
    log.info(f"Overall Success Rate: {final_success_rate:.2%}")
    log.info("=" * 60)

    log_file.write("\n" + "=" * 60 + "\n")
    log_file.write("FINAL RESULTS\n")
    log_file.write(f"Overall Success Rate: {final_success_rate:.2%}\n")
    log_file.close()

    # Save results JSON
    results["overall_success_rate"] = final_success_rate
    results["total_episodes"] = total_episodes
    results["total_successes"] = total_successes

    with open(os.path.join(log_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)

    log.info(f"Results saved to {log_dir}")

    return final_success_rate


if __name__ == "__main__":
    main()
