"""
StateVLA Dataset and DataLoader

Supports loading data from:
  - LIBERO benchmark (HDF5 format)
  - Custom datasets
"""

import logging
import os
import pickle
from typing import Dict, List, Optional, Tuple

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF

log = logging.getLogger(__name__)


class StateVLADataset(Dataset):
    """
    Dataset for StateVLA training.

    Loads demonstrations with:
      - Multi-camera RGB observations
      - Language embeddings
      - Robot states
      - Actions

    Supports sequence-based sampling for state prediction.
    Supports multi-step rollout for JEPA-style training.
    """

    def __init__(
        self,
        data_directory: str,
        language_embedding_path: Optional[str] = None,
        obs_dim: int = 32,
        action_dim: int = 7,
        state_dim: int = 45,
        max_len_data: int = 260,
        action_seq_len: int = 10,
        start_idx: int = 0,
        demos_per_task: int = 50,
        image_size: int = 224,
        camera_names: List[str] = None,
        max_rollout_steps: int = 5
    ):
        """
        Args:
            data_directory: Path to HDF5 data files
            language_embedding_path: Path to language embeddings pickle file
            obs_dim: Observation dimension
            action_dim: Action dimension
            state_dim: Robot state dimension
            max_len_data: Maximum trajectory length
            action_seq_len: Length of action sequence to predict
            start_idx: Starting demo index
            demos_per_task: Number of demos to load per task
            image_size: Target image size
            camera_names: List of camera names
        """
        self.data_directory = data_directory
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.max_len_data = max_len_data
        self.action_seq_len = action_seq_len
        self.start_idx = start_idx
        self.demos_per_task = demos_per_task
        self.image_size = image_size
        self.camera_names = camera_names or ['agentview', 'eye_in_hand']
        self.max_rollout_steps = max_rollout_steps

        log.info(f"Loading dataset from {data_directory}")

        # Load language embeddings
        self.tasks = self._load_language_embeddings(language_embedding_path)

        # Load demonstrations
        self._load_demonstrations()

        # Create sample slices
        self.slices = self._get_slices()

        # Compute action normalization statistics
        self._compute_action_stats()

        log.info(f"Loaded {self.num_data} trajectories, {len(self.slices)} samples")

    def _load_language_embeddings(self, embedding_path: Optional[str]) -> Dict:
        """Load language embeddings from pickle file."""
        if embedding_path is None:
            # Try to find embeddings based on data directory
            benchmark_type = os.path.basename(self.data_directory)
            possible_paths = [
                os.path.join(os.path.dirname(self.data_directory), "language_embeddings", f"{benchmark_type}.pkl"),
                os.path.join(self.data_directory, "..", "language_embeddings", f"{benchmark_type}.pkl"),
            ]

            for path in possible_paths:
                if os.path.exists(path):
                    embedding_path = path
                    break

            if embedding_path is None:
                log.warning("Language embeddings not found. Using dummy embeddings.")
                return {}

        log.info(f"Loading language embeddings from {embedding_path}")
        with open(embedding_path, 'rb') as f:
            return pickle.load(f)

    def _load_demonstrations(self):
        """Load demonstration data from HDF5 files."""
        self.data_embs = []
        self.actions = []
        self.masks = []
        self.agentview_rgb = []
        self.eye_in_hand_rgb = []
        self.robot_states = []

        file_list = [f for f in os.listdir(self.data_directory) if f.endswith('.hdf5')]

        for file in file_list:
            filename = os.path.basename(file).split('.')[0]
            # Remove '_demo' suffix if present
            if filename.endswith('_demo'):
                filename = filename[:-5]

            # Get task embedding
            task_emb = self.tasks.get(filename, torch.zeros(512))

            filepath = os.path.join(self.data_directory, file)
            with h5py.File(filepath, 'r') as f:
                log.info(f"Loading demo: {file}")

                demo_keys = list(f["data"].keys())
                indices = np.argsort([int(k[5:]) for k in demo_keys])

                for i in indices[self.start_idx:self.start_idx + self.demos_per_task]:
                    demo_name = demo_keys[i]
                    demo = f["data"][demo_name]
                    demo_length = demo.attrs["num_samples"]

                    # Initialize padded arrays
                    zero_actions = np.zeros((1, self.max_len_data, self.action_dim), dtype=np.float32)
                    zero_mask = np.zeros((1, self.max_len_data), dtype=np.float32)

                    # Load action data
                    action_data = demo['actions'][:]
                    zero_actions[0, :demo_length, :] = action_data
                    zero_mask[0, :demo_length] = 1

                    # Load observations
                    agentview = demo['obs']['agentview_rgb'][:]
                    eye_in_hand = demo['obs']['eye_in_hand_rgb'][:]

                    # Load robot states
                    joint_states = demo['obs']['joint_states'][:]
                    gripper_states = demo['obs']['gripper_states'][:]
                    if len(gripper_states.shape) == 1:
                        gripper_states = gripper_states[:, np.newaxis]
                    robot_state = np.concatenate((joint_states, gripper_states), axis=-1)

                    # Store data
                    self.actions.append(zero_actions)
                    self.masks.append(zero_mask)
                    self.agentview_rgb.append(agentview)
                    self.eye_in_hand_rgb.append(eye_in_hand)
                    self.robot_states.append(robot_state)
                    self.data_embs.append(task_emb)

        # Convert to tensors
        self.actions = torch.from_numpy(np.concatenate(self.actions)).float()
        self.masks = torch.from_numpy(np.concatenate(self.masks)).float()
        self.num_data = len(self.agentview_rgb)

    def _get_slices(self) -> List[Tuple[int, int, int]]:
        """Create sample slices for training."""
        slices = []

        for i in range(self.num_data):
            T = self._get_seq_length(i)

            if T - self.action_seq_len < 0:
                log.warning(f"Ignored short sequence #{i}: len={T}, window={self.action_seq_len}")
            else:
                slices += [
                    (i, start, start + self.action_seq_len)
                    for start in range(T - self.action_seq_len + 1)
                ]

        return slices

    def _get_seq_length(self, idx: int) -> int:
        """Get valid sequence length for a trajectory."""
        return int(self.masks[idx].sum().item())

    def get_all_actions(self) -> torch.Tensor:
        """Get all valid actions for computing statistics."""
        result = []
        for i in range(len(self.masks)):
            T = int(self.masks[i].sum().item())
            result.append(self.actions[i, :T, :])
        return torch.cat(result, dim=0)

    def _compute_action_stats(self):
        """Compute action normalization statistics (mean, std) for pos/rot only."""
        all_actions = self.get_all_actions()

        # Only normalize position/rotation (first 6 dims), not gripper
        pos_rot_actions = all_actions[:, :6]

        self.action_mean = pos_rot_actions.mean(dim=0)  # [6]
        self.action_std = pos_rot_actions.std(dim=0)    # [6]

        # Prevent division by zero
        self.action_std = torch.clamp(self.action_std, min=1e-6)

        log.info(f"Action stats - mean: {self.action_mean.numpy()}")
        log.info(f"Action stats - std: {self.action_std.numpy()}")

    def normalize_actions(self, actions: torch.Tensor) -> torch.Tensor:
        """
        Normalize actions (pos/rot only, gripper unchanged).

        Args:
            actions: [B, seq_len, 7] or [seq_len, 7]

        Returns:
            normalized actions
        """
        normalized = actions.clone()
        # Normalize only pos/rot (first 6 dims)
        normalized[..., :6] = (actions[..., :6] - self.action_mean.to(actions.device)) / self.action_std.to(actions.device)
        return normalized

    def denormalize_actions(self, actions: torch.Tensor) -> torch.Tensor:
        """
        Denormalize actions (pos/rot only, gripper unchanged).

        Args:
            actions: [B, seq_len, 7] or [seq_len, 7]

        Returns:
            denormalized actions
        """
        denormalized = actions.clone()
        # Denormalize only pos/rot (first 6 dims)
        denormalized[..., :6] = actions[..., :6] * self.action_std.to(actions.device) + self.action_mean.to(actions.device)
        return denormalized

    def get_action_stats(self) -> Dict[str, torch.Tensor]:
        """Get action normalization statistics."""
        return {
            'mean': self.action_mean,
            'std': self.action_std
        }

    def __len__(self) -> int:
        return len(self.slices)

    def __getitem__(self, idx: int) -> Dict:
        """
        Get a training sample.

        Returns:
            Dictionary containing:
              - obs: observation dict with images and embeddings
              - actions: action sequence [action_seq_len, action_dim]
              - prev_action: previous action [action_dim]
              - mask: validity mask
              - next_obs: next observation (for state prediction loss)
        """
        i, start, end = self.slices[idx]

        # Get images
        agentview = self.agentview_rgb[i][start:start + 1]
        eye_in_hand = self.eye_in_hand_rgb[i][start:start + 1]

        # Convert and normalize images
        agentview = torch.from_numpy(agentview).float().permute(0, 3, 1, 2) / 255.0
        eye_in_hand = torch.from_numpy(eye_in_hand).float().permute(0, 3, 1, 2) / 255.0

        # Resize images
        agentview = TF.resize(agentview, [self.image_size, self.image_size], antialias=True)
        eye_in_hand = TF.resize(eye_in_hand, [self.image_size, self.image_size], antialias=True)

        # Get language embedding
        task_emb = self.data_embs[i]
        if isinstance(task_emb, torch.Tensor):
            task_emb = task_emb.float()
        else:
            task_emb = torch.tensor(task_emb, dtype=torch.float32)

        # Get robot states
        robot_state = torch.from_numpy(self.robot_states[i][start:start + 1]).float()

        # Get actions
        actions = self.actions[i, start:end]
        mask = self.masks[i, start:end]

        # Get previous action (zeros if first step)
        if start > 0:
            prev_action = self.actions[i, start - 1]
        else:
            prev_action = torch.zeros(self.action_dim)

        # Build observation dict
        obs = {
            "agentview_image": agentview.squeeze(0),
            "eye_in_hand_image": eye_in_hand.squeeze(0),
            "lang_emb": task_emb,
            "robot_states": robot_state.squeeze(0)
        }

        # Get next observation for state prediction loss
        next_obs = None
        seq_length = self._get_seq_length(i)
        if start + 1 < seq_length:
            next_agentview = self.agentview_rgb[i][start + 1:start + 2]
            next_eye_in_hand = self.eye_in_hand_rgb[i][start + 1:start + 2]

            next_agentview = torch.from_numpy(next_agentview).float().permute(0, 3, 1, 2) / 255.0
            next_eye_in_hand = torch.from_numpy(next_eye_in_hand).float().permute(0, 3, 1, 2) / 255.0

            next_agentview = TF.resize(next_agentview, [self.image_size, self.image_size], antialias=True)
            next_eye_in_hand = TF.resize(next_eye_in_hand, [self.image_size, self.image_size], antialias=True)

            # Get next robot states
            next_robot_state = torch.from_numpy(self.robot_states[i][start + 1:start + 2]).float()

            next_obs = {
                "agentview_image": next_agentview.squeeze(0),
                "eye_in_hand_image": next_eye_in_hand.squeeze(0),
                "lang_emb": task_emb,
                "robot_states": next_robot_state.squeeze(0)
            }

        # Get future observations for multi-step rollout (JEPA)
        future_obs_list = []
        future_valid_mask = []
        for step in range(1, self.max_rollout_steps + 1):
            future_idx = start + step
            if future_idx < seq_length:
                future_agentview = self.agentview_rgb[i][future_idx:future_idx + 1]
                future_eye_in_hand = self.eye_in_hand_rgb[i][future_idx:future_idx + 1]

                future_agentview = torch.from_numpy(future_agentview).float().permute(0, 3, 1, 2) / 255.0
                future_eye_in_hand = torch.from_numpy(future_eye_in_hand).float().permute(0, 3, 1, 2) / 255.0

                future_agentview = TF.resize(future_agentview, [self.image_size, self.image_size], antialias=True)
                future_eye_in_hand = TF.resize(future_eye_in_hand, [self.image_size, self.image_size], antialias=True)

                future_robot_state = torch.from_numpy(self.robot_states[i][future_idx:future_idx + 1]).float()

                future_obs = {
                    "agentview_image": future_agentview.squeeze(0),
                    "eye_in_hand_image": future_eye_in_hand.squeeze(0),
                    "lang_emb": task_emb,
                    "robot_states": future_robot_state.squeeze(0)
                }
                future_obs_list.append(future_obs)
                future_valid_mask.append(1.0)
            else:
                # Padding with zeros for invalid future steps
                future_obs = {
                    "agentview_image": torch.zeros_like(obs["agentview_image"]),
                    "eye_in_hand_image": torch.zeros_like(obs["eye_in_hand_image"]),
                    "lang_emb": task_emb,
                    "robot_states": torch.zeros_like(obs["robot_states"])
                }
                future_obs_list.append(future_obs)
                future_valid_mask.append(0.0)

        # Normalize actions (pos/rot only, gripper unchanged)
        actions_normalized = self.normalize_actions(actions)
        prev_action_normalized = self.normalize_actions(prev_action.unsqueeze(0)).squeeze(0)

        return {
            "obs": obs,
            "actions": actions_normalized,
            "prev_action": prev_action_normalized,
            "mask": mask,
            "next_obs": next_obs,
            "future_obs_list": future_obs_list,
            "future_valid_mask": torch.tensor(future_valid_mask, dtype=torch.float32)
        }


def create_dataloader(
    data_directory: str,
    batch_size: int = 64,
    num_workers: int = 4,
    shuffle: bool = True,
    **dataset_kwargs
) -> Tuple[DataLoader, StateVLADataset]:
    """
    Create DataLoader for StateVLA training.

    Args:
        data_directory: Path to data
        batch_size: Batch size
        num_workers: Number of data loading workers
        shuffle: Whether to shuffle data
        **dataset_kwargs: Additional arguments for dataset

    Returns:
        (dataloader, dataset)
    """
    dataset = StateVLADataset(data_directory, **dataset_kwargs)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn
    )

    return dataloader, dataset


def collate_fn(batch: List[Dict]) -> Dict:
    """Custom collate function for batching."""
    obs_batch = {}
    next_obs_batch = {}

    # Collect observation keys
    obs_keys = batch[0]["obs"].keys()
    for key in obs_keys:
        obs_batch[key] = torch.stack([b["obs"][key] for b in batch])

    # Collect next observations (if available)
    has_next_obs = batch[0]["next_obs"] is not None
    if has_next_obs:
        for key in obs_keys:
            values = [b["next_obs"][key] if b["next_obs"] is not None else b["obs"][key] for b in batch]
            next_obs_batch[key] = torch.stack(values)

    # Collect future observations for multi-step rollout (JEPA)
    future_obs_list_batch = None
    future_valid_mask_batch = None
    if "future_obs_list" in batch[0] and batch[0]["future_obs_list"]:
        num_steps = len(batch[0]["future_obs_list"])
        future_obs_list_batch = []
        for step in range(num_steps):
            step_obs = {}
            for key in obs_keys:
                step_obs[key] = torch.stack([b["future_obs_list"][step][key] for b in batch])
            future_obs_list_batch.append(step_obs)
        future_valid_mask_batch = torch.stack([b["future_valid_mask"] for b in batch])

    return {
        "obs": obs_batch,
        "actions": torch.stack([b["actions"] for b in batch]),
        "prev_action": torch.stack([b["prev_action"] for b in batch]),
        "mask": torch.stack([b["mask"] for b in batch]),
        "next_obs": next_obs_batch if has_next_obs else None,
        "future_obs_list": future_obs_list_batch,
        "future_valid_mask": future_valid_mask_batch
    }
