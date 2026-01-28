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
        camera_names: List[str] = None
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

        log.info(f"Loading dataset from {data_directory}")

        # Load language embeddings
        self.tasks = self._load_language_embeddings(language_embedding_path)

        # Load demonstrations
        self._load_demonstrations()

        # Create sample slices
        self.slices = self._get_slices()

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
        if start + 1 < self._get_seq_length(i):
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

        return {
            "obs": obs,
            "actions": actions,
            "prev_action": prev_action,
            "mask": mask,
            "next_obs": next_obs
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

    return {
        "obs": obs_batch,
        "actions": torch.stack([b["actions"] for b in batch]),
        "prev_action": torch.stack([b["prev_action"] for b in batch]),
        "mask": torch.stack([b["mask"] for b in batch]),
        "next_obs": next_obs_batch if has_next_obs else None
    }
