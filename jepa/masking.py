"""
Masking strategies for JEPA training.

Supports:
- Random token masking
- Modality-aware masking (mask entire modalities)
"""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional


class MaskingStrategy:
    """Base class for masking strategies."""

    def generate_mask(
        self,
        batch_size: int,
        num_tokens: int,
        modality_ids: torch.Tensor,
        device: torch.device,
    ) -> torch.Tensor:
        """
        Generate a mask for tokens.

        Args:
            batch_size: Batch size
            num_tokens: Total number of tokens
            modality_ids: [B, num_tokens] modality ID for each token
            device: Device to create mask on

        Returns:
            mask: [B, num_tokens] - True = masked (to predict), False = visible
        """
        raise NotImplementedError


class RandomTokenMasking(MaskingStrategy):
    """
    Random token-level masking.

    Randomly masks a percentage of tokens regardless of modality.
    CLS token (modality_id = -1) is never masked.
    """

    def __init__(self, mask_ratio: float = 0.5):
        self.mask_ratio = mask_ratio

    def generate_mask(
        self,
        batch_size: int,
        num_tokens: int,
        modality_ids: torch.Tensor,
        device: torch.device,
    ) -> torch.Tensor:
        # Create random mask
        mask = torch.rand(batch_size, num_tokens, device=device) < self.mask_ratio

        # Never mask CLS token (modality_id = -1)
        cls_mask = modality_ids == -1
        mask = mask & ~cls_mask

        return mask


class ModalityAwareMasking(MaskingStrategy):
    """
    Modality-aware masking.

    Masks entire modalities with specified probabilities.
    At least one modality must remain visible for prediction.

    Modality IDs:
        -1: CLS token (never masked)
        0: agentview image
        1: wrist image
        2: language
        3: robot state
    """

    def __init__(
        self,
        modality_mask_probs: Optional[Dict[int, float]] = None,
        min_visible_modalities: int = 1,
        max_masked_modalities: int = 2,
    ):
        """
        Args:
            modality_mask_probs: Probability of masking each modality.
                Default: {0: 0.5, 1: 0.5, 2: 0.3, 3: 0.3}
            min_visible_modalities: Minimum number of modalities to keep visible
            max_masked_modalities: Maximum number of modalities to mask at once
        """
        self.modality_mask_probs = modality_mask_probs or {
            0: 0.5,  # agentview
            1: 0.5,  # wrist
            2: 0.3,  # language
            3: 0.3,  # robot state
        }
        self.min_visible = min_visible_modalities
        self.max_masked = max_masked_modalities
        self.modality_ids_list = list(self.modality_mask_probs.keys())

    def generate_mask(
        self,
        batch_size: int,
        num_tokens: int,
        modality_ids: torch.Tensor,
        device: torch.device,
    ) -> torch.Tensor:
        """
        Generate modality-aware mask.

        For each sample in the batch:
        1. Randomly select which modalities to mask based on probabilities
        2. Ensure at least min_visible modalities remain visible
        3. Mask all tokens belonging to selected modalities
        """
        mask = torch.zeros(batch_size, num_tokens, dtype=torch.bool, device=device)

        for b in range(batch_size):
            # Decide which modalities to mask for this sample
            masked_modalities = []

            for mod_id, prob in self.modality_mask_probs.items():
                if torch.rand(1).item() < prob:
                    masked_modalities.append(mod_id)

            # Ensure we don't mask too many
            if len(masked_modalities) > self.max_masked:
                # Randomly keep only max_masked
                indices = torch.randperm(len(masked_modalities))[:self.max_masked]
                masked_modalities = [masked_modalities[i] for i in indices]

            # Ensure at least min_visible modalities remain
            num_modalities = len(self.modality_ids_list)
            if num_modalities - len(masked_modalities) < self.min_visible:
                # Remove some from masked list
                num_to_keep = num_modalities - self.min_visible
                if num_to_keep > 0:
                    indices = torch.randperm(len(masked_modalities))[:num_to_keep]
                    masked_modalities = [masked_modalities[i] for i in indices]
                else:
                    masked_modalities = []

            # Ensure at least one modality is masked (for learning)
            if len(masked_modalities) == 0:
                # Randomly pick one to mask
                idx = torch.randint(len(self.modality_ids_list), (1,)).item()
                masked_modalities = [self.modality_ids_list[idx]]

            # Create mask for this sample
            sample_modality_ids = modality_ids[b]
            for mod_id in masked_modalities:
                mask[b] = mask[b] | (sample_modality_ids == mod_id)

        return mask


class BlockMasking(MaskingStrategy):
    """
    Block masking for images (I-JEPA style).

    Masks contiguous blocks within image patches.
    Only applies to image modalities.
    """

    def __init__(
        self,
        num_patches_per_side: int = 14,
        num_blocks: int = 4,
        block_scale: Tuple[float, float] = (0.15, 0.4),
        block_aspect_ratio: Tuple[float, float] = (0.75, 1.5),
        image_modality_ids: List[int] = [0, 1],
    ):
        """
        Args:
            num_patches_per_side: Number of patches per side (14 for 224/16)
            num_blocks: Number of blocks to mask per image
            block_scale: (min, max) scale of block relative to image
            block_aspect_ratio: (min, max) aspect ratio of block
            image_modality_ids: Modality IDs that are images
        """
        self.num_patches = num_patches_per_side
        self.num_blocks = num_blocks
        self.block_scale = block_scale
        self.block_aspect_ratio = block_aspect_ratio
        self.image_modality_ids = image_modality_ids

    def _sample_block_size(self) -> Tuple[int, int]:
        """Sample random block size."""
        scale = torch.empty(1).uniform_(*self.block_scale).item()
        aspect = torch.empty(1).uniform_(*self.block_aspect_ratio).item()

        block_area = self.num_patches * self.num_patches * scale
        h = int(round((block_area * aspect) ** 0.5))
        w = int(round((block_area / aspect) ** 0.5))

        h = min(max(h, 1), self.num_patches)
        w = min(max(w, 1), self.num_patches)

        return h, w

    def _sample_block_position(self, h: int, w: int) -> Tuple[int, int]:
        """Sample random block position."""
        top = torch.randint(0, self.num_patches - h + 1, (1,)).item()
        left = torch.randint(0, self.num_patches - w + 1, (1,)).item()
        return top, left

    def generate_mask(
        self,
        batch_size: int,
        num_tokens: int,
        modality_ids: torch.Tensor,
        device: torch.device,
    ) -> torch.Tensor:
        mask = torch.zeros(batch_size, num_tokens, dtype=torch.bool, device=device)

        for b in range(batch_size):
            sample_modality_ids = modality_ids[b]

            # For each image modality
            for img_mod_id in self.image_modality_ids:
                img_token_mask = sample_modality_ids == img_mod_id
                img_indices = torch.where(img_token_mask)[0]

                if len(img_indices) == 0:
                    continue

                # Create 2D mask for this image
                img_mask_2d = torch.zeros(
                    self.num_patches, self.num_patches,
                    dtype=torch.bool, device=device
                )

                # Sample and apply blocks
                for _ in range(self.num_blocks):
                    h, w = self._sample_block_size()
                    top, left = self._sample_block_position(h, w)
                    img_mask_2d[top:top+h, left:left+w] = True

                # Convert 2D mask to 1D indices
                img_mask_1d = img_mask_2d.flatten()

                # Apply to token mask
                mask[b, img_indices] = img_mask_1d[:len(img_indices)]

        return mask


def create_masking_strategy(
    strategy_name: str,
    **kwargs
) -> MaskingStrategy:
    """
    Factory function to create masking strategy.

    Args:
        strategy_name: 'random', 'modality_aware', or 'block'
        **kwargs: Arguments for the specific strategy

    Returns:
        MaskingStrategy instance
    """
    if strategy_name == 'random':
        return RandomTokenMasking(**kwargs)
    elif strategy_name == 'modality_aware':
        return ModalityAwareMasking(**kwargs)
    elif strategy_name == 'block':
        return BlockMasking(**kwargs)
    else:
        raise ValueError(f"Unknown masking strategy: {strategy_name}")
