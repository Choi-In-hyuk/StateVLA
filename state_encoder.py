"""
JEPA-based State Encoder for StateVLA.

Unified state encoder that:
1. Tokenizes all modalities (images, language, robot state)
2. Applies modality-aware masking
3. Encodes visible tokens via Mamba Context Encoder
4. Predicts masked token representations
5. Outputs z_t (CLS token) as the state representation

Training uses JEPA loss: MSE(predicted, target) + VICReg regularization
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple

from jepa.tokenizer import MultiModalTokenizer
from jepa.masking import ModalityAwareMasking, create_masking_strategy
from jepa.encoder import ContextEncoder, TargetEncoder, cosine_momentum_schedule
from jepa.predictor import JEPAPredictor


class JEPAStateEncoder(nn.Module):
    """
    JEPA-based State Encoder.

    Combines tokenization, masking, encoding, and prediction
    into a unified module for state representation learning.
    """

    def __init__(
        self,
        # Tokenizer config
        camera_names: List[str] = ['agentview', 'eye_in_hand'],
        image_size: int = 224,
        patch_size: int = 16,
        embed_dim: int = 256,
        lang_emb_dim: int = 512,
        robot_state_dim: int = 8,
        # Pretrained encoder config
        use_pretrained_vision: bool = False,
        use_pretrained_language: bool = False,
        vision_model_name: str = "google/siglip-base-patch16-224",
        language_model_name: str = "ViT-B/32",
        freeze_vision: bool = True,
        freeze_language: bool = True,
        # Encoder config
        encoder_depth: int = 12,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        # Predictor config
        predictor_embed_dim: int = 192,
        predictor_depth: int = 6,
        # Masking config
        mask_ratio: float = 0.5,
        masking_strategy: str = 'modality_aware',
        # Output config
        state_dim: int = 256,
        # Device
        device: str = 'cuda',
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.state_dim = state_dim
        self.device = device

        # Tokenizer
        self.tokenizer = MultiModalTokenizer(
            camera_names=camera_names,
            image_size=image_size,
            patch_size=patch_size,
            embed_dim=embed_dim,
            lang_emb_dim=lang_emb_dim,
            robot_state_dim=robot_state_dim,
            # Pretrained encoder options
            use_pretrained_vision=use_pretrained_vision,
            use_pretrained_language=use_pretrained_language,
            vision_model_name=vision_model_name,
            language_model_name=language_model_name,
            freeze_vision=freeze_vision,
            freeze_language=freeze_language,
            device=device,
        )

        # Masking strategy
        if masking_strategy == 'modality_aware':
            self.masking = ModalityAwareMasking(
                modality_mask_probs={0: 0.5, 1: 0.5, 2: 0.2, 3: 0.2},
                min_visible_modalities=1,
                max_masked_modalities=2,
            )
        else:
            self.masking = create_masking_strategy(
                masking_strategy,
                mask_ratio=mask_ratio,
            )

        # Context Encoder (Mamba)
        self.context_encoder = ContextEncoder(
            embed_dim=embed_dim,
            depth=encoder_depth,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            device=device,
        )

        # Target Encoder (EMA copy)
        self.target_encoder = TargetEncoder(self.context_encoder)

        # Predictor
        self.predictor = JEPAPredictor(
            embed_dim=embed_dim,
            predictor_embed_dim=predictor_embed_dim,
            depth=predictor_depth,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            device=device,
        )

        # State projection (CLS -> state_dim)
        self.state_proj = nn.Sequential(
            nn.Linear(embed_dim, state_dim),
            nn.LayerNorm(state_dim),
        )

    def forward(
        self,
        obs_dict: Dict[str, torch.Tensor],
        return_loss: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through JEPA State Encoder.

        Args:
            obs_dict: Dictionary containing:
                - 'agentview_image': [B, C, H, W]
                - 'eye_in_hand_image': [B, C, H, W]
                - 'lang_emb': [B, lang_emb_dim]
                - 'robot_states': [B, robot_state_dim]
            return_loss: Whether to compute JEPA loss components

        Returns:
            Dictionary containing:
                - 'z_t': [B, state_dim] state representation
                - 'predictions': [B, N_masked, D] (if return_loss)
                - 'targets': [B, N_masked, D] (if return_loss)
                - 'mask': [B, N] (if return_loss)
        """
        # 1. Tokenize all modalities
        tokens, modality_ids = self.tokenizer(obs_dict)
        B, N, D = tokens.shape
        device = tokens.device

        # 2. Generate mask
        if return_loss and self.training:
            mask = self.masking.generate_mask(
                batch_size=B,
                num_tokens=N,
                modality_ids=modality_ids,
                device=device,
            )
        else:
            # Inference: no masking
            mask = torch.zeros(B, N, dtype=torch.bool, device=device)

        # 3. Context Encoder (processes visible tokens)
        context_features, cls_output = self.context_encoder(tokens, mask)

        # 4. Project CLS to state representation
        z_t = self.state_proj(cls_output)

        outputs = {'z_t': z_t}

        if return_loss and self.training:
            # 5. Predictor (predict masked representations)
            predictions = self.predictor(context_features, mask)

            # 6. Target Encoder (get targets for masked tokens)
            with torch.no_grad():
                target_features = self.target_encoder(tokens)
                # Extract targets for masked positions (excluding CLS)
                targets = self._extract_masked_targets(target_features, mask)

            outputs['predictions'] = predictions
            outputs['targets'] = targets
            outputs['mask'] = mask

        return outputs

    def _extract_masked_targets(
        self,
        target_features: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Extract target features for masked positions.

        Args:
            target_features: [B, N, D] all token features from target encoder
            mask: [B, N] True = masked

        Returns:
            [B, max_masked, D] targets for masked positions
        """
        B, N, D = target_features.shape
        device = target_features.device

        # Count masked (excluding CLS at position 0)
        num_masked = mask[:, 1:].sum(dim=1)
        max_masked = num_masked.max().item()

        if max_masked == 0:
            return torch.zeros(B, 0, D, device=device)

        targets_list = []
        for b in range(B):
            # Get masked indices (excluding CLS)
            masked_idx = torch.where(mask[b, 1:])[0] + 1  # +1 to account for CLS
            n_masked = len(masked_idx)

            if n_masked > 0:
                target = target_features[b, masked_idx]  # [n_masked, D]
                # Pad if needed
                if n_masked < max_masked:
                    pad = torch.zeros(max_masked - n_masked, D, device=device)
                    target = torch.cat([target, pad], dim=0)
            else:
                target = torch.zeros(max_masked, D, device=device)

            targets_list.append(target)

        return torch.stack(targets_list, dim=0)

    @torch.no_grad()
    def encode(self, obs_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Inference-only encoding (no masking, no loss).

        Args:
            obs_dict: Observation dictionary

        Returns:
            z_t: [B, state_dim] state representation
        """
        self.eval()
        outputs = self.forward(obs_dict, return_loss=False)
        return outputs['z_t']

    @torch.no_grad()
    def update_target_encoder(self, momentum: float = 0.996):
        """Update target encoder via EMA."""
        self.target_encoder.update_ema(self.context_encoder, momentum)


def compute_jepa_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
    variance_weight: float = 1.0,
    covariance_weight: float = 0.04,
) -> Dict[str, torch.Tensor]:
    """
    Compute JEPA loss: MSE + VICReg regularization.

    Args:
        predictions: [B, N_masked, D] predicted representations
        targets: [B, N_masked, D] target representations
        mask: [B, N] original mask
        variance_weight: Weight for variance loss
        covariance_weight: Weight for covariance loss

    Returns:
        Dictionary with loss components
    """
    B, N_masked, D = predictions.shape

    if N_masked == 0:
        zero = torch.tensor(0.0, device=predictions.device)
        return {
            'total': zero,
            'mse': zero,
            'variance': zero,
            'covariance': zero,
        }

    # Count valid predictions per sample
    num_masked_per_sample = mask[:, 1:].sum(dim=1)  # [B]

    # MSE loss (invariance)
    # Only compute on valid (non-padded) predictions
    mse_loss = F.mse_loss(predictions, targets.detach(), reduction='none')
    # Mask out padded positions
    valid_mask = torch.zeros(B, N_masked, device=predictions.device)
    for b in range(B):
        valid_mask[b, :num_masked_per_sample[b].int()] = 1.0
    valid_mask = valid_mask.unsqueeze(-1)  # [B, N_masked, 1]
    mse_loss = (mse_loss * valid_mask).sum() / (valid_mask.sum() * D + 1e-8)

    # VICReg: Variance loss
    # Encourage each dimension to have variance >= 1
    pred_flat = predictions.reshape(-1, D)  # [B*N_masked, D]
    std = pred_flat.std(dim=0)  # [D]
    var_loss = F.relu(1.0 - std).mean()

    # VICReg: Covariance loss
    # Minimize off-diagonal elements of covariance matrix
    pred_centered = pred_flat - pred_flat.mean(dim=0, keepdim=True)
    cov = (pred_centered.T @ pred_centered) / (pred_flat.shape[0] - 1 + 1e-8)
    # Off-diagonal elements
    off_diag = cov - torch.diag(torch.diag(cov))
    cov_loss = (off_diag ** 2).sum() / D

    # Total loss
    total = mse_loss + variance_weight * var_loss + covariance_weight * cov_loss

    return {
        'total': total,
        'mse': mse_loss,
        'variance': var_loss,
        'covariance': cov_loss,
    }
