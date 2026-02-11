"""
JEPA-based State Encoder for StateVLA.

Two-phase architecture:
  Phase 1 (Temporal JEPA):
    - Tokenizes obs_t → ContextEncoder → z_t
    - Tokenizes obs_{t+1} → TargetEncoder (EMA) → z_{t+1}
    - TemporalPredictor(z_t, a_t) → z'_{t+1}
    - Loss: MSE(z'_{t+1}, z_{t+1}) + VICReg

  Phase 2 (Frozen Encoder):
    - Tokenizes obs_t → FrozenEncoder → z_t
    - z_t used as condition for Flow Matching policy
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional

from jepa.tokenizer import MultiModalTokenizer
from jepa.encoder import ContextEncoder, TargetEncoder
from jepa.temporal_predictor import TemporalPredictor


class JEPAStateEncoder(nn.Module):
    """
    JEPA-based State Encoder with Temporal Prediction.

    Phase 1: Learns state representations via temporal prediction.
             "이 액션을 하면 미래가 어떻게 될까?" 학습
    Phase 2: Provides frozen z_t for policy learning.
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
        # Temporal Predictor config
        action_dim: int = 7,
        temporal_hidden_dim: int = 512,
        # Legacy predictor config (unused but kept for compatibility)
        predictor_embed_dim: int = 192,
        predictor_depth: int = 6,
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
            use_pretrained_vision=use_pretrained_vision,
            use_pretrained_language=use_pretrained_language,
            vision_model_name=vision_model_name,
            language_model_name=language_model_name,
            freeze_vision=freeze_vision,
            freeze_language=freeze_language,
            device=device,
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

        # Temporal Predictor: z_t + a_t → z'_{t+1}
        self.temporal_predictor = TemporalPredictor(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=temporal_hidden_dim,
        )

        # State projection (CLS -> state_dim)
        self.state_proj = nn.Sequential(
            nn.Linear(embed_dim, state_dim),
            nn.LayerNorm(state_dim),
        )

    def _encode_obs(
        self,
        obs_dict: Dict[str, torch.Tensor],
        use_target: bool = False,
    ) -> torch.Tensor:
        """
        Encode observation to state representation z.

        Args:
            obs_dict: Observation dictionary
            use_target: If True, use target encoder (for z_{t+1} target)

        Returns:
            z: [B, state_dim] state representation
        """
        tokens, _ = self.tokenizer(obs_dict)

        if use_target:
            with torch.no_grad():
                features = self.target_encoder(tokens)  # [B, N, D]
                cls_output = features[:, -1]  # CLS token (last position)
        else:
            # No masking - process all tokens
            _, cls_output = self.context_encoder(tokens, mask=None)

        z = self.state_proj(cls_output)
        return z

    def forward_temporal(
        self,
        obs_dict: Dict[str, torch.Tensor],
        next_obs_dict: Dict[str, torch.Tensor],
        action: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Phase 1 forward pass: Temporal JEPA prediction.

        "현재 상황(z_t)에서 이 액션(a_t)을 하면 미래가 어떻게 될까?"

        Args:
            obs_dict: Current observation (time t)
            next_obs_dict: Next observation (time t+1)
            action: [B, action_dim] action at time t

        Returns:
            Dictionary containing:
                - 'z_t': [B, state_dim] current state
                - 'z_next_pred': [B, state_dim] predicted next state
                - 'z_next_target': [B, state_dim] target next state (detached)
        """
        # Encode current observation → z_t
        z_t = self._encode_obs(obs_dict, use_target=False)

        # Encode next observation → z_{t+1} (target, no gradient)
        z_next_target = self._encode_obs(next_obs_dict, use_target=True)

        # Predict next state: z_t + a_t → z'_{t+1}
        z_next_pred = self.temporal_predictor(z_t, action)

        return {
            'z_t': z_t,
            'z_next_pred': z_next_pred,
            'z_next_target': z_next_target.detach(),
        }

    def forward(
        self,
        obs_dict: Dict[str, torch.Tensor],
        return_loss: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        Standard forward pass (Phase 2 / inference).
        No masking, no temporal prediction - just encode obs_t → z_t.

        Args:
            obs_dict: Observation dictionary
            return_loss: Ignored (kept for API compatibility)

        Returns:
            Dictionary containing:
                - 'z_t': [B, state_dim] state representation
        """
        z_t = self._encode_obs(obs_dict, use_target=False)
        return {'z_t': z_t}

    @torch.no_grad()
    def encode(self, obs_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Inference-only encoding.

        Args:
            obs_dict: Observation dictionary

        Returns:
            z_t: [B, state_dim] state representation
        """
        self.eval()
        z_t = self._encode_obs(obs_dict, use_target=False)
        return z_t

    @torch.no_grad()
    def update_target_encoder(self, momentum: float = 0.996):
        """Update target encoder via EMA."""
        self.target_encoder.update_ema(self.context_encoder, momentum)
