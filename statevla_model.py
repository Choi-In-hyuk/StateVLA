"""
StateVLA: State-based Vision-Language-Action Model

Main model class that integrates:
  - Vision Encoder (ResNet)
  - Language Encoder (CLIP)
  - State Encoder (fusion)
  - State Predictor (Mamba)
  - Action Policy (Residual Flow Matching)

Key innovation: Predicts next state before generating actions,
enabling feedback-based correction.
"""

import os
import sys
_project_root = os.path.dirname(os.path.abspath(__file__))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import torch
import torch.nn as nn
from typing import Dict, Optional, Any, List

from state_encoder import StateEncoder, CrossAttentionStateEncoder
from state_predictor import StatePredictor
from action_policy import ActionPolicy


class StateVLA(nn.Module):
    """
    StateVLA: Vision-Language-Action model with state prediction.

    Architecture:
        obs + lang + prev_action → StateEncoder → z_t
        z_t → StatePredictor → z_{t+1}^pred
        z_t + z_{t+1}^pred + error → ActionPolicy → action

    The model maintains internal state buffers for closed-loop operation.
    """

    def __init__(
        self,
        # Encoder configs
        latent_dim: int = 256,
        lang_emb_dim: int = 512,
        obs_tok_len: int = 2,
        # State configs
        state_encoder_type: str = "mlp",
        state_dim: int = 256,
        state_predictor_layers: int = 4,
        # Action configs
        action_dim: int = 7,
        action_seq_len: int = 10,
        # Policy configs
        policy_layers: int = 3,
        policy_embed_dim: int = 256,
        use_correction: bool = True,
        # General configs
        dropout: float = 0.1,
        device: str = 'cuda'
    ):
        super().__init__()

        self.latent_dim = latent_dim
        self.lang_emb_dim = lang_emb_dim
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_seq_len = action_seq_len
        self.obs_tok_len = obs_tok_len
        self.device = device

        # State Encoder: obs + lang + prev_action → z_t
        if state_encoder_type == "cross_attention":
            self.state_encoder = CrossAttentionStateEncoder(
                latent_dim=latent_dim,
                lang_emb_dim=lang_emb_dim,
                action_dim=action_dim,
                state_dim=state_dim,
                obs_tok_len=obs_tok_len,
                num_heads=4,
                dropout=dropout
            )
        else:
            self.state_encoder = StateEncoder(
                latent_dim=latent_dim,
                lang_emb_dim=lang_emb_dim,
                action_dim=action_dim,
                state_dim=state_dim,
                obs_tok_len=obs_tok_len,
                use_prev_state=True,
                dropout=dropout
            )

        # State Predictor: z_t → z_{t+1}^pred
        self.state_predictor = StatePredictor(
            state_dim=state_dim,
            n_layer=state_predictor_layers,
            d_intermediate=state_dim,
            use_history=False,
            dropout=dropout,
            device=device
        )

        # Action Policy: z_t + z_{t+1}^pred + error → action
        self.action_policy = ActionPolicy(
            state_dim=state_dim,
            action_dim=action_dim,
            action_seq_len=action_seq_len,
            embed_dim=policy_embed_dim,
            n_layer=policy_layers,
            d_intermediate=policy_embed_dim,
            use_correction=use_correction,
            device=device
        )

        # Internal state buffers for closed-loop operation
        self.register_buffer('prev_state', None)
        self.register_buffer('prev_pred_state', None)

        # External encoders (set by model_factory)
        self.obs_encoder = None
        self.language_encoder = None

    def reset_state(self):
        """Reset internal state buffers. Call at the beginning of each episode."""
        self.prev_state = None
        self.prev_pred_state = None

    def compute_error(self, z_t: torch.Tensor) -> torch.Tensor:
        """
        Compute prediction error from previous step.

        Args:
            z_t: [B, state_dim] current actual state

        Returns:
            error: [B, state_dim] prediction error
        """
        # Check batch size compatibility to avoid dimension mismatch
        if self.prev_pred_state is not None and self.prev_pred_state.shape[0] == z_t.shape[0]:
            error = self.prev_pred_state - z_t
        else:
            error = torch.zeros_like(z_t)
        return error

    def forward(
        self,
        obs_features: torch.Tensor,
        lang_emb: torch.Tensor,
        prev_action: torch.Tensor,
        gt_actions: Optional[torch.Tensor] = None,
        sigma: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for training.

        Args:
            obs_features: [B, obs_tok_len, latent_dim] encoded observations
            lang_emb: [B, lang_emb_dim] or [B, 1, lang_emb_dim] language embedding
            prev_action: [B, action_dim] previous action
            gt_actions: [B, action_seq_len, action_dim] ground truth actions (for training)
            sigma: [B] diffusion timestep (for training)

        Returns:
            Dictionary containing:
                - z_t: current state
                - z_next_pred: predicted next state
                - error: prediction error
                - velocity (if training): predicted velocity
        """
        batch_size = obs_features.shape[0]

        # Get previous state (or initialize)
        # Check batch size compatibility to avoid dimension mismatch
        if self.prev_state is not None and self.prev_state.shape[0] == batch_size:
            prev_state = self.prev_state
        else:
            prev_state = None

        # 1. Compute current state
        z_t = self.state_encoder(obs_features, lang_emb, prev_action, prev_state)

        # 2. Compute prediction error
        error = self.compute_error(z_t)

        # 3. Predict next state
        z_next_pred = self.state_predictor(z_t)

        # 4. Update internal buffers
        self.prev_state = z_t.detach()
        self.prev_pred_state = z_next_pred.detach()

        outputs = {
            'z_t': z_t,
            'z_next_pred': z_next_pred,
            'error': error
        }

        # 5. If training, compute velocity for flow matching
        if gt_actions is not None and sigma is not None:
            # Sample noise and interpolate
            noise = torch.randn_like(gt_actions)
            sigma_expanded = sigma.view([batch_size, 1, 1])
            noisy_actions = (1 - sigma_expanded) * gt_actions + sigma_expanded * noise

            # Predict velocity
            velocity = self.action_policy(z_t, z_next_pred, error, noisy_actions, sigma)
            outputs['velocity'] = velocity
            outputs['noise'] = noise

        return outputs

    @torch.no_grad()
    def predict(
        self,
        obs_features: torch.Tensor,
        lang_emb: torch.Tensor,
        prev_action: torch.Tensor,
        sample_steps: int = 4
    ) -> torch.Tensor:
        """
        Generate actions at inference time.

        Args:
            obs_features: [B, obs_tok_len, latent_dim] encoded observations
            lang_emb: [B, lang_emb_dim] language embedding
            prev_action: [B, action_dim] previous action
            sample_steps: number of denoising steps

        Returns:
            actions: [B, action_seq_len, action_dim] generated actions
        """
        self.eval()
        batch_size = obs_features.shape[0]

        # Get previous state
        # Check batch size compatibility to avoid dimension mismatch
        if self.prev_state is not None and self.prev_state.shape[0] == batch_size:
            prev_state = self.prev_state
        else:
            prev_state = None

        # 1. Compute current state
        z_t = self.state_encoder(obs_features, lang_emb, prev_action, prev_state)

        # 2. Compute error
        error = self.compute_error(z_t)

        # 3. Predict next state
        z_next_pred = self.state_predictor(z_t)

        # 4. Update buffers
        self.prev_state = z_t.detach()
        self.prev_pred_state = z_next_pred.detach()

        # 5. Generate actions
        actions = self.action_policy.generate_actions(
            z_t, z_next_pred, error, sample_steps
        )

        return actions

    def get_action(
        self,
        obs_features: torch.Tensor,
        lang_emb: torch.Tensor,
        prev_action: torch.Tensor,
        sample_steps: int = 4,
        action_idx: int = -1
    ) -> torch.Tensor:
        """
        Get a single action for execution.

        Args:
            obs_features: [B, obs_tok_len, latent_dim]
            lang_emb: [B, lang_emb_dim]
            prev_action: [B, action_dim]
            sample_steps: denoising steps
            action_idx: which action in sequence to return (-1 = last)

        Returns:
            action: [B, action_dim] single action
        """
        actions = self.predict(obs_features, lang_emb, prev_action, sample_steps)
        return actions[:, action_idx, :]


class StateVLAWithEncoders(nn.Module):
    """
    Full StateVLA model with integrated vision and language encoders.

    This is the complete model that takes raw observations and language
    instructions as input.
    """

    def __init__(
        self,
        statevla: StateVLA,
        obs_encoder: nn.Module,
        language_encoder: nn.Module
    ):
        super().__init__()
        self.statevla = statevla
        self.obs_encoder = obs_encoder
        self.language_encoder = language_encoder

    def reset_state(self):
        """Reset internal state."""
        self.statevla.reset_state()

    def forward(
        self,
        obs_dict: Dict[str, torch.Tensor],
        prev_action: torch.Tensor,
        gt_actions: Optional[torch.Tensor] = None,
        sigma: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Full forward pass from raw inputs.

        Args:
            obs_dict: Dictionary with camera images and language
            prev_action: [B, action_dim] previous action
            gt_actions: [B, action_seq_len, action_dim] ground truth (training)
            sigma: [B] diffusion timestep (training)
        """
        device = next(self.statevla.parameters()).device

        # Encode observations
        obs_features = self.obs_encoder(obs_dict)  # [B, obs_tok_len, latent_dim]

        # Encode language
        if 'lang' in obs_dict:
            lang_emb = self.language_encoder(obs_dict['lang'])  # [B, 1, lang_emb_dim]
            if lang_emb.dim() == 3:
                lang_emb = lang_emb.squeeze(1)
        elif 'lang_emb' in obs_dict:
            lang_emb = obs_dict['lang_emb'].to(device)
        else:
            raise ValueError("obs_dict must contain 'lang' or 'lang_emb'")

        return self.statevla(obs_features, lang_emb, prev_action, gt_actions, sigma)

    @torch.no_grad()
    def predict(
        self,
        obs_dict: Dict[str, torch.Tensor],
        prev_action: torch.Tensor,
        sample_steps: int = 4
    ) -> torch.Tensor:
        """Generate actions from raw inputs."""
        device = next(self.statevla.parameters()).device

        # Encode observations
        obs_features = self.obs_encoder(obs_dict)

        # Encode language
        if 'lang' in obs_dict:
            lang_emb = self.language_encoder(obs_dict['lang'])
            if lang_emb.dim() == 3:
                lang_emb = lang_emb.squeeze(1)
        elif 'lang_emb' in obs_dict:
            lang_emb = obs_dict['lang_emb'].to(device)
        else:
            raise ValueError("obs_dict must contain 'lang' or 'lang_emb'")

        return self.statevla.predict(obs_features, lang_emb, prev_action, sample_steps)

    def get_action(
        self,
        obs_dict: Dict[str, torch.Tensor],
        prev_action: torch.Tensor,
        sample_steps: int = 4,
        action_idx: int = -1
    ) -> torch.Tensor:
        """Get single action from raw inputs."""
        actions = self.predict(obs_dict, prev_action, sample_steps)
        return actions[:, action_idx, :]
