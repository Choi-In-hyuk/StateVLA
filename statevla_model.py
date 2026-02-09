"""
StateVLA: State-based Vision-Language-Action Model with JEPA

JEPA-based architecture:
  - Multimodal Tokenization (ViT patches + language + robot state)
  - Mamba-based Context Encoder with masking
  - Target Encoder (EMA) for self-supervised learning
  - JEPA Predictor for masked token prediction
  - Flow Matching Action Policy

Key features:
  - Learns rich state representations via masked prediction
  - No explicit world model (z_t → z_{t+1}) - focuses on current state quality
  - VICReg regularization prevents representation collapse
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, List

from state_encoder import JEPAStateEncoder, compute_jepa_loss
from action_policy import ActionPolicy


class StateVLA(nn.Module):
    """
    JEPA-based StateVLA Model.

    Architecture:
        obs_dict → JEPAStateEncoder → z_t
        z_t → ActionPolicy (Flow Matching) → action

    Training:
        - JEPA loss: MSE(predicted_masked, target_masked) + VICReg
        - Action loss: Flow Matching loss
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
        # State config
        state_dim: int = 256,
        # Action config
        action_dim: int = 7,
        action_seq_len: int = 10,
        # Policy config
        policy_layers: int = 3,
        policy_embed_dim: int = 256,
        # Device
        device: str = 'cuda',
    ):
        super().__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_seq_len = action_seq_len
        self.device = device

        # JEPA State Encoder
        self.state_encoder = JEPAStateEncoder(
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
            # Encoder config
            encoder_depth=encoder_depth,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            predictor_embed_dim=predictor_embed_dim,
            predictor_depth=predictor_depth,
            mask_ratio=mask_ratio,
            masking_strategy=masking_strategy,
            state_dim=state_dim,
            device=device,
        )

        # Action Policy (Flow Matching)
        # Simplified: only uses z_t, no z_next_pred or error
        self.action_policy = ActionPolicy(
            state_dim=state_dim,
            action_dim=action_dim,
            action_seq_len=action_seq_len,
            embed_dim=policy_embed_dim,
            n_layer=policy_layers,
            d_intermediate=policy_embed_dim,
            use_correction=False,  # No correction in JEPA mode
            device=device,
        )

        # Action normalization buffers (pos/rot only, 6 dims)
        self.register_buffer('action_mean', torch.zeros(6))
        self.register_buffer('action_std', torch.ones(6))

    def set_action_stats(self, mean: torch.Tensor, std: torch.Tensor):
        """Set action normalization statistics from dataset."""
        self.action_mean.copy_(mean)
        self.action_std.copy_(std)

    def denormalize_actions(self, actions: torch.Tensor) -> torch.Tensor:
        """Denormalize pos/rot actions (gripper unchanged)."""
        denormalized = actions.clone()
        denormalized[..., :6] = actions[..., :6] * self.action_std + self.action_mean
        return denormalized

    def forward(
        self,
        obs_dict: Dict[str, torch.Tensor],
        gt_actions: Optional[torch.Tensor] = None,
        sigma: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for training.

        Args:
            obs_dict: Dictionary containing:
                - 'agentview_image': [B, C, H, W]
                - 'eye_in_hand_image': [B, C, H, W]
                - 'lang_emb': [B, lang_emb_dim]
                - 'robot_states': [B, robot_state_dim]
            gt_actions: [B, action_seq_len, action_dim] ground truth actions
            sigma: [B] diffusion timestep

        Returns:
            Dictionary containing:
                - 'z_t': [B, state_dim] state representation
                - 'predictions': [B, N_masked, D] JEPA predictions
                - 'targets': [B, N_masked, D] JEPA targets
                - 'mask': [B, N] mask used
                - 'velocity': [B, action_seq_len, action_dim] (if training)
                - 'noise': [B, action_seq_len, action_dim] (if training)
        """
        batch_size = obs_dict['lang_emb'].shape[0]

        # 1. JEPA State Encoding
        state_outputs = self.state_encoder(obs_dict, return_loss=True)
        z_t = state_outputs['z_t']

        outputs = {
            'z_t': z_t,
            'predictions': state_outputs.get('predictions'),
            'targets': state_outputs.get('targets'),
            'mask': state_outputs.get('mask'),
        }

        # 2. Action Policy (if training)
        if gt_actions is not None and sigma is not None:
            # Split actions: pos/rot (6 dims) and gripper (1 dim)
            pos_rot_actions = gt_actions[:, :, :6]
            gripper_actions = gt_actions[:, :, 6]  # [B, seq_len]

            # Sample noise for flow matching (pos/rot only)
            noise = torch.randn_like(pos_rot_actions)
            sigma_expanded = sigma.view(batch_size, 1, 1)
            noisy_pos_rot = (1 - sigma_expanded) * pos_rot_actions + sigma_expanded * noise

            # Create full noisy actions with dummy gripper
            dummy_gripper = torch.zeros(batch_size, gt_actions.shape[1], 1, device=z_t.device)
            noisy_actions = torch.cat([noisy_pos_rot, dummy_gripper], dim=-1)

            # Predict velocity and gripper logits
            z_next_dummy = torch.zeros_like(z_t)
            error_dummy = torch.zeros_like(z_t)

            velocity, gripper_logits = self.action_policy(
                z_t, z_next_dummy, error_dummy, noisy_actions, sigma
            )

            outputs['velocity'] = velocity  # [B, seq_len, 6] pos/rot velocity
            outputs['noise'] = noise        # [B, seq_len, 6] pos/rot noise
            outputs['gripper_logits'] = gripper_logits  # [B, seq_len]
            outputs['gripper_targets'] = gripper_actions  # [B, seq_len]

        return outputs

    @torch.no_grad()
    def predict(
        self,
        obs_dict: Dict[str, torch.Tensor],
        sample_steps: int = 4,
    ) -> torch.Tensor:
        """
        Generate actions at inference time.

        Args:
            obs_dict: Observation dictionary
            sample_steps: Number of denoising steps

        Returns:
            actions: [B, action_seq_len, action_dim]
        """
        self.eval()

        # Encode state (no masking in inference)
        z_t = self.state_encoder.encode(obs_dict)

        # Generate actions (normalized)
        z_next_dummy = torch.zeros_like(z_t)
        error_dummy = torch.zeros_like(z_t)

        actions = self.action_policy.generate_actions(
            z_t, z_next_dummy, error_dummy, sample_steps
        )

        # Denormalize pos/rot actions (gripper is already binary)
        actions = self.denormalize_actions(actions)

        return actions

    def get_action(
        self,
        obs_dict: Dict[str, torch.Tensor],
        sample_steps: int = 4,
        action_idx: int = -1,
    ) -> torch.Tensor:
        """
        Get a single action for execution.

        Args:
            obs_dict: Observation dictionary
            sample_steps: Denoising steps
            action_idx: Which action in sequence to return (-1 = last)

        Returns:
            action: [B, action_dim]
        """
        actions = self.predict(obs_dict, sample_steps)
        return actions[:, action_idx, :]

    @torch.no_grad()
    def update_target_encoder(self, momentum: float = 0.996):
        """Update target encoder via EMA."""
        self.state_encoder.update_target_encoder(momentum)


class StateVLATrainer(nn.Module):
    """
    Training wrapper for StateVLA with JEPA losses.

    Handles:
    - JEPA loss computation (MSE + VICReg)
    - Action loss computation (Flow Matching)
    - EMA target encoder updates
    """

    def __init__(
        self,
        model: StateVLA,
        jepa_loss_weight: float = 1.0,
        action_loss_weight: float = 1.0,
        variance_weight: float = 1.0,
        covariance_weight: float = 0.04,
        ema_momentum: float = 0.996,
        ema_momentum_schedule: str = 'cosine',
    ):
        super().__init__()
        self.model = model
        self.jepa_loss_weight = jepa_loss_weight
        self.action_loss_weight = action_loss_weight
        self.variance_weight = variance_weight
        self.covariance_weight = covariance_weight
        self.ema_momentum = ema_momentum
        self.ema_momentum_schedule = ema_momentum_schedule

    def compute_action_loss(
        self,
        velocity_pred: torch.Tensor,
        noise: torch.Tensor,
        pos_rot_actions: torch.Tensor,
        gripper_logits: torch.Tensor,
        gripper_targets: torch.Tensor,
    ) -> tuple:
        """
        Compute Flow Matching loss for pos/rot and BCE loss for gripper.

        Args:
            velocity_pred: [B, seq_len, 6] predicted velocity for pos/rot
            noise: [B, seq_len, 6] sampled noise for pos/rot
            pos_rot_actions: [B, seq_len, 6] ground truth pos/rot actions
            gripper_logits: [B, seq_len] gripper prediction logits
            gripper_targets: [B, seq_len] ground truth gripper actions (-1 or 1)

        Returns:
            total_loss: combined loss
            pos_rot_loss: position/rotation loss
            gripper_loss: gripper BCE loss
        """
        # Position/Rotation Flow Matching loss
        target_velocity = noise - pos_rot_actions
        pos_rot_loss = F.mse_loss(velocity_pred, target_velocity)

        # Gripper BCE loss (convert from {-1, 1} to {0, 1})
        gripper_binary = (gripper_targets > 0).float()
        gripper_loss = F.binary_cross_entropy_with_logits(gripper_logits, gripper_binary)

        total_loss = pos_rot_loss + gripper_loss

        return total_loss, pos_rot_loss, gripper_loss

    def forward(
        self,
        obs_dict: Dict[str, torch.Tensor],
        gt_actions: torch.Tensor,
        step: Optional[int] = None,
        total_steps: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Training forward pass.

        Args:
            obs_dict: Observation dictionary
            gt_actions: [B, action_seq_len, action_dim] ground truth actions
            step: Current training step (for EMA schedule)
            total_steps: Total training steps

        Returns:
            Dictionary with losses and outputs
        """
        batch_size = gt_actions.shape[0]
        device = gt_actions.device

        # Sample sigma for flow matching
        sigma = torch.rand(batch_size, device=device)

        # Forward pass
        outputs = self.model(obs_dict, gt_actions, sigma)

        # JEPA loss
        if outputs['predictions'] is not None and outputs['targets'] is not None:
            jepa_losses = compute_jepa_loss(
                outputs['predictions'],
                outputs['targets'],
                outputs['mask'],
                variance_weight=self.variance_weight,
                covariance_weight=self.covariance_weight,
            )
            jepa_loss = jepa_losses['total']
        else:
            jepa_loss = torch.tensor(0.0, device=device)
            jepa_losses = {'mse': jepa_loss, 'variance': jepa_loss, 'covariance': jepa_loss}

        # Action loss
        if outputs['velocity'] is not None:
            action_loss, pos_rot_loss, gripper_loss = self.compute_action_loss(
                outputs['velocity'],
                outputs['noise'],
                gt_actions[:, :, :6],  # pos/rot only
                outputs['gripper_logits'],
                outputs['gripper_targets'],
            )
        else:
            action_loss = torch.tensor(0.0, device=device)
            pos_rot_loss = torch.tensor(0.0, device=device)
            gripper_loss = torch.tensor(0.0, device=device)

        # Total loss
        total_loss = (
            self.jepa_loss_weight * jepa_loss +
            self.action_loss_weight * action_loss
        )

        return {
            'loss': total_loss,
            'jepa_loss': jepa_loss,
            'jepa_mse': jepa_losses['mse'],
            'jepa_variance': jepa_losses['variance'],
            'jepa_covariance': jepa_losses['covariance'],
            'action_loss': action_loss,
            'pos_rot_loss': pos_rot_loss,
            'gripper_loss': gripper_loss,
            'z_t': outputs['z_t'],
        }

    def update_target_encoder(self, step: int, total_steps: int):
        """
        Update target encoder with scheduled momentum.
        """
        if self.ema_momentum_schedule == 'cosine':
            import math
            progress = step / total_steps
            momentum = 1.0 - (1.0 - self.ema_momentum) * (
                1 + math.cos(math.pi * progress)
            ) / 2
        else:
            momentum = self.ema_momentum

        self.model.update_target_encoder(momentum)
