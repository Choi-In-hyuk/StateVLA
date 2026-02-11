"""
StateVLA: State-based Vision-Language-Action Model with Two-Phase Training

Phase 1 - Temporal JEPA (표현 학습):
  obs_t → Encoder → z_t
  z_t + a_t → TemporalPredictor → z'_{t+1}
  obs_{t+1} → TargetEncoder (EMA) → z_{t+1}
  Loss: MSE(z'_{t+1}, z_{t+1}) + VICReg

Phase 2 - Flow Matching (정책 학습):
  obs_t → FrozenEncoder → z_t
  z_t → FlowMatchingPolicy → a_t
  Loss: FlowMatching(velocity) + BCE(gripper)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, List

from state_encoder import JEPAStateEncoder
from jepa.temporal_predictor import compute_temporal_jepa_loss
from action_policy import ActionPolicy


class StateVLA(nn.Module):
    """
    Two-Phase StateVLA Model.

    Phase 1: Trains encoder + temporal predictor (learns physics/causality)
    Phase 2: Freezes encoder, trains action policy (learns smooth actions)
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
        # Predictor config (legacy, kept for compatibility)
        predictor_embed_dim: int = 192,
        predictor_depth: int = 6,
        # Masking config (legacy, kept for compatibility)
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
        # Temporal predictor config
        temporal_hidden_dim: int = 512,
        # Training phase
        training_phase: int = 1,
        # Device
        device: str = 'cuda',
    ):
        super().__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_seq_len = action_seq_len
        self.training_phase = training_phase
        self.device = device

        # JEPA State Encoder (used in both phases)
        self.state_encoder = JEPAStateEncoder(
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
            encoder_depth=encoder_depth,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            action_dim=action_dim,
            temporal_hidden_dim=temporal_hidden_dim,
            predictor_embed_dim=predictor_embed_dim,
            predictor_depth=predictor_depth,
            mask_ratio=mask_ratio,
            masking_strategy=masking_strategy,
            state_dim=state_dim,
            device=device,
        )

        # Action Policy (only needed for Phase 2 and inference)
        self.action_policy = ActionPolicy(
            state_dim=state_dim,
            action_dim=action_dim,
            action_seq_len=action_seq_len,
            embed_dim=policy_embed_dim,
            n_layer=policy_layers,
            d_intermediate=policy_embed_dim,
            use_correction=False,
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

    def freeze_encoder(self):
        """Freeze state encoder for Phase 2 training."""
        for param in self.state_encoder.parameters():
            param.requires_grad = False
        self.training_phase = 2

    def unfreeze_encoder(self):
        """Unfreeze state encoder (e.g., for fine-tuning)."""
        for param in self.state_encoder.parameters():
            param.requires_grad = True

    def forward_phase1(
        self,
        obs_dict: Dict[str, torch.Tensor],
        next_obs_dict: Dict[str, torch.Tensor],
        action: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Phase 1 forward pass: Temporal JEPA.

        Args:
            obs_dict: Current observation
            next_obs_dict: Next observation
            action: [B, action_dim] action at time t

        Returns:
            Dictionary with z_t, z_next_pred, z_next_target
        """
        return self.state_encoder.forward_temporal(obs_dict, next_obs_dict, action)

    def forward_phase2(
        self,
        obs_dict: Dict[str, torch.Tensor],
        gt_actions: torch.Tensor,
        sigma: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Phase 2 forward pass: Flow Matching policy.

        Args:
            obs_dict: Current observation
            gt_actions: [B, action_seq_len, action_dim] ground truth actions
            sigma: [B] diffusion timestep

        Returns:
            Dictionary with z_t, velocity, noise, gripper_logits, gripper_targets
        """
        batch_size = gt_actions.shape[0]

        # Encode state (frozen encoder)
        with torch.no_grad():
            state_outputs = self.state_encoder(obs_dict)
        z_t = state_outputs['z_t']

        outputs = {'z_t': z_t}

        # Split actions: pos/rot (6 dims) and gripper (1 dim)
        pos_rot_actions = gt_actions[:, :, :6]
        gripper_actions = gt_actions[:, :, 6]

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

        outputs['velocity'] = velocity
        outputs['noise'] = noise
        outputs['gripper_logits'] = gripper_logits
        outputs['gripper_targets'] = gripper_actions

        return outputs

    def forward(
        self,
        obs_dict: Dict[str, torch.Tensor],
        gt_actions: Optional[torch.Tensor] = None,
        sigma: Optional[torch.Tensor] = None,
        next_obs_dict: Optional[Dict[str, torch.Tensor]] = None,
        action: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Unified forward pass that routes to phase-specific methods.

        Phase 1: requires next_obs_dict, action
        Phase 2: requires gt_actions, sigma
        """
        if self.training_phase == 1 and next_obs_dict is not None and action is not None:
            return self.forward_phase1(obs_dict, next_obs_dict, action)
        elif gt_actions is not None and sigma is not None:
            return self.forward_phase2(obs_dict, gt_actions, sigma)
        else:
            # Inference / encode only
            state_outputs = self.state_encoder(obs_dict)
            return {'z_t': state_outputs['z_t']}

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

        # Encode state
        z_t = self.state_encoder.encode(obs_dict)

        # Generate actions
        z_next_dummy = torch.zeros_like(z_t)
        error_dummy = torch.zeros_like(z_t)

        actions = self.action_policy.generate_actions(
            z_t, z_next_dummy, error_dummy, sample_steps
        )

        # Denormalize pos/rot actions
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
    Two-Phase Training wrapper for StateVLA.

    Phase 1: Temporal JEPA loss (MSE + VICReg)
    Phase 2: Action loss (Flow Matching + Gripper BCE)
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
            velocity_pred: [B, seq_len, 6] predicted velocity
            noise: [B, seq_len, 6] sampled noise
            pos_rot_actions: [B, seq_len, 6] ground truth pos/rot
            gripper_logits: [B, seq_len] gripper prediction logits
            gripper_targets: [B, seq_len] ground truth gripper (-1 or 1)

        Returns:
            total_loss, pos_rot_loss, gripper_loss
        """
        # Position/Rotation Flow Matching loss
        target_velocity = noise - pos_rot_actions
        pos_rot_loss = F.mse_loss(velocity_pred, target_velocity)

        # Gripper BCE loss (convert from {-1, 1} to {0, 1})
        gripper_binary = (gripper_targets > 0).float()
        gripper_loss = F.binary_cross_entropy_with_logits(gripper_logits, gripper_binary)

        total_loss = pos_rot_loss + gripper_loss
        return total_loss, pos_rot_loss, gripper_loss

    def forward_phase1(
        self,
        obs_dict: Dict[str, torch.Tensor],
        next_obs_dict: Dict[str, torch.Tensor],
        action: torch.Tensor,
        step: Optional[int] = None,
        total_steps: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Phase 1 training: Temporal JEPA loss.

        Args:
            obs_dict: Current observation
            next_obs_dict: Next observation
            action: [B, action_dim] action at time t
            step: Current training step
            total_steps: Total training steps

        Returns:
            Dictionary with losses
        """
        outputs = self.model.forward_phase1(obs_dict, next_obs_dict, action)

        # Temporal JEPA loss
        jepa_losses = compute_temporal_jepa_loss(
            outputs['z_next_pred'],
            outputs['z_next_target'],
            variance_weight=self.variance_weight,
            covariance_weight=self.covariance_weight,
        )

        total_loss = self.jepa_loss_weight * jepa_losses['total']

        return {
            'loss': total_loss,
            'jepa_loss': jepa_losses['total'],
            'jepa_mse': jepa_losses['mse'],
            'jepa_variance': jepa_losses['variance'],
            'jepa_covariance': jepa_losses['covariance'],
            'action_loss': torch.tensor(0.0, device=total_loss.device),
            'pos_rot_loss': torch.tensor(0.0, device=total_loss.device),
            'gripper_loss': torch.tensor(0.0, device=total_loss.device),
            'z_t': outputs['z_t'],
        }

    def forward_phase2(
        self,
        obs_dict: Dict[str, torch.Tensor],
        gt_actions: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Phase 2 training: Flow Matching action loss.

        Args:
            obs_dict: Current observation
            gt_actions: [B, action_seq_len, action_dim] ground truth actions

        Returns:
            Dictionary with losses
        """
        batch_size = gt_actions.shape[0]
        device = gt_actions.device

        # Sample sigma for flow matching
        sigma = torch.rand(batch_size, device=device)

        outputs = self.model.forward_phase2(obs_dict, gt_actions, sigma)

        # Action loss
        action_loss, pos_rot_loss, gripper_loss = self.compute_action_loss(
            outputs['velocity'],
            outputs['noise'],
            gt_actions[:, :, :6],
            outputs['gripper_logits'],
            outputs['gripper_targets'],
        )

        total_loss = self.action_loss_weight * action_loss

        return {
            'loss': total_loss,
            'jepa_loss': torch.tensor(0.0, device=device),
            'jepa_mse': torch.tensor(0.0, device=device),
            'jepa_variance': torch.tensor(0.0, device=device),
            'jepa_covariance': torch.tensor(0.0, device=device),
            'action_loss': action_loss,
            'pos_rot_loss': pos_rot_loss,
            'gripper_loss': gripper_loss,
            'z_t': outputs['z_t'],
        }

    def forward(
        self,
        obs_dict: Dict[str, torch.Tensor],
        gt_actions: Optional[torch.Tensor] = None,
        step: Optional[int] = None,
        total_steps: Optional[int] = None,
        # Phase 1 specific
        next_obs_dict: Optional[Dict[str, torch.Tensor]] = None,
        action: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Unified training forward pass.

        Phase 1: requires next_obs_dict, action
        Phase 2: requires gt_actions
        """
        if self.model.training_phase == 1:
            return self.forward_phase1(obs_dict, next_obs_dict, action, step, total_steps)
        else:
            return self.forward_phase2(obs_dict, gt_actions)

    def update_target_encoder(self, step: int, total_steps: int):
        """Update target encoder with scheduled momentum."""
        if self.ema_momentum_schedule == 'cosine':
            import math
            progress = step / total_steps
            momentum = 1.0 - (1.0 - self.ema_momentum) * (
                1 + math.cos(math.pi * progress)
            ) / 2
        else:
            momentum = self.ema_momentum

        self.model.update_target_encoder(momentum)
