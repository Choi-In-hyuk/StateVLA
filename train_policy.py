"""
Training Policy: Training wrapper for StateVLA model.

Provides:
  - Loss computation (action loss + state prediction loss)
  - Training step management
  - Inference utilities
  - EMA support
"""

import os
import sys
_project_root = os.path.dirname(os.path.abspath(__file__))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Any, Tuple

from statevla_model import StateVLA, StateVLAWithEncoders


class StateVLATrainingModel(nn.Module):
    """
    Training wrapper for StateVLA.

    Handles:
      - Flow matching loss for actions
      - State prediction loss
      - Combined training objective
    """

    def __init__(
        self,
        model: StateVLAWithEncoders,
        # Loss weights
        action_loss_weight: float = 1.0,
        state_loss_weight: float = 0.1,
        # Flow matching configs
        use_ln_timestep: bool = False,
        # Training configs
        learning_rate: float = 1e-4,
        weight_decay: float = 0.05,
        # Inference configs
        sampling_steps: int = 4
    ):
        super().__init__()

        self.model = model
        self.action_loss_weight = action_loss_weight
        self.state_loss_weight = state_loss_weight
        self.use_ln_timestep = use_ln_timestep
        self.sampling_steps = sampling_steps

        # For tracking
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

        # Optional scaler for data normalization
        self.scaler = None

    def set_scaler(self, scaler):
        """Set data scaler for action normalization."""
        self.scaler = scaler

    def sample_timestep(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """
        Sample diffusion timesteps.

        Args:
            batch_size: Number of samples
            device: Device

        Returns:
            [B] timesteps in [0, 1]
        """
        if self.use_ln_timestep:
            # Log-normal distribution (more emphasis on middle timesteps)
            noise_t = torch.randn((batch_size,), device=device)
            timesteps = torch.sigmoid(noise_t)
        else:
            # Uniform distribution
            timesteps = torch.rand((batch_size,), device=device)

        return timesteps

    def compute_action_loss(
        self,
        velocity_pred: torch.Tensor,
        noise: torch.Tensor,
        gt_actions: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute flow matching loss for actions.

        Args:
            velocity_pred: [B, action_seq_len, action_dim] predicted velocity
            noise: [B, action_seq_len, action_dim] sampled noise
            gt_actions: [B, action_seq_len, action_dim] ground truth actions

        Returns:
            Scalar loss
        """
        # Target velocity: direction from clean to noisy
        target_velocity = noise - gt_actions

        # MSE loss
        loss = F.mse_loss(velocity_pred, target_velocity)

        return loss

    def compute_state_loss(
        self,
        z_next_pred: torch.Tensor,
        z_next_actual: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute state prediction loss.

        Args:
            z_next_pred: [B, state_dim] predicted next state
            z_next_actual: [B, state_dim] actual next state

        Returns:
            Scalar loss
        """
        # MSE loss (detach actual to prevent gradient flow)
        loss = F.mse_loss(z_next_pred, z_next_actual.detach())

        return loss

    def forward(
        self,
        obs_dict: Dict[str, torch.Tensor],
        prev_action: torch.Tensor,
        gt_actions: torch.Tensor,
        next_obs_dict: Optional[Dict[str, torch.Tensor]] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for training.

        Args:
            obs_dict: Current observation dictionary
            prev_action: [B, action_dim] previous action
            gt_actions: [B, action_seq_len, action_dim] ground truth actions
            next_obs_dict: Optional next observation for state loss

        Returns:
            Dictionary with losses and metrics
        """
        device = next(self.model.parameters()).device
        batch_size = gt_actions.shape[0]

        # Move to device
        gt_actions = gt_actions.to(device)
        prev_action = prev_action.to(device)

        # Sample timesteps
        sigma = self.sample_timestep(batch_size, device)

        # Forward pass through model
        outputs = self.model(obs_dict, prev_action, gt_actions, sigma)

        # Compute action loss
        action_loss = self.compute_action_loss(
            outputs['velocity'],
            outputs['noise'],
            gt_actions
        )

        # Compute state loss (if next observation provided)
        state_loss = torch.tensor(0.0, device=device)
        if next_obs_dict is not None and self.state_loss_weight > 0:
            # Get actual next state by encoding next observation
            with torch.no_grad():
                next_obs_features = self.model.obs_encoder(next_obs_dict)
                # Get language embedding (same as current)
                if 'lang' in obs_dict:
                    lang_emb = self.model.language_encoder(obs_dict['lang'])
                    if lang_emb.dim() == 3:
                        lang_emb = lang_emb.squeeze(1)
                else:
                    lang_emb = obs_dict['lang_emb'].to(device)

                # Compute actual next state
                z_next_actual = self.model.statevla.state_encoder(
                    next_obs_features,
                    lang_emb,
                    gt_actions[:, 0, :],  # First action becomes prev_action
                    outputs['z_t'].detach()
                )

            state_loss = self.compute_state_loss(outputs['z_next_pred'], z_next_actual)

        # Total loss
        total_loss = (
            self.action_loss_weight * action_loss +
            self.state_loss_weight * state_loss
        )

        return {
            'loss': total_loss,
            'action_loss': action_loss,
            'state_loss': state_loss,
            'z_t': outputs['z_t'],
            'z_next_pred': outputs['z_next_pred']
        }

    def training_step(
        self,
        batch: Dict[str, torch.Tensor],
        optimizer: torch.optim.Optimizer
    ) -> Dict[str, float]:
        """
        Execute a single training step.

        Args:
            batch: Training batch
            optimizer: Optimizer

        Returns:
            Dictionary with loss values
        """
        self.train()

        # Extract data from batch
        obs_dict = batch['obs']
        prev_action = batch.get('prev_action', torch.zeros(obs_dict[list(obs_dict.keys())[0]].shape[0], 7))
        gt_actions = batch['actions']
        next_obs_dict = batch.get('next_obs', None)

        # Reset state at episode boundaries if indicated
        if batch.get('reset_state', False):
            self.model.reset_state()

        # Forward pass
        optimizer.zero_grad()
        outputs = self.forward(obs_dict, prev_action, gt_actions, next_obs_dict)

        # Backward pass
        outputs['loss'].backward()
        optimizer.step()

        return {
            'loss': outputs['loss'].item(),
            'action_loss': outputs['action_loss'].item(),
            'state_loss': outputs['state_loss'].item()
        }

    @torch.no_grad()
    def predict(
        self,
        obs_dict: Dict[str, torch.Tensor],
        prev_action: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Generate actions at inference time.

        Args:
            obs_dict: Observation dictionary
            prev_action: Previous action (zeros if not provided)

        Returns:
            [B, action_seq_len, action_dim] actions
        """
        self.eval()
        device = next(self.model.parameters()).device

        if prev_action is None:
            batch_size = obs_dict[list(obs_dict.keys())[0]].shape[0]
            prev_action = torch.zeros(batch_size, self.model.statevla.action_dim, device=device)

        actions = self.model.predict(obs_dict, prev_action, self.sampling_steps)

        # Unscale if scaler available
        if self.scaler is not None:
            actions = self.scaler.inverse_scale_output(actions)

        return actions

    @torch.no_grad()
    def get_action(
        self,
        obs_dict: Dict[str, torch.Tensor],
        prev_action: Optional[torch.Tensor] = None,
        action_idx: int = -1
    ) -> torch.Tensor:
        """
        Get a single action for execution.

        Args:
            obs_dict: Observation dictionary
            prev_action: Previous action
            action_idx: Which action in sequence (-1 = last)

        Returns:
            [B, action_dim] single action
        """
        actions = self.predict(obs_dict, prev_action)
        return actions[:, action_idx, :]

    def reset_state(self):
        """Reset model internal state."""
        self.model.reset_state()

    def get_optimizer(self) -> torch.optim.Optimizer:
        """Create optimizer for training."""
        return torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )


class EMAWrapper:
    """
    Exponential Moving Average wrapper for model weights.
    """

    def __init__(self, model: nn.Module, decay: float = 0.995):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}

        # Initialize shadow weights
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        """Update shadow weights."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = (
                    self.decay * self.shadow[name] +
                    (1 - self.decay) * param.data
                )

    def apply_shadow(self):
        """Apply shadow weights to model (for evaluation)."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]

    def restore(self):
        """Restore original weights (after evaluation)."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]
        self.backup = {}


def create_training_model(
    model: StateVLAWithEncoders,
    action_loss_weight: float = 1.0,
    state_loss_weight: float = 0.1,
    learning_rate: float = 1e-4,
    weight_decay: float = 0.05,
    sampling_steps: int = 4,
    use_ema: bool = True,
    ema_decay: float = 0.995
) -> Tuple[StateVLATrainingModel, Optional[EMAWrapper]]:
    """
    Create training model with optional EMA.

    Args:
        model: StateVLA model
        action_loss_weight: Weight for action loss
        state_loss_weight: Weight for state prediction loss
        learning_rate: Learning rate
        weight_decay: Weight decay
        sampling_steps: Denoising steps for inference
        use_ema: Whether to use EMA
        ema_decay: EMA decay rate

    Returns:
        (training_model, ema_wrapper or None)
    """
    training_model = StateVLATrainingModel(
        model=model,
        action_loss_weight=action_loss_weight,
        state_loss_weight=state_loss_weight,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        sampling_steps=sampling_steps
    )

    ema = EMAWrapper(training_model, decay=ema_decay) if use_ema else None

    return training_model, ema
