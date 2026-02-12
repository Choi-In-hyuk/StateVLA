"""
ActionPolicy: Residual action policy with Flow Matching base and MLP correction.

This module implements the Residual Action Policy:
  - Base Policy: Flow Matching (z_{t+1}^pred → a_base)
  - Correction: MLP (z_t + error → Δa)
  - Final action: a = a_base + Δa
"""

import os
import sys

_project_root = os.path.dirname(os.path.abspath(__file__))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import torch
import torch.nn as nn
from typing import Optional

from mamba import MixerModel
from utils import MLP, TimeEmbedding


class GripperClassifier(nn.Module):
    """
    Binary classifier for gripper action (open/close).
    Separate from Flow Matching to handle discrete nature of gripper.
    """

    def __init__(
        self,
        state_dim: int = 256,
        action_seq_len: int = 10,
        hidden_dim: int = 256,
    ):
        super().__init__()

        self.action_seq_len = action_seq_len

        self.classifier = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, action_seq_len),  # Output logits for each timestep
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Predict gripper logits.

        Args:
            z: [B, state_dim] state representation

        Returns:
            logits: [B, action_seq_len] gripper logits (>0 = close, <0 = open)
        """
        return self.classifier(z)

    def predict(self, z: torch.Tensor) -> torch.Tensor:
        """
        Predict gripper actions as binary values.

        Args:
            z: [B, state_dim] state representation

        Returns:
            gripper: [B, action_seq_len, 1] gripper actions (-1 or 1)
        """
        logits = self.forward(z)
        # Convert to -1 (open) or 1 (close)
        gripper = torch.where(
            logits > 0, torch.ones_like(logits), -torch.ones_like(logits)
        )
        return gripper.unsqueeze(-1)  # [B, action_seq_len, 1]


class FlowMatchingPolicy(nn.Module):
    """
    Flow Matching based policy for action generation.

    Takes the predicted next state and generates base actions
    using a diffusion-style flow matching approach.

    Note: Only handles position/rotation (6 dims), gripper is separate.
    """

    def __init__(
        self,
        state_dim: int = 256,
        action_dim: int = 6,  # Changed from 7 to 6 (no gripper)
        action_seq_len: int = 10,
        embed_dim: int = 256,
        n_layer: int = 3,
        d_intermediate: int = 256,
        ssm_cfg: Optional[dict] = None,
        device: str = "cuda",
    ):
        super().__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim  # 6 (position + rotation, no gripper)
        self.action_seq_len = action_seq_len
        self.embed_dim = embed_dim

        if ssm_cfg is None:
            ssm_cfg = {"layer": "Mamba1", "d_state": 64, "d_conv": 4, "expand": 2}

        # State projection
        self.state_proj = nn.Linear(state_dim, embed_dim)

        # Action embedding
        self.action_emb = nn.Linear(action_dim, embed_dim)

        # Timestep embedding for diffusion
        self.sigma_emb = TimeEmbedding(embed_dim)

        # Positional embeddings
        # Sequence: [sigma_emb, state_emb, action_emb_1, ..., action_emb_n]
        seq_len = 1 + 1 + action_seq_len  # sigma + state + actions
        self.pos_emb = nn.Parameter(torch.randn(1, seq_len, embed_dim) * 0.02)

        # Mamba backbone
        self.backbone = MixerModel(
            d_model=embed_dim,
            n_layer=n_layer,
            d_intermediate=d_intermediate,
            ssm_cfg=ssm_cfg,
            rms_norm=True,
            device=device,
            dtype=torch.float32,
        )

        # Action prediction head
        self.action_pred = MLP(
            input_dim=embed_dim,
            output_dim=action_dim,
            hidden_dim=embed_dim,
            num_layers=2,
        )

    def forward(
        self,
        z_next_pred: torch.Tensor,
        noisy_actions: torch.Tensor,
        sigma: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass for training (predicts velocity for flow matching).

        Args:
            z_next_pred: [B, state_dim] predicted next state
            noisy_actions: [B, action_seq_len, action_dim] noisy/interpolated actions
            sigma: [B] diffusion timestep

        Returns:
            velocity: [B, action_seq_len, action_dim] predicted velocity
        """
        batch_size = z_next_pred.shape[0]

        # Embed state: [B, 1, embed_dim]
        state_emb = self.state_proj(z_next_pred).unsqueeze(1)

        # Embed timestep: [B, 1, embed_dim]
        sigma_emb = self.sigma_emb(sigma)

        # Embed actions: [B, action_seq_len, embed_dim]
        action_emb = self.action_emb(noisy_actions)

        # Concatenate: [B, 2 + action_seq_len, embed_dim]
        seq = torch.cat([sigma_emb, state_emb, action_emb], dim=1)

        # Add positional embeddings
        seq = seq + self.pos_emb

        # Pass through backbone
        output = self.backbone(seq)  # [B, seq_len, embed_dim]

        # Extract action tokens (last action_seq_len tokens)
        action_output = output[
            :, -self.action_seq_len :, :
        ]  # [B, action_seq_len, embed_dim]

        # Predict velocity
        velocity = self.action_pred(action_output)  # [B, action_seq_len, action_dim]

        return velocity

    @torch.no_grad()
    def generate(
        self, z_next_pred: torch.Tensor, sample_steps: int = 4, cfg_scale: float = 1.0
    ) -> torch.Tensor:
        """
        Generate actions using flow matching sampling.

        Args:
            z_next_pred: [B, state_dim] predicted next state
            sample_steps: number of denoising steps
            cfg_scale: classifier-free guidance scale (not used in basic version)

        Returns:
            actions: [B, action_seq_len, action_dim] generated actions
        """
        batch_size = z_next_pred.shape[0]
        device = z_next_pred.device

        # Start from noise
        actions = torch.randn(
            batch_size, self.action_seq_len, self.action_dim, device=device
        )

        # Iterative denoising
        step_size = 1.0 / sample_steps

        for i in range(sample_steps, 0, -1):
            t = i / sample_steps
            sigma = torch.full((batch_size,), t, device=device)

            # Predict velocity
            velocity = self.forward(z_next_pred, actions, sigma)

            # Update actions (Euler step)
            actions = actions - step_size * velocity

        return actions


class CorrectionMLP(nn.Module):
    """
    MLP for computing action correction based on current state and prediction error.
    """

    def __init__(
        self,
        state_dim: int = 256,
        action_dim: int = 7,
        action_seq_len: int = 10,
        hidden_dim: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.action_dim = action_dim
        self.action_seq_len = action_seq_len

        # Input: concatenated z_t and error
        input_dim = state_dim * 2

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, action_dim * action_seq_len),
        )

        # Initialize last layer with small weights
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, z_t: torch.Tensor, error: torch.Tensor) -> torch.Tensor:
        """
        Compute action correction.

        Args:
            z_t: [B, state_dim] current state
            error: [B, state_dim] prediction error

        Returns:
            delta_a: [B, action_seq_len, action_dim] correction
        """
        batch_size = z_t.shape[0]

        # Concatenate inputs
        x = torch.cat([z_t, error], dim=-1)  # [B, state_dim * 2]

        # Predict correction
        delta_a = self.mlp(x)  # [B, action_dim * action_seq_len]

        # Reshape
        delta_a = delta_a.view(batch_size, self.action_seq_len, self.action_dim)

        return delta_a


class ActionPolicy(nn.Module):
    """
    Residual Action Policy combining Flow Matching base and MLP correction.

    Architecture:
      - Position/Rotation (6 dims): Flow Matching + optional correction
      - Gripper (1 dim): Binary classifier (separate head)

    Final action: a = [a_pos_rot, a_gripper]
    """

    def __init__(
        self,
        state_dim: int = 256,
        action_dim: int = 7,  # Total action dim (6 pos/rot + 1 gripper)
        action_seq_len: int = 10,
        embed_dim: int = 256,
        n_layer: int = 3,
        d_intermediate: int = 256,
        correction_hidden_dim: int = 512,
        ssm_cfg: Optional[dict] = None,
        use_correction: bool = True,
        device: str = "cuda",
    ):
        super().__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim  # 7 total
        self.pos_rot_dim = 6  # Position + Rotation
        self.gripper_dim = 1  # Gripper
        self.action_seq_len = action_seq_len
        self.use_correction = use_correction

        # Base policy: Flow Matching for position/rotation only
        self.base_policy = FlowMatchingPolicy(
            state_dim=state_dim,
            action_dim=self.pos_rot_dim,  # 6 dims only
            action_seq_len=action_seq_len,
            embed_dim=embed_dim,
            n_layer=n_layer,
            d_intermediate=d_intermediate,
            ssm_cfg=ssm_cfg,
            device=device,
        )

        # Gripper classifier (separate binary head)
        self.gripper_head = GripperClassifier(
            state_dim=state_dim,
            action_seq_len=action_seq_len,
            hidden_dim=embed_dim,
        )

        # Correction MLP for position/rotation
        if use_correction:
            self.correction = CorrectionMLP(
                state_dim=state_dim,
                action_dim=self.pos_rot_dim,  # 6 dims only
                action_seq_len=action_seq_len,
                hidden_dim=correction_hidden_dim,
            )
            # Learnable correction weight
            self.correction_weight = nn.Parameter(torch.tensor(0.1))
        else:
            self.correction = None

    def forward(
        self,
        z_t: torch.Tensor,
        z_next_pred: torch.Tensor,
        error: torch.Tensor,
        noisy_actions: torch.Tensor,
        sigma: torch.Tensor,
    ) -> tuple:
        """
        Forward pass for training.

        Args:
            z_t: [B, state_dim] current state
            z_next_pred: [B, state_dim] predicted next state
            error: [B, state_dim] prediction error
            noisy_actions: [B, action_seq_len, action_dim] noisy actions (7 dims)
            sigma: [B] diffusion timestep

        Returns:
            velocity: [B, action_seq_len, 6] predicted velocity for pos/rot
            gripper_logits: [B, action_seq_len] gripper logits
        """
        # Extract position/rotation part of noisy actions (first 6 dims)
        noisy_pos_rot = noisy_actions[:, :, : self.pos_rot_dim]

        # Base policy velocity for position/rotation
        velocity = self.base_policy(z_next_pred, noisy_pos_rot, sigma)

        # Add correction if enabled
        if self.use_correction and self.correction is not None:
            delta_a = self.correction(z_t, error)
            velocity = velocity + self.correction_weight * delta_a

        # Gripper classification (separate head)
        gripper_logits = self.gripper_head(z_t)

        return velocity, gripper_logits

    @torch.no_grad()
    def generate_actions(
        self,
        z_t: torch.Tensor,
        z_next_pred: torch.Tensor,
        error: torch.Tensor,
        sample_steps: int = 4,
    ) -> torch.Tensor:
        """
        Generate actions at inference time.

        Args:
            z_t: [B, state_dim] current state
            z_next_pred: [B, state_dim] predicted next state
            error: [B, state_dim] prediction error
            sample_steps: number of denoising steps

        Returns:
            actions: [B, action_seq_len, action_dim] generated actions (7 dims)
        """
        batch_size = z_t.shape[0]
        device = z_t.device

        # Start from noise for position/rotation only
        pos_rot_actions = torch.randn(
            batch_size, self.action_seq_len, self.pos_rot_dim, device=device  # 6 dims
        )

        # Iterative denoising for position/rotation
        step_size = 1.0 / sample_steps

        for i in range(sample_steps, 0, -1):
            t = i / sample_steps
            sigma = torch.full((batch_size,), t, device=device)

            # Create dummy full actions for forward pass
            dummy_gripper = torch.zeros(
                batch_size, self.action_seq_len, 1, device=device
            )
            full_actions = torch.cat([pos_rot_actions, dummy_gripper], dim=-1)

            # Predict velocity (only for pos/rot)
            velocity, _ = self.forward(z_t, z_next_pred, error, full_actions, sigma)

            # Update pos/rot actions (Euler step)
            pos_rot_actions = pos_rot_actions - step_size * velocity

        # Get gripper prediction (binary classification)
        gripper_actions = self.gripper_head.predict(z_t)  # [B, action_seq_len, 1]

        # Combine position/rotation and gripper
        actions = torch.cat(
            [pos_rot_actions, gripper_actions], dim=-1
        )  # [B, seq_len, 7]

        return actions


class ActionFlowMatching(nn.Module):
    """
    Wrapper class for Flow Matching training and inference.

    Handles the flow matching loss computation during training
    and action generation during inference.
    """

    def __init__(self, policy: ActionPolicy, ln: bool = False):
        super().__init__()
        self.policy = policy
        self.ln = ln  # Use log-normal timestep sampling

    def forward(
        self,
        actions: torch.Tensor,
        z_t: torch.Tensor,
        z_next_pred: torch.Tensor,
        error: torch.Tensor,
    ) -> tuple:
        """
        Compute flow matching loss for pos/rot and BCE loss for gripper.

        Args:
            actions: [B, action_seq_len, action_dim] ground truth actions (7 dims)
            z_t: [B, state_dim] current state
            z_next_pred: [B, state_dim] predicted next state
            error: [B, state_dim] prediction error

        Returns:
            total_loss: combined loss (pos_rot + gripper)
            pos_rot_loss: position/rotation flow matching loss
            gripper_loss: gripper BCE loss
        """
        batch_size = actions.shape[0]
        device = actions.device

        # Split actions into pos/rot and gripper
        pos_rot_actions = actions[:, :, :6]  # [B, seq_len, 6]
        gripper_actions = actions[:, :, 6]  # [B, seq_len]

        # --- Position/Rotation Flow Matching Loss ---
        # Sample timesteps
        if self.ln:
            noise_t = torch.randn((batch_size,), device=device)
            time_steps = torch.sigmoid(noise_t)
        else:
            time_steps = torch.rand((batch_size,), device=device)

        # Expand for broadcasting
        time_expanded = time_steps.view([batch_size, 1, 1])

        # Sample noise for pos/rot only
        noise = torch.randn_like(pos_rot_actions)

        # Interpolate: x_t = (1 - t) * x_0 + t * noise
        interpolated_pos_rot = (
            1 - time_expanded
        ) * pos_rot_actions + time_expanded * noise

        # Add dummy gripper for full action tensor
        dummy_gripper = torch.zeros(batch_size, actions.shape[1], 1, device=device)
        interpolated = torch.cat([interpolated_pos_rot, dummy_gripper], dim=-1)

        # Predict velocity and gripper logits
        velocity_pred, gripper_logits = self.policy(
            z_t, z_next_pred, error, interpolated, time_steps
        )

        # Target velocity for pos/rot: noise - actions
        target_velocity = noise - pos_rot_actions

        # MSE loss for pos/rot
        pos_rot_loss = ((target_velocity - velocity_pred) ** 2).mean()

        # --- Gripper BCE Loss ---
        # Convert gripper actions from {-1, 1} to {0, 1} for BCE
        gripper_targets = (gripper_actions > 0).float()  # [B, seq_len]

        # BCE loss
        gripper_loss = nn.functional.binary_cross_entropy_with_logits(
            gripper_logits, gripper_targets
        )

        # Combined loss
        total_loss = pos_rot_loss + gripper_loss

        return total_loss, pos_rot_loss, gripper_loss

    @torch.no_grad()
    def generate_actions(
        self,
        z_t: torch.Tensor,
        z_next_pred: torch.Tensor,
        error: torch.Tensor,
        sample_steps: int = 4,
    ) -> torch.Tensor:
        """
        Generate actions using the policy.
        """
        return self.policy.generate_actions(z_t, z_next_pred, error, sample_steps)
