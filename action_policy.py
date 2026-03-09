"""
ActionPolicy: Action policy with Flow Matching for all 7 action dimensions.

Handles full 7D actions (pos/rot + gripper) with a unified flow matching loss.
This avoids the temporal-memory issue of a separate gripper classifier.
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


class FlowMatchingPolicy(nn.Module):
    """
    Flow Matching based policy for action generation.

    Takes the predicted next state and generates actions
    using a diffusion-style flow matching approach.

    Handles all action dimensions (pos/rot + gripper).
    """

    def __init__(
        self,
        state_dim: int = 256,
        action_dim: int = 7,
        action_seq_len: int = 10,
        embed_dim: int = 256,
        n_layer: int = 3,
        d_intermediate: int = 256,
        ssm_cfg: Optional[dict] = None,
        device: str = "cuda",
    ):
        super().__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_seq_len = action_seq_len
        self.embed_dim = embed_dim

        if ssm_cfg is None:
            ssm_cfg = {"layer": "Mamba1", "d_state": 64, "d_conv": 4, "expand": 2}

        # State projection: takes concat([z_t, z_goal]) → state_dim * 2
        self.state_proj = nn.Linear(state_dim * 2, embed_dim)

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

        # Spatial cross-attention: action tokens attend to image patch features
        self.spatial_cross_attn = nn.MultiheadAttention(
            embed_dim=embed_dim, num_heads=4, batch_first=True, dropout=0.0
        )
        self.spatial_cross_attn_norm = nn.LayerNorm(embed_dim)

    def forward(
        self,
        z_state: torch.Tensor,
        noisy_actions: torch.Tensor,
        sigma: torch.Tensor,
        spatial_features: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Forward pass for training (predicts velocity for flow matching).

        Args:
            z_state: [B, state_dim * 2] concat([z_t, z_goal])
            noisy_actions: [B, action_seq_len, action_dim] noisy/interpolated actions
            sigma: [B] diffusion timestep
            spatial_features: [B, N_patches, embed_dim] image patch features (optional)

        Returns:
            velocity: [B, action_seq_len, action_dim] predicted velocity
        """
        # Embed state: [B, 1, embed_dim]
        state_emb = self.state_proj(z_state).unsqueeze(1)

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
        action_output = output[:, -self.action_seq_len:, :]  # [B, action_seq_len, embed_dim]

        # Cross-attention to spatial patch features
        if spatial_features is not None:
            ca_out, _ = self.spatial_cross_attn(
                action_output, spatial_features, spatial_features
            )
            action_output = self.spatial_cross_attn_norm(action_output + ca_out)

        # Predict velocity
        velocity = self.action_pred(action_output)  # [B, action_seq_len, action_dim]

        return velocity

    @torch.no_grad()
    def generate(
        self, z_state: torch.Tensor, sample_steps: int = 4, cfg_scale: float = 1.0,
        spatial_features: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Generate actions using flow matching sampling.

        Args:
            z_state: [B, state_dim * 2] concat([z_t, z_goal])
            sample_steps: number of denoising steps

        Returns:
            actions: [B, action_seq_len, action_dim] generated actions
        """
        batch_size = z_state.shape[0]
        device = z_state.device

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
            velocity = self.forward(z_state, actions, sigma, spatial_features=spatial_features)

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
    Action Policy using Flow Matching for all 7 action dimensions.

    Architecture:
      - Full 7D Flow Matching (pos/rot + gripper unified)
      - Optional correction MLP

    Final action: a = a_base + correction (if use_correction=True)
    """

    def __init__(
        self,
        state_dim: int = 256,
        action_dim: int = 7,
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
        self.action_dim = action_dim
        self.action_seq_len = action_seq_len
        self.use_correction = use_correction

        # Base policy: Flow Matching for all action dims
        self.base_policy = FlowMatchingPolicy(
            state_dim=state_dim,
            action_dim=action_dim,
            action_seq_len=action_seq_len,
            embed_dim=embed_dim,
            n_layer=n_layer,
            d_intermediate=d_intermediate,
            ssm_cfg=ssm_cfg,
            device=device,
        )

        # Correction MLP (optional)
        if use_correction:
            self.correction = CorrectionMLP(
                state_dim=state_dim,
                action_dim=action_dim,
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
        z_goal: torch.Tensor,
        noisy_actions: torch.Tensor,
        sigma: torch.Tensor,
        spatial_features: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Forward pass for training.

        Args:
            z_t: [B, state_dim] current state
            z_goal: [B, state_dim] predicted goal state (from GoalPredictor)
            noisy_actions: [B, action_seq_len, action_dim] noisy actions (7 dims)
            sigma: [B] diffusion timestep
            spatial_features: [B, N_patches, embed_dim] image patch features (optional)

        Returns:
            velocity: [B, action_seq_len, action_dim] predicted velocity
        """
        z_state = torch.cat([z_t, z_goal], dim=-1)  # [B, state_dim * 2]
        velocity = self.base_policy(z_state, noisy_actions, sigma, spatial_features=spatial_features)

        if self.use_correction and self.correction is not None:
            error = z_goal - z_t  # goal direction as correction signal
            delta_a = self.correction(z_t, error)
            velocity = velocity + self.correction_weight * delta_a

        return velocity

    @torch.no_grad()
    def generate_actions(
        self,
        z_t: torch.Tensor,
        z_goal: torch.Tensor,
        sample_steps: int = 4,
        spatial_features: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Generate actions at inference time.

        Args:
            z_t: [B, state_dim] current state
            z_goal: [B, state_dim] predicted goal state
            sample_steps: number of denoising steps
            spatial_features: [B, N_patches, embed_dim] image patch features (optional)
        """
        batch_size = z_t.shape[0]
        device = z_t.device

        actions = torch.randn(
            batch_size, self.action_seq_len, self.action_dim, device=device
        )

        step_size = 1.0 / sample_steps

        for i in range(sample_steps, 0, -1):
            t = i / sample_steps
            sigma = torch.full((batch_size,), t, device=device)
            velocity = self.forward(z_t, z_goal, actions, sigma,
                                    spatial_features=spatial_features)
            actions = actions - step_size * velocity

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
        z_goal: torch.Tensor,
        spatial_features: torch.Tensor = None,
    ) -> tuple:
        """
        Compute unified flow matching loss for all 7 action dims.

        Args:
            actions: [B, action_seq_len, action_dim] ground truth actions (7 dims)
            z_t: [B, state_dim] current state
            z_goal: [B, state_dim] predicted goal state (from GoalPredictor)
            spatial_features: [B, N_patches, embed_dim] optional spatial features

        Returns:
            (flow_loss, sigma): flow matching MSE loss and sampled sigma
        """
        batch_size = actions.shape[0]
        device = actions.device

        # Sample timesteps
        if self.ln:
            noise_t = torch.randn((batch_size,), device=device)
            time_steps = torch.sigmoid(noise_t)
        else:
            time_steps = torch.rand((batch_size,), device=device)

        # Expand for broadcasting
        time_expanded = time_steps.view([batch_size, 1, 1])

        # Sample noise for all 7 dims
        noise = torch.randn_like(actions)

        # Interpolate: x_t = (1 - t) * x_0 + t * noise
        interpolated = (1 - time_expanded) * actions + time_expanded * noise

        # Predict velocity
        velocity_pred = self.policy(z_t, z_goal, interpolated, time_steps,
                                    spatial_features=spatial_features)

        # Target velocity: noise - actions
        target_velocity = noise - actions

        # MSE loss for all 7 dims
        flow_loss = ((target_velocity - velocity_pred) ** 2).mean()

        return flow_loss, time_steps

    @torch.no_grad()
    def generate_actions(
        self,
        z_t: torch.Tensor,
        z_goal: torch.Tensor,
        sample_steps: int = 4,
        spatial_features: torch.Tensor = None,
    ) -> torch.Tensor:
        """Generate actions using the policy."""
        return self.policy.generate_actions(z_t, z_goal, sample_steps,
                                            spatial_features=spatial_features)
