"""
StatePredictor: Predicts the next state using Mamba backbone.

This module takes the current state and predicts the next state,
enabling the model to anticipate future world states.
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


class StatePredictor(nn.Module):
    """
    State Predictor that uses Mamba to predict the next state.

    Takes:
      - z_t: Current state [B, state_dim]
      - state_history: Optional history of states [B, history_len, state_dim]

    Returns:
      - z_{t+1}^pred: Predicted next state [B, state_dim]
    """

    def __init__(
        self,
        state_dim: int = 256,
        n_layer: int = 4,
        d_intermediate: int = 256,
        history_len: int = 1,
        use_history: bool = False,
        dropout: float = 0.1,
        ssm_cfg: Optional[dict] = None,
        device: str = 'cuda'
    ):
        super().__init__()

        self.state_dim = state_dim
        self.history_len = history_len
        self.use_history = use_history

        # Default SSM configuration
        if ssm_cfg is None:
            ssm_cfg = {
                "layer": "Mamba1",
                "d_state": 64,
                "d_conv": 4,
                "expand": 2
            }

        # Input projection
        self.input_proj = nn.Linear(state_dim, state_dim)

        # Positional embedding for sequence
        max_seq_len = history_len + 1 if use_history else 2
        self.pos_emb = nn.Parameter(torch.randn(1, max_seq_len, state_dim) * 0.02)

        # Mamba backbone for state prediction
        self.mamba = MixerModel(
            d_model=state_dim,
            n_layer=n_layer,
            d_intermediate=d_intermediate,
            ssm_cfg=ssm_cfg,
            rms_norm=True,
            fused_add_norm=False,
            residual_in_fp32=False,
            device=device,
            dtype=torch.float32
        )

        # Prediction head
        self.pred_head = nn.Sequential(
            nn.Linear(state_dim, state_dim * 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(state_dim * 2, state_dim)
        )

        # Layer norm for output
        self.layer_norm = nn.LayerNorm(state_dim)

        # Residual connection weight (learnable)
        self.residual_weight = nn.Parameter(torch.tensor(0.5))

    def forward(
        self,
        z_t: torch.Tensor,
        state_history: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Predict the next state.

        Args:
            z_t: [B, state_dim] current state
            state_history: [B, history_len, state_dim] optional state history

        Returns:
            z_next_pred: [B, state_dim] predicted next state
        """
        batch_size = z_t.shape[0]

        # Project input
        z_t_proj = self.input_proj(z_t)

        if self.use_history and state_history is not None:
            # Concatenate history with current state
            # state_history: [B, history_len, state_dim]
            # z_t: [B, state_dim] -> [B, 1, state_dim]
            z_t_seq = z_t_proj.unsqueeze(1)
            seq = torch.cat([state_history, z_t_seq], dim=1)  # [B, history_len + 1, state_dim]
        else:
            # Create a simple 2-token sequence: [current, predict_token]
            # We use current state twice to create a sequence
            z_t_seq = z_t_proj.unsqueeze(1)  # [B, 1, state_dim]
            predict_token = torch.zeros_like(z_t_seq)  # [B, 1, state_dim]
            seq = torch.cat([z_t_seq, predict_token], dim=1)  # [B, 2, state_dim]

        # Add positional embeddings
        seq_len = seq.shape[1]
        seq = seq + self.pos_emb[:, :seq_len, :]

        # Pass through Mamba
        mamba_out = self.mamba(seq)  # [B, seq_len, state_dim]

        # Take the last token as prediction
        last_hidden = mamba_out[:, -1, :]  # [B, state_dim]

        # Prediction head
        z_next_pred = self.pred_head(last_hidden)

        # Residual connection with current state
        z_next_pred = self.residual_weight * z_next_pred + (1 - self.residual_weight) * z_t

        # Layer normalization
        z_next_pred = self.layer_norm(z_next_pred)

        return z_next_pred


class RecurrentStatePredictor(nn.Module):
    """
    Alternative State Predictor using recurrent Mamba updates.

    This version maintains an internal hidden state that gets
    updated with each new observation, similar to RNN-style processing.
    """

    def __init__(
        self,
        state_dim: int = 256,
        n_layer: int = 4,
        d_intermediate: int = 256,
        ssm_cfg: Optional[dict] = None,
        device: str = 'cuda'
    ):
        super().__init__()

        self.state_dim = state_dim

        if ssm_cfg is None:
            ssm_cfg = {
                "layer": "Mamba1",
                "d_state": 64,
                "d_conv": 4,
                "expand": 2
            }

        # Mamba for recurrent updates
        self.mamba = MixerModel(
            d_model=state_dim,
            n_layer=n_layer,
            d_intermediate=d_intermediate,
            ssm_cfg=ssm_cfg,
            rms_norm=True,
            device=device,
            dtype=torch.float32
        )

        # Gating mechanism
        self.gate = nn.Sequential(
            nn.Linear(state_dim * 2, state_dim),
            nn.Sigmoid()
        )

        # Update transform
        self.update_transform = nn.Sequential(
            nn.Linear(state_dim, state_dim),
            nn.Tanh()
        )

        self.layer_norm = nn.LayerNorm(state_dim)

    def forward(
        self,
        z_t: torch.Tensor,
        hidden_state: Optional[torch.Tensor] = None
    ) -> tuple:
        """
        Recurrent state prediction.

        Args:
            z_t: [B, state_dim] current state
            hidden_state: [B, state_dim] internal hidden state

        Returns:
            z_next_pred: [B, state_dim] predicted next state
            new_hidden_state: [B, state_dim] updated hidden state
        """
        batch_size = z_t.shape[0]

        if hidden_state is None:
            hidden_state = torch.zeros(batch_size, self.state_dim, device=z_t.device)

        # Create sequence from current state
        z_t_seq = z_t.unsqueeze(1)  # [B, 1, state_dim]

        # Pass through Mamba
        mamba_out = self.mamba(z_t_seq)  # [B, 1, state_dim]
        mamba_out = mamba_out.squeeze(1)  # [B, state_dim]

        # Gated update
        gate_input = torch.cat([mamba_out, hidden_state], dim=-1)
        gate = self.gate(gate_input)

        update = self.update_transform(mamba_out)

        # New hidden state: gated combination
        new_hidden_state = gate * update + (1 - gate) * hidden_state

        # Prediction is based on updated hidden state
        z_next_pred = self.layer_norm(new_hidden_state)

        return z_next_pred, new_hidden_state
