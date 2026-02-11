"""
Temporal Predictor for JEPA Phase 1.

Predicts future state z_{t+1} from current state z_t and action a_t.
Core idea: "현재 상황(z_t)에서 이 액션(a_t)을 하면 미래가 어떻게 될까?"

Uses residual prediction: z'_{t+1} = z_t + delta
  - Small actions cause small state changes
  - Easier to learn than absolute prediction
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict


class TemporalPredictor(nn.Module):
    """
    Predicts z_{t+1} from z_t and a_t.

    Architecture:
        z_t → state_proj → hidden
        a_t → action_proj → hidden
        concat(state_hidden, action_hidden) → MLP → delta
        z'_{t+1} = z_t + delta  (residual prediction)
    """

    def __init__(
        self,
        state_dim: int = 256,
        action_dim: int = 7,
        hidden_dim: int = 512,
    ):
        """
        Args:
            state_dim: Dimension of state representation z_t
            action_dim: Dimension of action (7 = 6 pos/rot + 1 gripper)
            hidden_dim: Hidden dimension of predictor MLP
        """
        super().__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim

        # Project state and action to hidden space
        self.state_proj = nn.Linear(state_dim, hidden_dim)
        self.action_proj = nn.Linear(action_dim, hidden_dim)

        # Predictor MLP: predicts delta (residual)
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, state_dim),
        )

        self._init_weights()

    def _init_weights(self):
        # Initialize last layer with small weights for stable residual start
        nn.init.zeros_(self.predictor[-1].weight)
        nn.init.zeros_(self.predictor[-1].bias)

    def forward(
        self,
        z_t: torch.Tensor,
        a_t: torch.Tensor,
    ) -> torch.Tensor:
        """
        Predict next state from current state and action.

        Args:
            z_t: [B, state_dim] current state representation
            a_t: [B, action_dim] action taken at time t

        Returns:
            z_next_pred: [B, state_dim] predicted next state
        """
        state_emb = self.state_proj(z_t)     # [B, hidden_dim]
        action_emb = self.action_proj(a_t)   # [B, hidden_dim]

        combined = torch.cat([state_emb, action_emb], dim=-1)  # [B, hidden_dim*2]
        delta = self.predictor(combined)      # [B, state_dim]

        # Residual prediction
        z_next_pred = z_t + delta

        return z_next_pred


def compute_temporal_jepa_loss(
    z_next_pred: torch.Tensor,
    z_next_target: torch.Tensor,
    variance_weight: float = 1.0,
    covariance_weight: float = 0.04,
) -> Dict[str, torch.Tensor]:
    """
    Compute Temporal JEPA loss: MSE + VICReg regularization.

    Args:
        z_next_pred: [B, state_dim] predicted next state
        z_next_target: [B, state_dim] target next state (from target encoder, detached)
        variance_weight: Weight for variance loss (prevents collapse)
        covariance_weight: Weight for covariance loss (decorrelation)

    Returns:
        Dictionary with loss components
    """
    B, D = z_next_pred.shape

    # MSE loss (invariance) - match predicted to target
    mse_loss = F.mse_loss(z_next_pred, z_next_target.detach())

    # VICReg: Variance loss - encourage each dimension to have std >= 1
    std_pred = z_next_pred.std(dim=0)  # [D]
    var_loss = F.relu(1.0 - std_pred).mean()

    # VICReg: Covariance loss - minimize off-diagonal covariance
    pred_centered = z_next_pred - z_next_pred.mean(dim=0, keepdim=True)
    cov = (pred_centered.T @ pred_centered) / (B - 1 + 1e-8)  # [D, D]
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
