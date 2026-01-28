"""
StateEncoder: Fuses observations, language, and previous action into a latent state.

This module takes multi-modal inputs and produces a unified state representation
that captures the current world state for the StateVLA model.
"""

import torch
import torch.nn as nn
from typing import Optional


class StateEncoder(nn.Module):
    """
    State Encoder that fuses observation, language, and action information.

    Takes:
      - obs_features: Visual features from camera images [B, obs_tok_len, latent_dim]
      - lang_emb: Language embedding [B, lang_emb_dim] or [B, 1, lang_emb_dim]
      - prev_action: Previous action taken [B, action_dim]
      - prev_state: Previous state (optional) [B, state_dim]

    Returns:
      - z_t: Current state latent [B, state_dim]
    """

    def __init__(
        self,
        latent_dim: int = 256,
        lang_emb_dim: int = 512,
        action_dim: int = 7,
        state_dim: int = 256,
        obs_tok_len: int = 2,
        use_prev_state: bool = True,
        dropout: float = 0.1
    ):
        super().__init__()

        self.latent_dim = latent_dim
        self.lang_emb_dim = lang_emb_dim
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.obs_tok_len = obs_tok_len
        self.use_prev_state = use_prev_state

        # Projection layers for each modality
        self.obs_proj = nn.Linear(latent_dim * obs_tok_len, state_dim)
        self.lang_proj = nn.Linear(lang_emb_dim, state_dim)
        self.action_proj = nn.Linear(action_dim, state_dim)

        if use_prev_state:
            self.prev_state_proj = nn.Linear(state_dim, state_dim)
            fusion_input_dim = state_dim * 4
        else:
            fusion_input_dim = state_dim * 3

        # Fusion MLP
        self.fusion = nn.Sequential(
            nn.Linear(fusion_input_dim, state_dim * 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(state_dim * 2, state_dim * 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(state_dim * 2, state_dim)
        )

        # Layer normalization for stability
        self.layer_norm = nn.LayerNorm(state_dim)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights with small values for stability."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.1)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(
        self,
        obs_features: torch.Tensor,
        lang_emb: torch.Tensor,
        prev_action: torch.Tensor,
        prev_state: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass to compute current state.

        Args:
            obs_features: [B, obs_tok_len, latent_dim] visual features
            lang_emb: [B, lang_emb_dim] or [B, 1, lang_emb_dim] language embedding
            prev_action: [B, action_dim] previous action
            prev_state: [B, state_dim] previous state (optional)

        Returns:
            z_t: [B, state_dim] current state latent
        """
        batch_size = obs_features.shape[0]

        # Flatten observation features: [B, obs_tok_len, latent_dim] -> [B, obs_tok_len * latent_dim]
        obs_flat = obs_features.view(batch_size, -1)
        obs_emb = self.obs_proj(obs_flat)  # [B, state_dim]

        # Handle language embedding shape
        if lang_emb.dim() == 3:
            lang_emb = lang_emb.squeeze(1)  # [B, 1, lang_emb_dim] -> [B, lang_emb_dim]
        lang_emb_proj = self.lang_proj(lang_emb)  # [B, state_dim]

        # Project previous action
        action_emb = self.action_proj(prev_action)  # [B, state_dim]

        # Concatenate all embeddings
        if self.use_prev_state and prev_state is not None:
            prev_state_emb = self.prev_state_proj(prev_state)  # [B, state_dim]
            fused = torch.cat([obs_emb, lang_emb_proj, action_emb, prev_state_emb], dim=-1)
        else:
            # Use zeros for prev_state if not provided
            if self.use_prev_state:
                prev_state_emb = torch.zeros(batch_size, self.state_dim, device=obs_features.device)
                prev_state_emb = self.prev_state_proj(prev_state_emb)
                fused = torch.cat([obs_emb, lang_emb_proj, action_emb, prev_state_emb], dim=-1)
            else:
                fused = torch.cat([obs_emb, lang_emb_proj, action_emb], dim=-1)

        # Fusion and normalization
        z_t = self.fusion(fused)  # [B, state_dim]
        z_t = self.layer_norm(z_t)

        return z_t


class CrossAttentionStateEncoder(nn.Module):
    """
    Alternative State Encoder using Cross-Attention for fusion.

    This version uses attention mechanism to fuse different modalities,
    allowing the model to learn which inputs are most relevant.
    """

    def __init__(
        self,
        latent_dim: int = 256,
        lang_emb_dim: int = 512,
        action_dim: int = 7,
        state_dim: int = 256,
        obs_tok_len: int = 2,
        num_heads: int = 4,
        dropout: float = 0.1
    ):
        super().__init__()

        self.state_dim = state_dim
        self.obs_tok_len = obs_tok_len

        # Projection layers
        self.obs_proj = nn.Linear(latent_dim, state_dim)
        self.lang_proj = nn.Linear(lang_emb_dim, state_dim)
        self.action_proj = nn.Linear(action_dim, state_dim)
        self.prev_state_proj = nn.Linear(state_dim, state_dim)

        # Learnable query for state
        self.state_query = nn.Parameter(torch.randn(1, 1, state_dim))

        # Cross-attention
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=state_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        # Output MLP
        self.output_mlp = nn.Sequential(
            nn.Linear(state_dim, state_dim * 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(state_dim * 2, state_dim)
        )

        self.layer_norm = nn.LayerNorm(state_dim)

    def forward(
        self,
        obs_features: torch.Tensor,
        lang_emb: torch.Tensor,
        prev_action: torch.Tensor,
        prev_state: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass with cross-attention fusion.

        Args:
            obs_features: [B, obs_tok_len, latent_dim]
            lang_emb: [B, lang_emb_dim] or [B, 1, lang_emb_dim]
            prev_action: [B, action_dim]
            prev_state: [B, state_dim] (optional)

        Returns:
            z_t: [B, state_dim]
        """
        batch_size = obs_features.shape[0]

        # Project observations: [B, obs_tok_len, state_dim]
        obs_emb = self.obs_proj(obs_features)

        # Project language: [B, 1, state_dim]
        if lang_emb.dim() == 2:
            lang_emb = lang_emb.unsqueeze(1)
        lang_emb_proj = self.lang_proj(lang_emb)

        # Project action: [B, 1, state_dim]
        action_emb = self.action_proj(prev_action).unsqueeze(1)

        # Project prev_state: [B, 1, state_dim]
        if prev_state is not None:
            prev_state_emb = self.prev_state_proj(prev_state).unsqueeze(1)
        else:
            prev_state_emb = torch.zeros(batch_size, 1, self.state_dim, device=obs_features.device)

        # Concatenate all as key-value pairs: [B, obs_tok_len + 3, state_dim]
        kv = torch.cat([obs_emb, lang_emb_proj, action_emb, prev_state_emb], dim=1)

        # Expand query: [B, 1, state_dim]
        query = self.state_query.expand(batch_size, -1, -1)

        # Cross-attention: query attends to all modalities
        attn_out, _ = self.cross_attn(query, kv, kv)

        # Output processing
        z_t = self.output_mlp(attn_out.squeeze(1))
        z_t = self.layer_norm(z_t)

        return z_t
