"""
JEPA Predictor for predicting masked token representations.

Takes context features (from visible tokens) and predicts
representations for masked positions.
"""

import torch
import torch.nn as nn
from typing import Optional

from mamba import MixerModel


class JEPAPredictor(nn.Module):
    """
    Predictor head for JEPA.

    Takes context features from visible tokens and predicts
    the latent representations for masked tokens.

    Architecture:
    1. Project context features to predictor dimension
    2. Add learnable mask tokens at masked positions
    3. Process through lightweight Mamba
    4. Project back to encoder dimension
    """

    def __init__(
        self,
        embed_dim: int = 256,
        predictor_embed_dim: int = 192,
        depth: int = 6,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        rms_norm: bool = True,
        device: str = 'cuda',
        dtype: torch.dtype = None,
    ):
        """
        Args:
            embed_dim: Encoder output dimension
            predictor_embed_dim: Internal predictor dimension (usually smaller)
            depth: Number of Mamba blocks
            d_state: Mamba state dimension
            d_conv: Mamba convolution dimension
            expand: Mamba expansion factor
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.predictor_embed_dim = predictor_embed_dim

        # Project from encoder dim to predictor dim
        self.input_proj = nn.Linear(embed_dim, predictor_embed_dim)

        # Learnable mask token
        self.mask_token = nn.Parameter(torch.zeros(1, 1, predictor_embed_dim))

        # SSM configuration
        ssm_cfg = {
            'd_state': d_state,
            'd_conv': d_conv,
            'expand': expand,
            'layer': 'Mamba1',
        }

        # Lightweight Mamba backbone for prediction
        self.backbone = MixerModel(
            d_model=predictor_embed_dim,
            n_layer=depth,
            d_intermediate=0,
            ssm_cfg=ssm_cfg,
            rms_norm=rms_norm,
            fused_add_norm=False,
            residual_in_fp32=False,
            device=device,
            dtype=dtype,
        )

        # Project back to encoder dimension
        self.output_proj = nn.Linear(predictor_embed_dim, embed_dim)

        # Layer norm
        self.norm = nn.LayerNorm(predictor_embed_dim)

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        nn.init.xavier_uniform_(self.input_proj.weight)
        nn.init.zeros_(self.input_proj.bias)
        nn.init.xavier_uniform_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

    def forward(
        self,
        context_features: torch.Tensor,
        mask: torch.Tensor,
        all_positions: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Predict representations for masked tokens.

        Args:
            context_features: [B, N_visible-1, D] features from context encoder
                              (excludes CLS token)
            mask: [B, N] original mask (True = masked)
            all_positions: [B, N, D] optional position embeddings for all tokens

        Returns:
            predictions: [B, N_masked, D] predicted representations for masked tokens
        """
        B = context_features.shape[0]
        N = mask.shape[1]
        device = context_features.device

        # Project context to predictor dimension
        context_proj = self.input_proj(context_features)  # [B, N_visible-1, pred_dim]

        # Count masked tokens (excluding CLS at position 0)
        # mask[:, 0] should be False (CLS never masked)
        num_masked_per_sample = mask[:, 1:].sum(dim=1)  # [B]
        max_masked = num_masked_per_sample.max().item()

        if max_masked == 0:
            # No masked tokens, return empty
            return torch.zeros(B, 0, self.embed_dim, device=device)

        # Build full sequence with mask tokens at masked positions
        # We need to reconstruct the sequence order

        # For each sample, we need to:
        # 1. Start with visible context features
        # 2. Insert mask tokens at masked positions

        predictions_list = []

        for b in range(B):
            # Get mask for this sample (excluding CLS)
            sample_mask = mask[b, 1:]  # [N-1]

            # Visible indices (in context_features, which excludes CLS)
            visible_idx = torch.where(~sample_mask)[0]
            # Masked indices
            masked_idx = torch.where(sample_mask)[0]

            n_visible = len(visible_idx)
            n_masked = len(masked_idx)

            if n_masked == 0:
                # No masked tokens for this sample
                pred = torch.zeros(max_masked, self.embed_dim, device=device)
                predictions_list.append(pred)
                continue

            # Get context features for this sample
            ctx = context_proj[b, :n_visible]  # [n_visible, pred_dim]

            # Create mask tokens
            mask_tokens = self.mask_token.squeeze(0).expand(n_masked, -1)  # [n_masked, pred_dim]

            # Build full sequence (visible + masked in original order)
            full_seq = torch.zeros(N - 1, self.predictor_embed_dim, device=device)
            full_seq[visible_idx] = ctx
            full_seq[masked_idx] = mask_tokens

            predictions_list.append((full_seq, masked_idx, n_masked))

        # Batch process
        # Stack sequences (they all have same length N-1)
        full_seqs = torch.stack([p[0] if isinstance(p, tuple) else torch.zeros(N - 1, self.predictor_embed_dim, device=device) for p in predictions_list], dim=0)

        # Process through Mamba
        hidden = self.backbone(full_seqs)  # [B, N-1, pred_dim]
        hidden = self.norm(hidden)

        # Extract predictions for masked positions
        predictions = []
        for b in range(B):
            if isinstance(predictions_list[b], tuple):
                _, masked_idx, n_masked = predictions_list[b]
                pred = hidden[b, masked_idx]  # [n_masked, pred_dim]
                # Project to encoder dimension
                pred = self.output_proj(pred)  # [n_masked, embed_dim]

                # Pad to max_masked
                if n_masked < max_masked:
                    pad = torch.zeros(max_masked - n_masked, self.embed_dim, device=device)
                    pred = torch.cat([pred, pad], dim=0)
            else:
                pred = torch.zeros(max_masked, self.embed_dim, device=device)

            predictions.append(pred)

        predictions = torch.stack(predictions, dim=0)  # [B, max_masked, embed_dim]

        return predictions


class SimpleJEPAPredictor(nn.Module):
    """
    Simplified JEPA Predictor.

    Instead of reconstructing the full sequence, directly predicts
    masked representations from pooled context.
    """

    def __init__(
        self,
        embed_dim: int = 256,
        predictor_embed_dim: int = 192,
        depth: int = 4,
        max_masked_tokens: int = 200,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.predictor_embed_dim = predictor_embed_dim
        self.max_masked_tokens = max_masked_tokens

        # Project and pool context
        self.context_proj = nn.Linear(embed_dim, predictor_embed_dim)

        # Learnable query tokens for predictions
        self.query_tokens = nn.Parameter(
            torch.zeros(1, max_masked_tokens, predictor_embed_dim)
        )

        # Cross-attention-like MLP (simplified)
        self.predictor = nn.Sequential(
            nn.Linear(predictor_embed_dim * 2, predictor_embed_dim),
            nn.GELU(),
            nn.Linear(predictor_embed_dim, predictor_embed_dim),
            nn.GELU(),
            nn.Linear(predictor_embed_dim, embed_dim),
        )

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.query_tokens, std=0.02)

    def forward(
        self,
        context_features: torch.Tensor,
        mask: torch.Tensor,
        all_positions: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Predict masked representations.

        Args:
            context_features: [B, N_visible-1, D] from context encoder
            mask: [B, N] original mask

        Returns:
            [B, N_masked, D] predictions
        """
        B = context_features.shape[0]
        device = context_features.device

        # Pool context features
        context_pooled = context_features.mean(dim=1)  # [B, D]
        context_pooled = self.context_proj(context_pooled)  # [B, pred_dim]

        # Count masked tokens
        num_masked = mask[:, 1:].sum(dim=1)  # [B]
        max_masked = num_masked.max().item()

        if max_masked == 0:
            return torch.zeros(B, 0, self.embed_dim, device=device)

        # Use query tokens
        queries = self.query_tokens[:, :max_masked].expand(B, -1, -1)  # [B, max_masked, pred_dim]

        # Combine with context
        context_expanded = context_pooled.unsqueeze(1).expand(-1, max_masked, -1)
        combined = torch.cat([queries, context_expanded], dim=-1)  # [B, max_masked, pred_dim*2]

        # Predict
        predictions = self.predictor(combined)  # [B, max_masked, embed_dim]

        return predictions
