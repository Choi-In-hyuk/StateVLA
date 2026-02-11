"""
Mamba-based Context and Target Encoders for JEPA.

Context Encoder: Processes visible tokens and produces representations.
Target Encoder: EMA copy of Context Encoder, processes all tokens for targets.
"""

import copy
import torch
import torch.nn as nn
from typing import Tuple, Optional

from mamba import MixerModel


class ContextEncoder(nn.Module):
    """
    Context Encoder for JEPA.

    Processes visible (non-masked) tokens using Mamba backbone.
    Outputs both token-level features and a CLS representation (z_t).
    """

    def __init__(
        self,
        embed_dim: int = 256,
        depth: int = 12,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        rms_norm: bool = True,
        fused_add_norm: bool = False,
        residual_in_fp32: bool = False,
        device: str = 'cuda',
        dtype: torch.dtype = None,
    ):
        """
        Args:
            embed_dim: Token embedding dimension
            depth: Number of Mamba blocks
            d_state: Mamba state dimension
            d_conv: Mamba convolution dimension
            expand: Mamba expansion factor
            rms_norm: Use RMSNorm instead of LayerNorm
            fused_add_norm: Use fused operations
            residual_in_fp32: Keep residual in fp32
        """
        super().__init__()
        self.embed_dim = embed_dim

        # SSM configuration
        ssm_cfg = {
            'd_state': d_state,
            'd_conv': d_conv,
            'expand': expand,
            'layer': 'Mamba1',
        }

        # Mamba backbone
        self.backbone = MixerModel(
            d_model=embed_dim,
            n_layer=depth,
            d_intermediate=0,  # No MLP, pure Mamba
            ssm_cfg=ssm_cfg,
            rms_norm=rms_norm,
            fused_add_norm=fused_add_norm,
            residual_in_fp32=residual_in_fp32,
            device=device,
            dtype=dtype,
        )

        # Layer norm for output
        self.norm = nn.LayerNorm(embed_dim)

    def forward(
        self,
        tokens: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through context encoder.

        Args:
            tokens: [B, N, D] all tokens (CLS at last position for Mamba causal)
            mask: [B, N] True = masked tokens to exclude from processing
                  If None, process all tokens (inference mode)

        Returns:
            features: [B, N_visible-1, D] features for visible tokens (excl CLS)
            cls_output: [B, D] CLS token output (z_t)
        """
        B, N, D = tokens.shape

        if mask is not None:
            # Extract only visible tokens
            # CLS token (last position) is always visible
            visible_mask = ~mask  # True = visible

            max_visible = visible_mask.sum(dim=1).max().item()

            visible_tokens_list = []
            visible_indices_list = []

            for b in range(B):
                vis_idx = torch.where(visible_mask[b])[0]
                vis_tokens = tokens[b, vis_idx]  # [n_visible, D]

                # Pad if needed
                n_vis = vis_tokens.shape[0]
                if n_vis < max_visible:
                    pad = torch.zeros(max_visible - n_vis, D, device=tokens.device, dtype=tokens.dtype)
                    vis_tokens = torch.cat([vis_tokens, pad], dim=0)

                visible_tokens_list.append(vis_tokens)
                visible_indices_list.append(vis_idx)

            visible_tokens = torch.stack(visible_tokens_list, dim=0)  # [B, max_visible, D]
        else:
            # Inference mode: process all tokens
            visible_tokens = tokens
            max_visible = N

        # Process through Mamba backbone
        hidden = self.backbone(visible_tokens)  # [B, max_visible, D]

        # Apply final norm
        hidden = self.norm(hidden)

        # CLS output is the LAST token (Mamba causal: last sees all previous)
        cls_output = hidden[:, -1]  # [B, D]

        # Features are all tokens except CLS (last)
        features = hidden[:, :-1]  # [B, max_visible-1, D]

        return features, cls_output

    def forward_full(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Process all tokens without masking.
        Used for target encoder compatibility.

        Args:
            tokens: [B, N, D] all tokens

        Returns:
            [B, N, D] all token features
        """
        hidden = self.backbone(tokens)
        hidden = self.norm(hidden)
        return hidden


class TargetEncoder(nn.Module):
    """
    Target Encoder for JEPA (EMA copy of Context Encoder).

    Provides target representations for masked tokens.
    Updated via exponential moving average from Context Encoder.
    """

    def __init__(self, context_encoder: ContextEncoder):
        """
        Args:
            context_encoder: Context encoder to copy from
        """
        super().__init__()

        # Deep copy of context encoder
        self.encoder = copy.deepcopy(context_encoder)

        # Freeze all parameters
        for param in self.encoder.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Process all tokens (no masking).

        Args:
            tokens: [B, N, D] all tokens

        Returns:
            [B, N, D] all token features
        """
        return self.encoder.forward_full(tokens)

    @torch.no_grad()
    def update_ema(self, context_encoder: ContextEncoder, momentum: float = 0.996):
        """
        Update target encoder via EMA.

        target = momentum * target + (1 - momentum) * context

        Args:
            context_encoder: Source context encoder
            momentum: EMA momentum (higher = slower update)
        """
        for target_param, context_param in zip(
            self.encoder.parameters(),
            context_encoder.parameters()
        ):
            target_param.data.mul_(momentum).add_(context_param.data, alpha=1 - momentum)

        # Also update buffers (e.g., running stats in norms)
        for target_buf, context_buf in zip(
            self.encoder.buffers(),
            context_encoder.buffers()
        ):
            target_buf.data.copy_(context_buf.data)


def cosine_momentum_schedule(
    current_step: int,
    total_steps: int,
    base_momentum: float = 0.996,
    max_momentum: float = 1.0,
) -> float:
    """
    Cosine annealing schedule for EMA momentum.

    Momentum increases from base_momentum to max_momentum over training.

    Args:
        current_step: Current training step
        total_steps: Total training steps
        base_momentum: Starting momentum
        max_momentum: Ending momentum

    Returns:
        Current momentum value
    """
    import math
    progress = current_step / total_steps
    momentum = max_momentum - (max_momentum - base_momentum) * (
        1 + math.cos(math.pi * progress)
    ) / 2
    return momentum
