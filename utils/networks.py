import torch
import torch.nn as nn
import math
from typing import Optional


class MLP(nn.Module):
    """Multi-Layer Perceptron with configurable layers and activation."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: Optional[int] = None,
        num_layers: int = 2,
        activation: nn.Module = nn.SiLU(),
        dropout: float = 0.0,
        output_activation: bool = False
    ):
        super().__init__()

        if hidden_dim is None:
            hidden_dim = input_dim

        layers = []

        if num_layers == 1:
            layers.append(nn.Linear(input_dim, output_dim))
        else:
            # Input layer
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(activation)
            if dropout > 0:
                layers.append(nn.Dropout(dropout))

            # Hidden layers
            for _ in range(num_layers - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                layers.append(activation)
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))

            # Output layer
            layers.append(nn.Linear(hidden_dim, output_dim))

        if output_activation:
            layers.append(activation)

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TimeEmbedding(nn.Module):
    """
    Sinusoidal time embedding for diffusion models.
    Maps scalar timesteps to high-dimensional embeddings.
    """

    def __init__(
        self,
        embed_dim: int,
        freq_dim: int = 256,
        max_period: float = 10000.0
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.freq_dim = freq_dim
        self.max_period = max_period

        # MLP to process sinusoidal features
        self.mlp = nn.Sequential(
            nn.Linear(freq_dim, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim)
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: [B] tensor of timesteps (typically in [0, 1])

        Returns:
            [B, 1, embed_dim] time embeddings
        """
        # Ensure t is 1D
        if t.dim() == 0:
            t = t.unsqueeze(0)

        # Create sinusoidal frequencies
        half_dim = self.freq_dim // 2
        freqs = torch.exp(
            -math.log(self.max_period) * torch.arange(half_dim, device=t.device) / half_dim
        )

        # [B, half_dim]
        args = t.unsqueeze(-1) * freqs.unsqueeze(0)

        # [B, freq_dim]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)

        # [B, embed_dim]
        embedding = self.mlp(embedding)

        # [B, 1, embed_dim] for sequence concatenation
        return embedding.unsqueeze(1)


class GatedMLP(nn.Module):
    """Gated MLP used in Mamba blocks."""

    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        activation: nn.Module = nn.SiLU(),
        bias: bool = True,
        dropout: float = 0.0
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias)
        self.activation = activation
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class AdaLNZero(nn.Module):
    """
    Adaptive Layer Normalization with zero initialization.
    Used for conditioning in diffusion models.
    """

    def __init__(self, hidden_size: int):
        super().__init__()
        self.modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )
        # Zero initialize the output
        nn.init.zeros_(self.modulation[-1].weight)
        nn.init.zeros_(self.modulation[-1].bias)

    def forward(self, c: torch.Tensor):
        """
        Args:
            c: [B, hidden_size] conditioning signal

        Returns:
            Tuple of 6 modulation parameters, each [B, hidden_size]
        """
        return self.modulation(c).chunk(6, dim=-1)


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Apply adaptive layer normalization modulation."""
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
