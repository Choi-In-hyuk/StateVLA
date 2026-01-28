import os
import sys
_project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from typing import List

import torch
import torch.nn as nn

from utils.networks.clip import build_model, load_clip, tokenize


class LangClip(nn.Module):
    """CLIP-based language encoder for text instructions."""

    def __init__(self, freeze_backbone: bool = True, model_name: str = "ViT-B/32"):
        super(LangClip, self).__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._load_clip(model_name)
        if freeze_backbone:
            for param in self.clip_rn50.parameters():
                param.requires_grad = False

    def _load_clip(self, model_name: str) -> None:
        model, _ = load_clip(model_name, device=self.device)
        self.clip_rn50 = build_model(model.state_dict()).to(self.device)

    def forward(self, x: List) -> torch.Tensor:
        """
        Encode text instructions.

        Args:
            x: List of text strings

        Returns:
            [B, 1, 512] text embeddings
        """
        tokens = tokenize(x).to(self.device)
        emb = self.clip_rn50.encode_text(tokens)
        return torch.unsqueeze(emb, 1)

    @property
    def embed_dim(self) -> int:
        """Return the embedding dimension (512 for ViT-B/32)."""
        return 512
