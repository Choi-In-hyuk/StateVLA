"""
Multimodal Tokenizers for JEPA StateVLA.

Converts different modalities into a unified token sequence:
- Images -> ViT patch tokens (simple or pretrained SigLIP)
- Language -> embedding token (simple or pretrained CLIP)
- Robot state -> embedding token
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
from einops import rearrange


class ImageTokenizer(nn.Module):
    """
    Simple image tokenizer using Conv2d patch embedding.
    For from-scratch training.
    """

    def __init__(
        self,
        image_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        embed_dim: int = 256,
    ):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2
        self.embed_dim = embed_dim

        self.patch_embed = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches, embed_dim)
        )

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        w = self.patch_embed.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.zeros_(self.patch_embed.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """[B, C, H, W] -> [B, num_patches, embed_dim]"""
        x = self.patch_embed(x)
        x = rearrange(x, 'b d h w -> b (h w) d')
        x = x + self.pos_embed
        return x


class PretrainedImageTokenizer(nn.Module):
    """
    Image tokenizer using pretrained SigLIP vision encoder.
    Extracts patch-level features for JEPA masking.
    """

    def __init__(
        self,
        model_name: str = "google/siglip-base-patch16-224",
        embed_dim: int = 256,
        freeze_backbone: bool = True,
        device: str = 'cuda',
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.device = device
        self.freeze_backbone = freeze_backbone

        self._load_encoder(model_name, freeze_backbone)

        # Project from SigLIP hidden size to embed_dim
        self.proj = nn.Linear(self.hidden_size, embed_dim)

        # Learnable position embedding (will be added after projection)
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches, embed_dim)
        )
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def _load_encoder(self, model_name: str, freeze_backbone: bool):
        """Load SigLIP vision encoder."""
        from transformers import AutoModel, AutoConfig

        print(f"Loading SigLIP vision encoder from {model_name}...")

        config = AutoConfig.from_pretrained(model_name)

        # Get hidden size and num patches from config
        if hasattr(config, 'vision_config'):
            vision_config = config.vision_config
            self.hidden_size = vision_config.hidden_size
            image_size = vision_config.image_size
            patch_size = vision_config.patch_size
        else:
            self.hidden_size = getattr(config, 'hidden_size', 768)
            image_size = getattr(config, 'image_size', 224)
            patch_size = getattr(config, 'patch_size', 16)

        self.num_patches = (image_size // patch_size) ** 2
        self.image_size = image_size
        self.patch_size = patch_size

        # Load model
        model = AutoModel.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
        )

        # Extract vision encoder
        if hasattr(model, 'vision_model'):
            self.encoder = model.vision_model
        elif hasattr(model, 'vision_tower'):
            self.encoder = model.vision_tower
        else:
            self.encoder = model

        if freeze_backbone:
            for param in self.encoder.parameters():
                param.requires_grad = False
            print("SigLIP backbone frozen")

        print(f"SigLIP loaded. Hidden size: {self.hidden_size}, Patches: {self.num_patches}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract patch-level features from SigLIP.

        Args:
            x: [B, C, H, W] image tensor

        Returns:
            [B, num_patches, embed_dim] patch tokens
        """
        # Resize if needed
        if x.shape[-1] != self.image_size:
            x = F.interpolate(x, size=(self.image_size, self.image_size), mode='bilinear', align_corners=False)

        # Get encoder output
        if self.freeze_backbone:
            with torch.no_grad():
                outputs = self.encoder(x)
        else:
            outputs = self.encoder(x)

        # Extract patch features (exclude CLS token if present)
        if hasattr(outputs, 'last_hidden_state'):
            features = outputs.last_hidden_state
        else:
            features = outputs

        # SigLIP typically has [B, num_patches, hidden_size] without CLS
        # But some models have CLS at position 0
        if features.shape[1] == self.num_patches + 1:
            features = features[:, 1:]  # Remove CLS token

        # Project to embed_dim
        features = self.proj(features)

        # Add position embedding
        features = features + self.pos_embed

        return features


class LanguageTokenizer(nn.Module):
    """
    Simple language tokenizer.
    Expects pre-computed embeddings.
    """

    def __init__(
        self,
        lang_emb_dim: int = 512,
        embed_dim: int = 256,
        num_tokens: int = 1,
    ):
        super().__init__()
        self.lang_emb_dim = lang_emb_dim
        self.embed_dim = embed_dim
        self.num_tokens = num_tokens

        self.proj = nn.Linear(lang_emb_dim, embed_dim * num_tokens)
        self.pos_embed = nn.Parameter(torch.zeros(1, num_tokens, embed_dim))

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.xavier_uniform_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, lang_emb: torch.Tensor) -> torch.Tensor:
        """[B, lang_emb_dim] -> [B, num_tokens, embed_dim]"""
        B = lang_emb.shape[0]
        x = self.proj(lang_emb)
        x = x.view(B, self.num_tokens, self.embed_dim)
        x = x + self.pos_embed
        return x


class PretrainedLanguageTokenizer(nn.Module):
    """
    Language tokenizer using pretrained CLIP text encoder.
    Encodes text directly from strings.
    """

    def __init__(
        self,
        model_name: str = "ViT-B/32",
        embed_dim: int = 256,
        freeze_backbone: bool = True,
        device: str = 'cuda',
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.device = device
        self.num_tokens = 1

        self._load_encoder(model_name, freeze_backbone)

        # Project from CLIP hidden size to embed_dim
        self.proj = nn.Linear(self.hidden_size, embed_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def _load_encoder(self, model_name: str, freeze_backbone: bool):
        """Load CLIP text encoder."""
        import sys
        import os
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from utils.networks.clip import build_model, load_clip, tokenize

        print(f"Loading CLIP text encoder: {model_name}...")

        model, _ = load_clip(model_name, device=self.device)
        self.clip_model = build_model(model.state_dict())
        self.tokenize = tokenize

        self.hidden_size = 512  # CLIP ViT-B/32 text embedding dim

        if freeze_backbone:
            for param in self.clip_model.parameters():
                param.requires_grad = False
            print("CLIP text encoder frozen")

    def forward(self, text_or_emb: torch.Tensor) -> torch.Tensor:
        """
        Encode text or use pre-computed embeddings.

        Args:
            text_or_emb: [B, hidden_size] pre-computed CLIP embeddings

        Returns:
            [B, 1, embed_dim] language tokens
        """
        B = text_or_emb.shape[0]

        # Assume pre-computed embeddings
        x = self.proj(text_or_emb)
        x = x.view(B, 1, self.embed_dim)
        x = x + self.pos_embed

        return x

    def encode_text(self, texts: List[str]) -> torch.Tensor:
        """
        Encode text strings directly.

        Args:
            texts: List of text strings

        Returns:
            [B, hidden_size] CLIP text embeddings
        """
        tokens = self.tokenize(texts).to(self.device)
        with torch.no_grad():
            emb = self.clip_model.encode_text(tokens)
        return emb


class RobotStateTokenizer(nn.Module):
    """Converts robot state to tokens."""

    def __init__(
        self,
        robot_state_dim: int = 8,
        embed_dim: int = 256,
        num_tokens: int = 1,
    ):
        super().__init__()
        self.robot_state_dim = robot_state_dim
        self.embed_dim = embed_dim
        self.num_tokens = num_tokens

        self.proj = nn.Sequential(
            nn.Linear(robot_state_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim * num_tokens),
        )

        self.pos_embed = nn.Parameter(torch.zeros(1, num_tokens, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, robot_state: torch.Tensor) -> torch.Tensor:
        """[B, robot_state_dim] -> [B, num_tokens, embed_dim]"""
        B = robot_state.shape[0]
        x = self.proj(robot_state)
        x = x.view(B, self.num_tokens, self.embed_dim)
        x = x + self.pos_embed
        return x


class MultiModalTokenizer(nn.Module):
    """
    Unified multimodal tokenizer.

    Supports both:
    - Simple (from-scratch) tokenizers
    - Pretrained (SigLIP + CLIP) tokenizers
    """

    MODALITY_AGENTVIEW = 0
    MODALITY_WRIST = 1
    MODALITY_LANGUAGE = 2
    MODALITY_ROBOT_STATE = 3

    def __init__(
        self,
        camera_names: List[str] = ['agentview', 'eye_in_hand'],
        image_size: int = 224,
        patch_size: int = 16,
        embed_dim: int = 256,
        lang_emb_dim: int = 512,
        robot_state_dim: int = 8,
        # Pretrained encoder options
        use_pretrained_vision: bool = False,
        use_pretrained_language: bool = False,
        vision_model_name: str = "google/siglip-base-patch16-224",
        language_model_name: str = "ViT-B/32",
        freeze_vision: bool = True,
        freeze_language: bool = True,
        device: str = 'cuda',
    ):
        super().__init__()
        self.camera_names = camera_names
        self.embed_dim = embed_dim
        self.use_pretrained_vision = use_pretrained_vision
        self.use_pretrained_language = use_pretrained_language
        self.device = device

        # Image tokenizers
        if use_pretrained_vision:
            # Shared pretrained backbone with per-camera projection
            self._init_pretrained_vision(
                camera_names, vision_model_name, embed_dim, freeze_vision, device
            )
        else:
            self.num_patches_per_image = (image_size // patch_size) ** 2
            self.image_tokenizers = nn.ModuleDict({
                name: ImageTokenizer(
                    image_size=image_size,
                    patch_size=patch_size,
                    embed_dim=embed_dim,
                )
                for name in camera_names
            })

        # Language tokenizer
        if use_pretrained_language:
            self.language_tokenizer = PretrainedLanguageTokenizer(
                model_name=language_model_name,
                embed_dim=embed_dim,
                freeze_backbone=freeze_language,
                device=device,
            )
        else:
            self.language_tokenizer = LanguageTokenizer(
                lang_emb_dim=lang_emb_dim,
                embed_dim=embed_dim,
                num_tokens=1,
            )

        # Robot state tokenizer (always simple)
        self.robot_state_tokenizer = RobotStateTokenizer(
            robot_state_dim=robot_state_dim,
            embed_dim=embed_dim,
            num_tokens=1,
        )

        # CLS token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # Modality embeddings
        self.modality_embed = nn.Embedding(4, embed_dim)

        # Calculate total tokens
        self.num_image_tokens = len(camera_names) * self.num_patches_per_image
        self.num_lang_tokens = 1
        self.num_robot_tokens = 1
        self.total_tokens = 1 + self.num_image_tokens + self.num_lang_tokens + self.num_robot_tokens

        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def _init_pretrained_vision(
        self,
        camera_names: List[str],
        model_name: str,
        embed_dim: int,
        freeze: bool,
        device: str,
    ):
        """Initialize shared pretrained vision encoder."""
        from transformers import AutoModel, AutoConfig

        print(f"Loading shared SigLIP encoder: {model_name}...")

        config = AutoConfig.from_pretrained(model_name)

        if hasattr(config, 'vision_config'):
            vision_config = config.vision_config
            self.hidden_size = vision_config.hidden_size
            image_size = vision_config.image_size
            patch_size = vision_config.patch_size
        else:
            self.hidden_size = getattr(config, 'hidden_size', 768)
            image_size = getattr(config, 'image_size', 224)
            patch_size = getattr(config, 'patch_size', 16)

        self.num_patches_per_image = (image_size // patch_size) ** 2
        self.vision_image_size = image_size

        # Load model
        model = AutoModel.from_pretrained(model_name, torch_dtype=torch.float32)

        if hasattr(model, 'vision_model'):
            self.vision_encoder = model.vision_model
        elif hasattr(model, 'vision_tower'):
            self.vision_encoder = model.vision_tower
        else:
            self.vision_encoder = model

        if freeze:
            for param in self.vision_encoder.parameters():
                param.requires_grad = False

        # Per-camera projection heads
        self.image_projections = nn.ModuleDict({
            name: nn.Linear(self.hidden_size, embed_dim)
            for name in camera_names
        })

        # Per-camera position embeddings
        self.image_pos_embeds = nn.ParameterDict({
            name: nn.Parameter(torch.zeros(1, self.num_patches_per_image, embed_dim))
            for name in camera_names
        })

        for name in camera_names:
            nn.init.trunc_normal_(self.image_pos_embeds[name], std=0.02)

        print(f"SigLIP loaded. Hidden: {self.hidden_size}, Patches: {self.num_patches_per_image}")

    def _encode_image_pretrained(self, img: torch.Tensor, camera_name: str) -> torch.Tensor:
        """Encode image using pretrained SigLIP."""
        # Resize if needed
        if img.shape[-1] != self.vision_image_size:
            img = F.interpolate(
                img,
                size=(self.vision_image_size, self.vision_image_size),
                mode='bilinear',
                align_corners=False
            )

        # Get features
        with torch.no_grad():
            outputs = self.vision_encoder(img)

        if hasattr(outputs, 'last_hidden_state'):
            features = outputs.last_hidden_state
        else:
            features = outputs

        # Remove CLS if present
        if features.shape[1] == self.num_patches_per_image + 1:
            features = features[:, 1:]

        # Project and add position embedding
        features = self.image_projections[camera_name](features)
        features = features + self.image_pos_embeds[camera_name]

        return features

    def forward(
        self,
        obs_dict: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Tokenize all modalities.

        Args:
            obs_dict: Dictionary containing:
                - 'agentview_image': [B, C, H, W]
                - 'eye_in_hand_image': [B, C, H, W]
                - 'lang_emb': [B, lang_emb_dim]
                - 'robot_states': [B, robot_state_dim]

        Returns:
            tokens: [B, total_tokens, embed_dim]
            modality_ids: [B, total_tokens]
        """
        B = obs_dict['lang_emb'].shape[0]
        device = obs_dict['lang_emb'].device

        all_tokens = []
        all_modality_ids = []

        # CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        all_tokens.append(cls_tokens)
        all_modality_ids.append(torch.full((B, 1), -1, device=device, dtype=torch.long))

        # Image tokens
        for i, name in enumerate(self.camera_names):
            if name == 'agentview':
                img_key = 'agentview_image'
                modality_id = self.MODALITY_AGENTVIEW
            elif name == 'eye_in_hand':
                img_key = 'eye_in_hand_image'
                modality_id = self.MODALITY_WRIST
            else:
                img_key = f'{name}_image'
                modality_id = i

            img = obs_dict[img_key]

            if self.use_pretrained_vision:
                img_tokens = self._encode_image_pretrained(img, name)
            else:
                img_tokens = self.image_tokenizers[name](img)

            # Add modality embedding
            img_tokens = img_tokens + self.modality_embed(
                torch.tensor(modality_id, device=device)
            )

            all_tokens.append(img_tokens)
            all_modality_ids.append(
                torch.full((B, self.num_patches_per_image), modality_id, device=device, dtype=torch.long)
            )

        # Language token
        lang_emb = obs_dict['lang_emb']
        lang_tokens = self.language_tokenizer(lang_emb)
        lang_tokens = lang_tokens + self.modality_embed(
            torch.tensor(self.MODALITY_LANGUAGE, device=device)
        )
        all_tokens.append(lang_tokens)
        all_modality_ids.append(
            torch.full((B, self.num_lang_tokens), self.MODALITY_LANGUAGE, device=device, dtype=torch.long)
        )

        # Robot state token
        robot_state = obs_dict['robot_states']
        robot_tokens = self.robot_state_tokenizer(robot_state)
        robot_tokens = robot_tokens + self.modality_embed(
            torch.tensor(self.MODALITY_ROBOT_STATE, device=device)
        )
        all_tokens.append(robot_tokens)
        all_modality_ids.append(
            torch.full((B, self.num_robot_tokens), self.MODALITY_ROBOT_STATE, device=device, dtype=torch.long)
        )

        tokens = torch.cat(all_tokens, dim=1)
        modality_ids = torch.cat(all_modality_ids, dim=1)

        return tokens, modality_ids

    def get_modality_ranges(self) -> Dict[str, Tuple[int, int]]:
        """Get token index ranges for each modality."""
        ranges = {}
        idx = 1  # Skip CLS

        for name in self.camera_names:
            ranges[name] = (idx, idx + self.num_patches_per_image)
            idx += self.num_patches_per_image

        ranges['language'] = (idx, idx + self.num_lang_tokens)
        idx += self.num_lang_tokens

        ranges['robot_state'] = (idx, idx + self.num_robot_tokens)

        return ranges
