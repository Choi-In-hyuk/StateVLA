"""
Eagle2-VLM Vision Encoder for StateVLA.

Uses SigLIP vision encoder from Eagle2 models.
"""

import torch
import torch.nn as nn
from typing import List, Dict, Optional
import einops


class Eagle2VisionEncoder(nn.Module):
    """
    Vision encoder using Eagle2's SigLIP backbone.

    Eagle2 uses SigLIP (Sigmoid Loss for Language Image Pre-training) as its
    vision encoder, which provides strong visual representations.
    """

    def __init__(
        self,
        model_name: str = "nvidia/Eagle2-1B",
        latent_dim: int = 256,
        freeze_backbone: bool = True,
        device: str = 'cuda'
    ):
        """
        Args:
            model_name: Eagle2 model name from HuggingFace
            latent_dim: Output dimension for state encoding
            freeze_backbone: Whether to freeze vision encoder weights
            device: Device to load model on
        """
        super().__init__()
        self.model_name = model_name
        self.latent_dim = latent_dim
        self.device = device

        self._load_vision_encoder(model_name, freeze_backbone)

        # Projection layer to match latent_dim
        self.projection = nn.Linear(self.hidden_size, latent_dim)

    def _load_vision_encoder(self, model_name: str, freeze_backbone: bool):
        """Load vision encoder from Eagle2 model."""
        from transformers import AutoModel, AutoProcessor, AutoConfig
        import os

        # Force disable flash attention
        os.environ['TRANSFORMERS_NO_FLASH_ATTN'] = '1'

        print(f"Loading Eagle2 vision encoder from {model_name}...")

        # Load config and force eager attention
        config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        config._attn_implementation = 'eager'

        # Also set for vision_config if it exists
        if hasattr(config, 'vision_config'):
            config.vision_config._attn_implementation = 'eager'
            if hasattr(config.vision_config, 'attn_implementation'):
                config.vision_config.attn_implementation = 'eager'

        # Load full model to extract vision tower
        full_model = AutoModel.from_pretrained(
            model_name,
            config=config,
            trust_remote_code=True,
            torch_dtype=torch.float32,
            device_map=None,
            attn_implementation="eager"  # Avoid flash_attn requirement
        )

        # Extract vision tower (SigLIP encoder)
        if hasattr(full_model, 'vision_tower'):
            self.vision_encoder = full_model.vision_tower
        elif hasattr(full_model, 'vision_model'):
            self.vision_encoder = full_model.vision_model
        elif hasattr(full_model, 'model') and hasattr(full_model.model, 'vision_tower'):
            self.vision_encoder = full_model.model.vision_tower
        else:
            # Try to find vision component
            for name, module in full_model.named_modules():
                if 'vision' in name.lower() and hasattr(module, 'forward'):
                    self.vision_encoder = module
                    break
            else:
                raise ValueError(f"Could not find vision encoder in {model_name}")

        # Get hidden size
        if hasattr(self.vision_encoder, 'config'):
            self.hidden_size = getattr(self.vision_encoder.config, 'hidden_size', 1152)
        else:
            self.hidden_size = 1152  # Default SigLIP hidden size

        # Load processor for image preprocessing
        self.processor = AutoProcessor.from_pretrained(
            model_name,
            trust_remote_code=True
        )

        if freeze_backbone:
            for param in self.vision_encoder.parameters():
                param.requires_grad = False
            print("Vision encoder weights frozen")

        # Clean up full model to save memory
        del full_model
        torch.cuda.empty_cache()

        print(f"Vision encoder loaded. Hidden size: {self.hidden_size}")

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Encode images.

        Args:
            images: [B, C, H, W] or [B, T, C, H, W] image tensor

        Returns:
            [B, latent_dim] or [B, T, latent_dim] encoded features
        """
        batch_size = images.shape[0]
        time_series = False
        t_steps = 1

        if len(images.shape) == 5:
            t_steps = images.shape[1]
            images = einops.rearrange(images, 'b t c h w -> (b t) c h w')
            time_series = True

        # Forward through vision encoder
        outputs = self.vision_encoder(images)

        # Handle different output formats
        if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
            features = outputs.pooler_output
        elif hasattr(outputs, 'last_hidden_state'):
            # Use CLS token or mean pooling
            features = outputs.last_hidden_state[:, 0]  # CLS token
        elif isinstance(outputs, torch.Tensor):
            if len(outputs.shape) == 3:
                features = outputs[:, 0]  # CLS token
            else:
                features = outputs
        else:
            features = outputs

        # Project to latent_dim
        features = self.projection(features)

        if time_series:
            features = einops.rearrange(
                features, '(b t) d -> b t d',
                b=batch_size, t=t_steps
            )

        return features


class MultiImageEagle2Encoder(nn.Module):
    """
    Multi-camera encoder using Eagle2 vision backbone.

    Creates separate projection heads for each camera while sharing
    the vision encoder backbone.
    """

    def __init__(
        self,
        camera_names: List[str],
        model_name: str = "nvidia/Eagle2-1B",
        latent_dim: int = 256,
        freeze_backbone: bool = True,
        share_backbone: bool = True,
        device: str = 'cuda'
    ):
        """
        Args:
            camera_names: List of camera names
            model_name: Eagle2 model name
            latent_dim: Output dimension
            freeze_backbone: Whether to freeze backbone
            share_backbone: Whether to share backbone across cameras
            device: Device
        """
        super().__init__()

        self.camera_names = camera_names
        self.latent_dim = latent_dim
        self.share_backbone = share_backbone

        if share_backbone:
            # Single shared backbone with per-camera projection heads
            self._load_shared_backbone(model_name, freeze_backbone, device)

            # Per-camera projection heads
            self.projections = nn.ModuleDict({
                f"{cam}_image": nn.Linear(self.hidden_size, latent_dim)
                for cam in camera_names
            })
        else:
            # Separate encoder for each camera
            self.encoders = nn.ModuleDict({
                f"{cam}_image": Eagle2VisionEncoder(
                    model_name=model_name,
                    latent_dim=latent_dim,
                    freeze_backbone=freeze_backbone,
                    device=device
                )
                for cam in camera_names
            })

        self._dummy_variable = nn.Parameter(torch.zeros(0))

    def _load_shared_backbone(self, model_name: str, freeze_backbone: bool, device: str):
        """Load shared vision backbone."""
        from transformers import AutoModel, AutoProcessor, AutoConfig
        import os

        # Force disable flash attention
        os.environ['TRANSFORMERS_NO_FLASH_ATTN'] = '1'

        print(f"Loading shared Eagle2 backbone from {model_name}...")

        # Load config and force eager attention everywhere
        config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        config._attn_implementation = 'eager'

        # Set for all sub-configs
        if hasattr(config, 'vision_config') and config.vision_config is not None:
            config.vision_config._attn_implementation = 'eager'
            if hasattr(config.vision_config, 'attn_implementation'):
                config.vision_config.attn_implementation = 'eager'
        if hasattr(config, 'text_config') and config.text_config is not None:
            config.text_config._attn_implementation = 'eager'
            if hasattr(config.text_config, 'attn_implementation'):
                config.text_config.attn_implementation = 'eager'

        full_model = AutoModel.from_pretrained(
            model_name,
            config=config,
            trust_remote_code=True,
            torch_dtype=torch.float32,
            device_map=None,
            attn_implementation="eager"  # Avoid flash_attn requirement
        )

        # Force eager attention on the loaded vision encoder components
        if hasattr(full_model, 'config'):
            full_model.config._attn_implementation = 'eager'

        # Extract vision tower
        if hasattr(full_model, 'vision_tower'):
            self.vision_encoder = full_model.vision_tower
        elif hasattr(full_model, 'vision_model'):
            self.vision_encoder = full_model.vision_model
        elif hasattr(full_model, 'model') and hasattr(full_model.model, 'vision_tower'):
            self.vision_encoder = full_model.model.vision_tower
        else:
            for name, module in full_model.named_modules():
                if 'vision' in name.lower():
                    self.vision_encoder = module
                    break
            else:
                raise ValueError(f"Could not find vision encoder in {model_name}")

        # Force eager attention on vision encoder
        if hasattr(self.vision_encoder, 'config'):
            self.vision_encoder.config._attn_implementation = 'eager'
            if hasattr(self.vision_encoder.config, 'attn_implementation'):
                self.vision_encoder.config.attn_implementation = 'eager'

        # Get hidden size
        if hasattr(self.vision_encoder, 'config'):
            self.hidden_size = getattr(self.vision_encoder.config, 'hidden_size', 1152)
        else:
            self.hidden_size = 1152

        if freeze_backbone:
            for param in self.vision_encoder.parameters():
                param.requires_grad = False

        del full_model
        torch.cuda.empty_cache()

        print(f"Shared backbone loaded. Hidden size: {self.hidden_size}")

    def forward(self, obs_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Encode multi-camera observations.

        Args:
            obs_dict: Dictionary with camera images
                     Keys: "{camera_name}_image"

        Returns:
            [B, num_cameras, latent_dim] encoded features
        """
        features = []

        for camera_name in self.camera_names:
            image_key = f"{camera_name}_image"
            if image_key not in obs_dict:
                raise KeyError(f"Expected key '{image_key}' not found in obs_dict")

            image = obs_dict[image_key]

            if self.share_backbone:
                # Shared backbone encoding with position interpolation for flexible image sizes
                outputs = self.vision_encoder(image, interpolate_pos_encoding=True)

                if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
                    feat = outputs.pooler_output
                elif hasattr(outputs, 'last_hidden_state'):
                    feat = outputs.last_hidden_state[:, 0]
                elif isinstance(outputs, torch.Tensor):
                    feat = outputs[:, 0] if len(outputs.shape) == 3 else outputs
                else:
                    feat = outputs

                feat = self.projections[image_key](feat)
            else:
                # Separate encoder
                feat = self.encoders[image_key](image)

            features.append(feat)

        # Stack: [B, num_cameras, latent_dim]
        return torch.stack(features, dim=1)


if __name__ == "__main__":
    # Test the encoder
    print("Testing Eagle2 Vision Encoder...")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Test single encoder
    encoder = Eagle2VisionEncoder(
        model_name="nvidia/Eagle2-1B",
        latent_dim=256,
        freeze_backbone=True,
        device=device
    ).to(device)

    # Test input
    dummy_image = torch.randn(2, 3, 384, 384).to(device)
    output = encoder(dummy_image)
    print(f"Single encoder output shape: {output.shape}")

    # Test multi-camera encoder
    camera_names = ["agentview", "eye_in_hand"]
    multi_encoder = MultiImageEagle2Encoder(
        camera_names=camera_names,
        model_name="nvidia/Eagle2-1B",
        latent_dim=256,
        freeze_backbone=True,
        share_backbone=True,
        device=device
    ).to(device)

    obs_dict = {
        "agentview_image": torch.randn(2, 3, 384, 384).to(device),
        "eye_in_hand_image": torch.randn(2, 3, 384, 384).to(device)
    }

    output = multi_encoder(obs_dict)
    print(f"Multi encoder output shape: {output.shape}")
