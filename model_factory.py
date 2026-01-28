"""
Model Factory: Factory functions for creating StateVLA models.

Provides convenient functions to create fully configured StateVLA models
with all encoders and components properly initialized.

Supports:
  - Vision Encoders: ResNet, Eagle2-VLM (SigLIP)
  - Language Encoders: CLIP, Qwen
"""

import os
import sys
_project_root = os.path.dirname(os.path.abspath(__file__))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import torch
import torch.nn as nn
from typing import List, Optional, Dict, Any

from statevla_model import StateVLA, StateVLAWithEncoders
from backbone.resnet import MultiImageResNetEncoder
from backbone.clip import LangClip


class ObservationEncoder(nn.Module):
    """
    Wrapper for image encoder that formats output for StateVLA.
    """

    def __init__(
        self,
        image_encoder: nn.Module,
        camera_names: List[str],
        latent_dim: int
    ):
        super().__init__()
        self.image_encoder = image_encoder
        self.camera_names = camera_names
        self.latent_dim = latent_dim

    def forward(self, obs_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Encode observations.

        Args:
            obs_dict: Dictionary with camera images

        Returns:
            [B, obs_tok_len, latent_dim] encoded features
        """
        encoded = self.image_encoder(obs_dict)  # [B, num_cameras, latent_dim]
        batch_size = encoded.shape[0]
        return encoded.view(batch_size, -1, self.latent_dim)


def create_statevla_model(
    # Camera configuration
    camera_names: Optional[List[str]] = None,
    # Dimension configs
    latent_dim: int = 256,
    lang_emb_dim: int = 512,
    state_encoder_type: str = 'mlp',
    state_dim: int = 256,
    action_dim: int = 7,
    action_seq_len: int = 10,
    # Architecture configs
    state_predictor_layers: int = 4,
    policy_layers: int = 3,
    policy_embed_dim: int = 256,
    use_correction: bool = True,
    # Encoder configs
    image_encoder_type: str = 'resnet',  # 'resnet' or 'eagle2'
    eagle2_model_name: str = 'nvidia/Eagle2-1B',
    freeze_vision_encoder: bool = True,
    language_encoder_type: str = 'clip',  # 'clip' or 'qwen'
    qwen_model_name: str = 'Qwen/Qwen-7B-Chat',
    use_language_encoder: bool = True,
    freeze_language_encoder: bool = True,
    clip_model_name: str = 'ViT-B/32',
    # General configs
    dropout: float = 0.1,
    device: str = 'cuda',
    # Optional dataloader for auto-config
    dataloader: Any = None
) -> StateVLAWithEncoders:
    """
    Create a complete StateVLA model with all encoders.

    Args:
        camera_names: List of camera names (e.g., ['agentview', 'eye_in_hand'])
        latent_dim: Vision encoder output dimension
        lang_emb_dim: Language embedding dimension
        state_encoder_type: Type of state encoder ('mlp' or 'cross_attention')
        state_dim: State latent dimension
        action_dim: Robot action dimension
        action_seq_len: Length of action sequence to predict
        state_predictor_layers: Number of Mamba layers in state predictor
        policy_layers: Number of Mamba layers in action policy
        policy_embed_dim: Embedding dimension for action policy
        use_correction: Whether to use correction MLP
        image_encoder_type: Type of vision encoder ('resnet' or 'eagle2')
        eagle2_model_name: Eagle2 model name if using eagle2 encoder
        freeze_vision_encoder: Whether to freeze vision encoder weights
        language_encoder_type: Type of language encoder ('clip' or 'qwen')
        qwen_model_name: Qwen model name if using qwen encoder
        use_language_encoder: Whether to create language encoder
        freeze_language_encoder: Whether to freeze language encoder weights
        clip_model_name: CLIP model variant
        dropout: Dropout rate
        device: Device to place model on
        dataloader: Optional dataloader for auto-configuration

    Returns:
        StateVLAWithEncoders: Complete model ready for training/inference
    """
    # Auto-detect camera names from dataloader if not provided
    if camera_names is None:
        if dataloader is not None:
            camera_names = _extract_camera_names(dataloader)
        else:
            camera_names = ['agentview', 'eye_in_hand']  # Default

    obs_tok_len = len(camera_names)

    # Create vision encoder
    if image_encoder_type == 'eagle2':
        from backbone.eagle2 import MultiImageEagle2Encoder
        image_encoder = MultiImageEagle2Encoder(
            camera_names=camera_names,
            model_name=eagle2_model_name,
            latent_dim=latent_dim,
            freeze_backbone=freeze_vision_encoder,
            share_backbone=True,
            device=device
        ).to(device)
    else:  # resnet (default)
        image_encoder = MultiImageResNetEncoder(
            camera_names=camera_names,
            latent_dim=latent_dim
        ).to(device)

    obs_encoder = ObservationEncoder(
        image_encoder=image_encoder,
        camera_names=camera_names,
        latent_dim=latent_dim
    ).to(device)

    # Create language encoder
    if use_language_encoder:
        if language_encoder_type == 'qwen':
            from backbone.qwen import QwenLanguageEncoder
            language_encoder = QwenLanguageEncoder(
                model_name=qwen_model_name,
                freeze_backbone=freeze_language_encoder,
                device=device
            ).to(device)
            lang_emb_dim = language_encoder.embed_dim
        else:  # clip (default)
            language_encoder = LangClip(
                freeze_backbone=freeze_language_encoder,
                model_name=clip_model_name
            ).to(device)
            lang_emb_dim = language_encoder.embed_dim
    else:
        language_encoder = None

    # Create core StateVLA model
    statevla = StateVLA(
        latent_dim=latent_dim,
        lang_emb_dim=lang_emb_dim,
        obs_tok_len=obs_tok_len,
        state_encoder_type=state_encoder_type,
        state_dim=state_dim,
        state_predictor_layers=state_predictor_layers,
        action_dim=action_dim,
        action_seq_len=action_seq_len,
        policy_layers=policy_layers,
        policy_embed_dim=policy_embed_dim,
        use_correction=use_correction,
        dropout=dropout,
        device=device
    ).to(device)

    # Create complete model
    model = StateVLAWithEncoders(
        statevla=statevla,
        obs_encoder=obs_encoder,
        language_encoder=language_encoder
    ).to(device)

    return model


def create_statevla_core(
    latent_dim: int = 256,
    lang_emb_dim: int = 512,
    state_dim: int = 256,
    action_dim: int = 7,
    action_seq_len: int = 10,
    obs_tok_len: int = 2,
    state_predictor_layers: int = 4,
    policy_layers: int = 3,
    policy_embed_dim: int = 256,
    use_correction: bool = True,
    dropout: float = 0.1,
    device: str = 'cuda'
) -> StateVLA:
    """
    Create only the core StateVLA model (without encoders).

    Useful when you want to use custom encoders or pre-computed embeddings.

    Returns:
        StateVLA: Core model without vision/language encoders
    """
    return StateVLA(
        latent_dim=latent_dim,
        lang_emb_dim=lang_emb_dim,
        obs_tok_len=obs_tok_len,
        state_dim=state_dim,
        state_predictor_layers=state_predictor_layers,
        action_dim=action_dim,
        action_seq_len=action_seq_len,
        policy_layers=policy_layers,
        policy_embed_dim=policy_embed_dim,
        use_correction=use_correction,
        dropout=dropout,
        device=device
    ).to(device)


def _extract_camera_names(dataloader) -> List[str]:
    """
    Extract camera names from dataloader.

    Args:
        dataloader: PyTorch dataloader

    Returns:
        List of camera names
    """
    try:
        # Try to get a sample batch
        sample = next(iter(dataloader))
        if isinstance(sample, dict):
            obs_dict = sample
        elif isinstance(sample, (tuple, list)):
            obs_dict = sample[0]
        else:
            return ['agentview', 'eye_in_hand']

        # Find keys ending with '_image'
        camera_names = []
        for key in obs_dict.keys():
            if key.endswith('_image'):
                camera_name = key.replace('_image', '')
                camera_names.append(camera_name)

        return camera_names if camera_names else ['agentview', 'eye_in_hand']
    except Exception:
        return ['agentview', 'eye_in_hand']


def count_parameters(model: nn.Module) -> Dict[str, int]:
    """
    Count model parameters.

    Args:
        model: PyTorch model

    Returns:
        Dictionary with parameter counts
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = total - trainable

    return {
        'total': total,
        'trainable': trainable,
        'frozen': frozen,
        'total_millions': total / 1e6,
        'trainable_millions': trainable / 1e6
    }


def load_statevla_checkpoint(
    checkpoint_path: str,
    model: Optional[StateVLAWithEncoders] = None,
    device: str = 'cuda',
    **kwargs
) -> StateVLAWithEncoders:
    """
    Load StateVLA from checkpoint.

    Args:
        checkpoint_path: Path to checkpoint file
        model: Optional existing model to load weights into
        device: Device to load model on
        **kwargs: Arguments for create_statevla_model if model not provided

    Returns:
        Model with loaded weights
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)

    if model is None:
        # Create model from checkpoint config if available
        if 'config' in checkpoint:
            config = checkpoint['config']
            model = create_statevla_model(**config, device=device)
        else:
            model = create_statevla_model(**kwargs, device=device)

    # Load state dict
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    elif 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint)

    return model
