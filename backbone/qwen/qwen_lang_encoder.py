"""
Qwen-based Language Encoder for StateVLA.

Uses Qwen-7B-Chat tokenizer and language model for text encoding.
"""

import torch
import torch.nn as nn
from typing import List, Union, Optional


class QwenLanguageEncoder(nn.Module):
    """
    Language encoder using Qwen model for text instructions.

    Uses Qwen-7B-Chat or Qwen2 series models to encode language instructions
    into embeddings for robot manipulation tasks.
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen-7B-Chat",
        embed_dim: int = 4096,
        freeze_backbone: bool = True,
        use_pooled_output: bool = True,
        max_length: int = 77,
        device: str = 'cuda'
    ):
        """
        Args:
            model_name: Qwen model name from HuggingFace
            embed_dim: Output embedding dimension (4096 for Qwen-7B)
            freeze_backbone: Whether to freeze model weights
            use_pooled_output: Whether to use mean pooling (True) or last token (False)
            max_length: Maximum sequence length
            device: Device to load model on
        """
        super().__init__()

        self.model_name = model_name
        self._embed_dim = embed_dim
        self.use_pooled_output = use_pooled_output
        self.max_length = max_length
        self.device = device

        self._load_model(model_name, freeze_backbone)

    def _load_model(self, model_name: str, freeze_backbone: bool):
        """Load Qwen model and tokenizer."""
        from transformers import AutoTokenizer, AutoModel, AutoConfig
        import os

        # Force disable flash attention
        os.environ['TRANSFORMERS_NO_FLASH_ATTN'] = '1'

        print(f"Loading Qwen language encoder from {model_name}...")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            padding_side='right'
        )

        # Add padding token if not exists
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load config and force eager attention
        config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        config._attn_implementation = 'eager'

        # Load model (only the base model, not for generation)
        self.model = AutoModel.from_pretrained(
            model_name,
            config=config,
            trust_remote_code=True,
            torch_dtype=torch.float32,
            device_map=None,
            attn_implementation="eager"  # Avoid flash_attn requirement
        )

        # Get actual embedding dimension from model config
        if hasattr(self.model.config, 'hidden_size'):
            self._embed_dim = self.model.config.hidden_size
        print(f"Model hidden size: {self._embed_dim}")

        if freeze_backbone:
            for param in self.model.parameters():
                param.requires_grad = False
            print("Language encoder weights frozen")

        print(f"Qwen language encoder loaded successfully")

    def forward(self, texts: Union[List[str], torch.Tensor]) -> torch.Tensor:
        """
        Encode text instructions.

        Args:
            texts: List of text strings or pre-tokenized tensor

        Returns:
            [B, 1, embed_dim] text embeddings
        """
        if isinstance(texts, torch.Tensor):
            # Already embedded, just return with proper shape
            if texts.dim() == 2:
                return texts.unsqueeze(1)
            return texts

        # Tokenize
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        ).to(self.device)

        # Forward pass
        with torch.no_grad() if not self.training else torch.enable_grad():
            outputs = self.model(**inputs)

        # Get hidden states
        if hasattr(outputs, 'last_hidden_state'):
            hidden_states = outputs.last_hidden_state
        else:
            hidden_states = outputs[0]

        # Pool the output
        if self.use_pooled_output:
            # Mean pooling over non-padding tokens
            attention_mask = inputs['attention_mask']
            mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
            sum_embeddings = torch.sum(hidden_states * mask_expanded, dim=1)
            sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
            embeddings = sum_embeddings / sum_mask
        else:
            # Use last token (for causal LM)
            # Find the last non-padding token for each sequence
            attention_mask = inputs['attention_mask']
            sequence_lengths = attention_mask.sum(dim=1) - 1
            batch_size = hidden_states.shape[0]
            embeddings = hidden_states[
                torch.arange(batch_size, device=hidden_states.device),
                sequence_lengths
            ]

        # Return with shape [B, 1, embed_dim]
        return embeddings.unsqueeze(1)

    def encode(self, texts: List[str]) -> torch.Tensor:
        """
        Encode texts (alias for forward).

        Args:
            texts: List of text strings

        Returns:
            [B, embed_dim] text embeddings (squeezed)
        """
        return self.forward(texts).squeeze(1)

    @property
    def embed_dim(self) -> int:
        """Return the embedding dimension."""
        return self._embed_dim


class QwenLanguageEncoderLite(nn.Module):
    """
    Lightweight Qwen language encoder using only the tokenizer and embeddings.

    This version doesn't load the full Qwen model, only uses the embedding layer.
    Useful for faster loading and lower memory usage when full context understanding
    is not required.
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen-7B-Chat",
        embed_dim: int = 512,
        max_length: int = 77,
        device: str = 'cuda'
    ):
        """
        Args:
            model_name: Qwen model name for tokenizer
            embed_dim: Output embedding dimension
            max_length: Maximum sequence length
            device: Device
        """
        super().__init__()

        self.model_name = model_name
        self._embed_dim = embed_dim
        self.max_length = max_length
        self.device = device

        self._load_tokenizer(model_name)

        # Learnable embedding layer
        vocab_size = self.tokenizer.vocab_size
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.projection = nn.Linear(embed_dim, embed_dim)

    def _load_tokenizer(self, model_name: str):
        """Load only the tokenizer."""
        from transformers import AutoTokenizer

        print(f"Loading Qwen tokenizer from {model_name}...")

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            padding_side='right'
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        print("Tokenizer loaded")

    def forward(self, texts: Union[List[str], torch.Tensor]) -> torch.Tensor:
        """
        Encode text instructions.

        Args:
            texts: List of text strings

        Returns:
            [B, 1, embed_dim] text embeddings
        """
        if isinstance(texts, torch.Tensor):
            if texts.dim() == 2:
                return texts.unsqueeze(1)
            return texts

        # Tokenize
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        ).to(self.device)

        # Embed tokens
        embeddings = self.embedding(inputs['input_ids'])

        # Mean pooling
        attention_mask = inputs['attention_mask']
        mask_expanded = attention_mask.unsqueeze(-1).expand(embeddings.size()).float()
        sum_embeddings = torch.sum(embeddings * mask_expanded, dim=1)
        sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
        pooled = sum_embeddings / sum_mask

        # Project
        output = self.projection(pooled)

        return output.unsqueeze(1)

    @property
    def embed_dim(self) -> int:
        return self._embed_dim


if __name__ == "__main__":
    # Test the encoder
    print("Testing Qwen Language Encoder...")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Test full encoder (requires Qwen model)
    try:
        encoder = QwenLanguageEncoder(
            model_name="Qwen/Qwen-7B-Chat",
            freeze_backbone=True,
            device=device
        ).to(device)

        texts = [
            "pick up the butter and place it in the basket",
            "open the drawer and put the apple inside"
        ]

        output = encoder(texts)
        print(f"Full encoder output shape: {output.shape}")
        print(f"Embedding dim: {encoder.embed_dim}")

    except Exception as e:
        print(f"Could not load full Qwen model: {e}")
        print("Testing lite version...")

        # Test lite encoder
        encoder_lite = QwenLanguageEncoderLite(
            model_name="Qwen/Qwen-7B-Chat",
            embed_dim=512,
            device=device
        ).to(device)

        texts = ["pick up the butter", "open the drawer"]
        output = encoder_lite(texts)
        print(f"Lite encoder output shape: {output.shape}")
