"""
Generate language embeddings for LIBERO tasks.

Supports multiple encoders:
  - Qwen (Qwen-7B-Chat, Qwen2 series)
  - CLIP (ViT-B/32, etc.)
"""

import os
import pickle
import argparse

import torch


def get_task_names_from_hdf5_dir(data_dir: str):
    """Extract task names from HDF5 filenames."""
    task_names = []
    for f in os.listdir(data_dir):
        if f.endswith('.hdf5'):
            # Remove '_demo.hdf5' suffix
            task_name = f.replace('_demo.hdf5', '').replace('.hdf5', '')
            # Convert underscores to spaces for natural language
            task_description = task_name.replace('_', ' ')
            task_names.append((task_name, task_description))
    return task_names


def generate_qwen_embeddings(data_dir: str, output_path: str, model_name: str = "Qwen/Qwen-7B-Chat", use_cpu: bool = False):
    """Generate embeddings using Qwen model."""
    from transformers import AutoTokenizer, AutoModel

    # Force CPU if GPU not compatible or requested
    device = "cpu" if use_cpu else ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading Qwen model {model_name} on {device}...")

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModel.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32
    ).to(device)
    model.eval()

    embed_dim = model.config.hidden_size
    print(f"Model hidden size: {embed_dim}")

    # Get task names
    task_names = get_task_names_from_hdf5_dir(data_dir)
    print(f"Found {len(task_names)} tasks")

    embeddings = {}

    with torch.no_grad():
        for task_name, task_description in task_names:
            print(f"  Processing: {task_description}")

            # Tokenize
            inputs = tokenizer(
                task_description,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=77
            ).to(device)

            # Forward pass
            outputs = model(**inputs)

            # Mean pooling
            hidden_states = outputs.last_hidden_state
            attention_mask = inputs['attention_mask']
            mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
            sum_embeddings = torch.sum(hidden_states * mask_expanded, dim=1)
            sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
            text_features = sum_embeddings / sum_mask

            # Store as tensor
            embeddings[task_name] = text_features.cpu().float().squeeze(0)

    # Save embeddings
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'wb') as f:
        pickle.dump(embeddings, f)

    print(f"Saved embeddings to {output_path}")
    print(f"Embedding shape: {list(embeddings.values())[0].shape}")

    return embeddings


def generate_clip_embeddings(data_dir: str, output_path: str, model_name: str = "ViT-B/32"):
    """Generate embeddings using CLIP model."""
    import clip

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading CLIP model {model_name} on {device}...")
    model, _ = clip.load(model_name, device=device)

    # Get task names
    task_names = get_task_names_from_hdf5_dir(data_dir)
    print(f"Found {len(task_names)} tasks")

    embeddings = {}

    with torch.no_grad():
        for task_name, task_description in task_names:
            print(f"  Processing: {task_description}")

            # Tokenize and encode
            text = clip.tokenize([task_description]).to(device)
            text_features = model.encode_text(text)

            # Normalize
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

            # Store as tensor
            embeddings[task_name] = text_features.cpu().float().squeeze(0)

    # Save embeddings
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'wb') as f:
        pickle.dump(embeddings, f)

    print(f"Saved embeddings to {output_path}")
    print(f"Embedding shape: {list(embeddings.values())[0].shape}")

    return embeddings


def main():
    parser = argparse.ArgumentParser(description='Generate language embeddings for LIBERO')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Path to LIBERO dataset directory (e.g., libero_object)')
    parser.add_argument('--output_path', type=str, default=None,
                        help='Output path for embeddings pickle file')
    parser.add_argument('--encoder', type=str, default='clip',
                        choices=['qwen', 'clip'],
                        help='Encoder type: qwen or clip')
    parser.add_argument('--model_name', type=str, default=None,
                        help='Model name (default: Qwen/Qwen-7B-Chat for qwen, ViT-B/32 for clip)')
    parser.add_argument('--cpu', action='store_true',
                        help='Force CPU inference (useful for GPU compatibility issues)')
    args = parser.parse_args()

    # Default model names
    if args.model_name is None:
        if args.encoder == 'qwen':
            args.model_name = "Qwen/Qwen2-7B-Instruct"
        else:
            args.model_name = "ViT-B/32"

    # Default output path
    if args.output_path is None:
        dataset_name = os.path.basename(args.data_dir.rstrip('/'))
        parent_dir = os.path.dirname(args.data_dir.rstrip('/'))
        args.output_path = os.path.join(parent_dir, 'language_embeddings', f'{dataset_name}.pkl')

    # Generate embeddings
    if args.encoder == 'qwen':
        generate_qwen_embeddings(args.data_dir, args.output_path, args.model_name, use_cpu=args.cpu)
    else:
        generate_clip_embeddings(args.data_dir, args.output_path, args.model_name)


if __name__ == '__main__':
    main()
