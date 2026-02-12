"""
Generate CLIP language embeddings for LIBERO tasks.

Encodes task names from HDF5 filenames using CLIP ViT-B/32 text encoder
and saves them as a pickle file for the dataloader.

Usage:
    python generate_language_embeddings.py --data_dir /path/to/libero_object
"""

import argparse
import os
import pickle
import sys

import torch

# Add project root to path for CLIP imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils.networks.clip import build_model, load_clip, tokenize


def get_task_names(data_dir: str):
    """Extract task names from HDF5 filenames."""
    tasks = {}
    for f in sorted(os.listdir(data_dir)):
        if not f.endswith('.hdf5'):
            continue
        name = f.replace('.hdf5', '')
        if name.endswith('_demo'):
            name = name[:-5]
        # Convert underscores to spaces for CLIP encoding
        text = name.replace('_', ' ')
        tasks[name] = text
    return tasks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str,
                        default='/home/choi/choi_ws/openvla/datasets/libero_object')
    parser.add_argument('--clip_model', type=str, default='ViT-B/32')
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    # Get task names
    tasks = get_task_names(args.data_dir)
    print(f"Found {len(tasks)} tasks:")
    for name, text in tasks.items():
        print(f"  {name} -> \"{text}\"")

    # Load CLIP
    print(f"\nLoading CLIP {args.clip_model}...")
    model, _ = load_clip(args.clip_model, device=args.device)
    clip_model = build_model(model.state_dict()).to(args.device)
    clip_model.eval()

    # Encode task names
    print("Encoding task names...")
    embeddings = {}
    with torch.no_grad():
        for name, text in tasks.items():
            tokens = tokenize([text]).to(args.device)
            emb = clip_model.encode_text(tokens)  # [1, 512]
            embeddings[name] = emb.squeeze(0).cpu()
            print(f"  {name}: shape={embeddings[name].shape}, norm={embeddings[name].norm():.4f}")

    # Save
    benchmark_type = os.path.basename(args.data_dir)
    out_dir = os.path.join(os.path.dirname(args.data_dir), 'language_embeddings')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f'{benchmark_type}.pkl')

    with open(out_path, 'wb') as f:
        pickle.dump(embeddings, f)

    print(f"\nSaved to: {out_path}")
    print(f"Keys: {list(embeddings.keys())}")

    # Verify
    with open(out_path, 'rb') as f:
        loaded = pickle.load(f)
    print(f"Verification: loaded {len(loaded)} embeddings, shapes match: "
          f"{all(v.shape == torch.Size([512]) for v in loaded.values())}")


if __name__ == '__main__':
    main()
