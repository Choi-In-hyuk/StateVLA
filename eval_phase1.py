"""
Phase 1 Representation Quality Evaluation

Evaluates the learned latent space (z_t) from Phase 1 Temporal JEPA:
  1. t-SNE / UMAP Visualization (task-colored, timestep-colored)
  2. Temporal Smoothness (cosine similarity between consecutive z_t)
  3. Linear Probing (task classification from frozen z_t)
  4. Nearest Neighbor Analysis (semantic consistency check)

Usage:
  python eval_phase1.py --checkpoint checkpoints/phase1_xxx/checkpoint_best.pt
"""

import argparse
import os
import logging

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import yaml

from statevla_model import StateVLA
from dataloader import StateVLADataset, collate_fn

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


def load_model(config, checkpoint_path, device):
    """Load Phase 1 model from checkpoint."""
    model_config = config['model']
    phase_config = config['training'].get('phase1', {})

    model = StateVLA(
        camera_names=config['cameras']['names'],
        image_size=model_config.get('image_size', 224),
        patch_size=model_config.get('patch_size', 16),
        embed_dim=model_config.get('embed_dim', 256),
        lang_emb_dim=model_config.get('lang_emb_dim', 512),
        robot_state_dim=model_config.get('robot_state_dim', 9),
        use_pretrained_vision=model_config.get('use_pretrained_vision', False),
        use_pretrained_language=model_config.get('use_pretrained_language', False),
        vision_model_name=model_config.get('vision_model_name', 'google/siglip-base-patch16-224'),
        language_model_name=model_config.get('language_model_name', 'ViT-B/32'),
        freeze_vision=model_config.get('freeze_vision', True),
        freeze_language=model_config.get('freeze_language', True),
        encoder_depth=model_config.get('encoder_depth', 12),
        d_state=model_config.get('d_state', 16),
        d_conv=model_config.get('d_conv', 4),
        expand=model_config.get('expand', 2),
        predictor_embed_dim=model_config.get('predictor_embed_dim', 192),
        predictor_depth=model_config.get('predictor_depth', 6),
        mask_ratio=model_config.get('mask_ratio', 0.5),
        masking_strategy=model_config.get('masking_strategy', 'modality_aware'),
        state_dim=model_config.get('state_dim', 256),
        action_dim=model_config.get('action_dim', 7),
        action_seq_len=model_config.get('action_seq_len', 10),
        policy_layers=model_config.get('policy_layers', 3),
        policy_embed_dim=model_config.get('policy_embed_dim', 256),
        temporal_hidden_dim=phase_config.get('temporal_predictor_hidden_dim', 512),
        training_phase=1,
        device=str(device),
    )

    ckpt = torch.load(checkpoint_path, map_location='cpu')
    state_dict = ckpt.get('model_state_dict', ckpt)

    # Remove 'model.' prefix if saved from StateVLATrainer
    cleaned = {}
    for k, v in state_dict.items():
        key = k.replace('model.', '') if k.startswith('model.') else k
        cleaned[key] = v

    # Filter out size-mismatched keys (e.g. action_dim 6→7 change)
    current_state = model.state_dict()
    filtered = {}
    for k, v in cleaned.items():
        if k in current_state and current_state[k].shape != v.shape:
            log.info(f"Skipping {k}: shape mismatch ({v.shape} → {current_state[k].shape})")
            continue
        filtered[k] = v
    model.load_state_dict(filtered, strict=False)
    model.to(device)
    model.eval()

    log.info(f"Loaded checkpoint: epoch {ckpt.get('epoch', '?')}, loss {ckpt.get('loss', '?'):.4f}")
    return model


@torch.no_grad()
def extract_representations(model, dataset, device, max_samples=2000, batch_size=64):
    """Extract z_t representations from all samples."""
    n = min(max_samples, len(dataset))
    indices = np.random.choice(len(dataset), n, replace=False)
    indices.sort()

    subset = Subset(dataset, indices)
    loader = DataLoader(subset, batch_size=batch_size, shuffle=False,
                        num_workers=2, collate_fn=collate_fn, pin_memory=True)

    all_z = []
    all_task_ids = []
    all_timesteps = []

    # Build trajectory_id -> task_id mapping
    # Each HDF5 file = 1 task, demos_per_task trajectories per file
    demos_per_task = dataset.demos_per_task
    traj_to_task = {}
    for traj_id in range(dataset.num_data):
        traj_to_task[traj_id] = traj_id // demos_per_task

    for i, batch_idx in enumerate(range(0, n, batch_size)):
        actual_indices = indices[batch_idx:batch_idx + batch_size]
        # Get trajectory IDs from slices
        for idx in actual_indices:
            traj_id, start, end = dataset.slices[idx]
            all_task_ids.append(traj_to_task[traj_id])

    # Reset and re-extract with task labels
    all_task_ids_final = all_task_ids
    all_task_ids = []

    for batch in loader:
        obs = {k: v.to(device) for k, v in batch['obs'].items()}
        z_t = model.state_encoder.encode(obs)  # [B, state_dim]
        all_z.append(z_t.cpu())

    all_z = torch.cat(all_z, dim=0).numpy()
    all_task_labels = np.array(all_task_ids_final)

    num_tasks = len(np.unique(all_task_labels))
    log.info(f"Extracted {len(all_z)} representations, {num_tasks} unique tasks")
    return all_z, all_task_labels, num_tasks


@torch.no_grad()
def extract_trajectory_representations(model, dataset, device, num_trajectories=20, batch_size=32):
    """Extract z_t for full trajectories (for temporal smoothness)."""
    # Group slices by trajectory
    traj_slices = {}
    for idx, (traj_id, start, end) in enumerate(dataset.slices):
        if traj_id not in traj_slices:
            traj_slices[traj_id] = []
        traj_slices[traj_id].append((idx, start))

    # Sort by start time within each trajectory
    for traj_id in traj_slices:
        traj_slices[traj_id].sort(key=lambda x: x[1])

    # Select trajectories
    traj_ids = list(traj_slices.keys())[:num_trajectories]

    trajectories = []
    for traj_id in traj_ids:
        seen_starts = set()
        z_list = []
        items = traj_slices[traj_id]

        # Collect unique start timesteps
        batch_items = []
        for idx, start in items:
            if start not in seen_starts:
                seen_starts.add(start)
                batch_items.append(idx)

        # Encode in batches
        subset = Subset(dataset, batch_items)
        loader = DataLoader(subset, batch_size=batch_size, shuffle=False,
                            num_workers=0, collate_fn=collate_fn)

        for batch in loader:
            obs = {k: v.to(device) for k, v in batch['obs'].items()}
            z_t = model.state_encoder.encode(obs)
            z_list.append(z_t.cpu())

        if z_list:
            trajectories.append(torch.cat(z_list, dim=0))

    log.info(f"Extracted {len(trajectories)} trajectories")
    return trajectories


def eval_tsne(z, task_labels, save_dir, task_names=None):
    """t-SNE visualization colored by task."""
    log.info("Running t-SNE...")
    tsne = TSNE(n_components=2, perplexity=30, random_state=42, max_iter=1000)
    z_2d = tsne.fit_transform(z)

    num_tasks = len(np.unique(task_labels))
    cmap = plt.cm.get_cmap('tab10' if num_tasks <= 10 else 'tab20')

    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    for tid in np.unique(task_labels):
        mask = task_labels == tid
        label = task_names[tid] if task_names and tid < len(task_names) else f'Task {tid}'
        ax.scatter(z_2d[mask, 0], z_2d[mask, 1], c=[cmap(tid % 20)],
                   label=label, alpha=0.5, s=10)
    ax.set_title(f't-SNE of Phase 1 Latent Space (z_t)\n{num_tasks} tasks, {len(z)} samples')
    ax.legend(loc='best', fontsize=7, markerscale=2)
    ax.set_xlabel('t-SNE 1')
    ax.set_ylabel('t-SNE 2')

    path = os.path.join(save_dir, 'tsne_by_task.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    log.info(f"Saved: {path}")


def eval_temporal_smoothness(trajectories, save_dir):
    """Analyze temporal smoothness of representations."""
    log.info("Analyzing temporal smoothness...")

    all_cosine_sims = []
    all_l2_dists = []
    per_traj_sims = []

    for traj_z in trajectories:
        if len(traj_z) < 2:
            continue
        # Cosine similarity between consecutive frames
        z_curr = F.normalize(traj_z[:-1], dim=-1)
        z_next = F.normalize(traj_z[1:], dim=-1)
        cosine_sim = (z_curr * z_next).sum(dim=-1)
        all_cosine_sims.append(cosine_sim)

        # L2 distance
        l2_dist = torch.norm(traj_z[1:] - traj_z[:-1], dim=-1)
        all_l2_dists.append(l2_dist)

        per_traj_sims.append(cosine_sim.mean().item())

    all_cosine_sims = torch.cat(all_cosine_sims).numpy()
    all_l2_dists = torch.cat(all_l2_dists).numpy()

    # Print stats
    log.info(f"Cosine Similarity (consecutive frames):")
    log.info(f"  Mean: {all_cosine_sims.mean():.4f}")
    log.info(f"  Std:  {all_cosine_sims.std():.4f}")
    log.info(f"  Min:  {all_cosine_sims.min():.4f}")
    log.info(f"  Max:  {all_cosine_sims.max():.4f}")
    log.info(f"L2 Distance (consecutive frames):")
    log.info(f"  Mean: {all_l2_dists.mean():.4f}")
    log.info(f"  Std:  {all_l2_dists.std():.4f}")

    # Plot 1: Cosine similarity distribution
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    axes[0].hist(all_cosine_sims, bins=50, edgecolor='black', alpha=0.7)
    axes[0].axvline(all_cosine_sims.mean(), color='red', linestyle='--',
                    label=f'Mean={all_cosine_sims.mean():.3f}')
    axes[0].set_title('Cosine Similarity Distribution\n(consecutive frames)')
    axes[0].set_xlabel('Cosine Similarity')
    axes[0].set_ylabel('Count')
    axes[0].legend()

    # Plot 2: Cosine similarity over time for sample trajectories
    for i, traj_z in enumerate(trajectories[:5]):
        if len(traj_z) < 2:
            continue
        z_curr = F.normalize(traj_z[:-1], dim=-1)
        z_next = F.normalize(traj_z[1:], dim=-1)
        sim = (z_curr * z_next).sum(dim=-1).numpy()
        axes[1].plot(sim, alpha=0.7, label=f'Traj {i}')
    axes[1].set_title('Cosine Similarity Over Time\n(sample trajectories)')
    axes[1].set_xlabel('Timestep')
    axes[1].set_ylabel('Cosine Similarity')
    axes[1].legend(fontsize=8)

    # Plot 3: L2 distance distribution
    axes[2].hist(all_l2_dists, bins=50, edgecolor='black', alpha=0.7, color='orange')
    axes[2].axvline(all_l2_dists.mean(), color='red', linestyle='--',
                    label=f'Mean={all_l2_dists.mean():.3f}')
    axes[2].set_title('L2 Distance Distribution\n(consecutive frames)')
    axes[2].set_xlabel('L2 Distance')
    axes[2].set_ylabel('Count')
    axes[2].legend()

    fig.suptitle('Temporal Smoothness Analysis', fontsize=14, y=1.02)
    path = os.path.join(save_dir, 'temporal_smoothness.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    log.info(f"Saved: {path}")

    return {
        'cosine_mean': float(all_cosine_sims.mean()),
        'cosine_std': float(all_cosine_sims.std()),
        'l2_mean': float(all_l2_dists.mean()),
        'l2_std': float(all_l2_dists.std()),
    }


def eval_linear_probe(z, task_labels, save_dir, epochs=100, lr=1e-3):
    """Linear probing: train linear classifier on frozen z_t for task classification."""
    log.info("Running linear probing...")

    num_tasks = len(np.unique(task_labels))
    if num_tasks < 2:
        log.warning("Only 1 task found, skipping linear probe")
        return {'accuracy': 0.0}

    # Train/test split
    n = len(z)
    perm = np.random.permutation(n)
    split = int(0.8 * n)
    train_idx, test_idx = perm[:split], perm[split:]

    z_train = torch.from_numpy(z[train_idx]).float()
    y_train = torch.from_numpy(task_labels[train_idx]).long()
    z_test = torch.from_numpy(z[test_idx]).float()
    y_test = torch.from_numpy(task_labels[test_idx]).long()

    # Simple linear classifier
    dim = z.shape[1]
    classifier = nn.Linear(dim, num_tasks)
    optimizer = torch.optim.Adam(classifier.parameters(), lr=lr)

    # Train
    classifier.train()
    for epoch in range(epochs):
        logits = classifier(z_train)
        loss = F.cross_entropy(logits, y_train)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Evaluate
    classifier.eval()
    with torch.no_grad():
        pred = classifier(z_test).argmax(dim=-1).numpy()
        acc = accuracy_score(y_test.numpy(), pred)

    log.info(f"Linear Probe Accuracy: {acc:.4f} ({num_tasks} tasks)")

    return {'accuracy': float(acc), 'num_tasks': num_tasks}


def eval_nearest_neighbor(z, task_labels, save_dir, k=5):
    """Nearest neighbor analysis: check if neighbors share same task."""
    log.info(f"Running {k}-NN analysis...")

    nn_model = NearestNeighbors(n_neighbors=k + 1, metric='cosine')
    nn_model.fit(z)
    distances, indices = nn_model.kneighbors(z)

    # Exclude self (first neighbor)
    neighbor_indices = indices[:, 1:]
    neighbor_labels = task_labels[neighbor_indices]

    # Compute same-task ratio
    same_task = (neighbor_labels == task_labels[:, None]).astype(float)
    same_task_ratio = same_task.mean()
    per_k_ratio = same_task.mean(axis=0)

    num_tasks = len(np.unique(task_labels))
    random_baseline = 1.0 / num_tasks

    log.info(f"Same-task ratio (k={k}): {same_task_ratio:.4f} (random baseline: {random_baseline:.4f})")
    for ki in range(k):
        log.info(f"  k={ki + 1}: {per_k_ratio[ki]:.4f}")

    # Plot
    fig, ax = plt.subplots(figsize=(8, 5))
    ks = list(range(1, k + 1))
    ax.bar(ks, per_k_ratio, color='steelblue', edgecolor='black', alpha=0.8)
    ax.axhline(random_baseline, color='red', linestyle='--',
               label=f'Random baseline ({random_baseline:.3f})')
    ax.set_xlabel('k-th Nearest Neighbor')
    ax.set_ylabel('Same-Task Ratio')
    ax.set_title(f'Nearest Neighbor Task Consistency\n({num_tasks} tasks, {len(z)} samples)')
    ax.set_xticks(ks)
    ax.legend()

    path = os.path.join(save_dir, 'nearest_neighbor.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    log.info(f"Saved: {path}")

    return {
        'same_task_ratio': float(same_task_ratio),
        'random_baseline': float(random_baseline),
        'per_k_ratio': per_k_ratio.tolist(),
    }


@torch.no_grad()
def eval_temporal_prediction(model, dataset, device, save_dir, max_samples=1000, batch_size=32):
    """
    Evaluate temporal prediction MSE: the core Phase 1 objective.

    Measures how well the world model predicts z_{t+1} from z_t + a_t.
    This is the MOST DIRECT metric for Phase 1 quality.

    Good values:
      MSE < 0.1  → good prediction
      Cosine Sim > 0.9 → representations are directionally aligned
    """
    log.info("Evaluating temporal prediction (core Phase 1 objective)...")

    # Only use samples that have a valid next observation
    valid_indices = []
    for idx, (traj_id, start, end) in enumerate(dataset.slices):
        seq_length = dataset._get_seq_length(traj_id)
        if start + 1 < seq_length:
            valid_indices.append(idx)

    n = min(max_samples, len(valid_indices))
    selected = np.random.choice(len(valid_indices), n, replace=False).tolist()
    selected = [valid_indices[i] for i in selected]

    subset = Subset(dataset, selected)
    loader = DataLoader(subset, batch_size=batch_size, shuffle=False,
                        num_workers=2, collate_fn=collate_fn, pin_memory=True)

    all_mse = []
    all_cosine = []

    for batch in loader:
        next_obs = batch.get('next_obs')
        if next_obs is None:
            continue

        obs = {k: v.to(device) for k, v in batch['obs'].items()}
        next_obs = {k: v.to(device) for k, v in next_obs.items()}
        actions = batch['actions'][:, 0, :].to(device)  # a_t: first action [B, 7]

        # Phase 1 forward: z_t + a_t → z'_{t+1} vs z_{t+1}
        outputs = model.forward_phase1(obs, next_obs, actions)
        z_next_pred = outputs['z_next_pred']    # [B, 256]
        z_next_target = outputs['z_next_target']  # [B, 256]

        # MSE per sample
        mse = ((z_next_pred - z_next_target) ** 2).mean(dim=-1)  # [B]
        all_mse.append(mse.cpu())

        # Cosine similarity
        cos = F.cosine_similarity(z_next_pred, z_next_target, dim=-1)  # [B]
        all_cosine.append(cos.cpu())

    all_mse = torch.cat(all_mse).numpy()
    all_cosine = torch.cat(all_cosine).numpy()

    log.info(f"Temporal Prediction Results ({n} samples):")
    log.info(f"  MSE:        {all_mse.mean():.4f} ± {all_mse.std():.4f}")
    log.info(f"  Cosine Sim: {all_cosine.mean():.4f} ± {all_cosine.std():.4f}")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].hist(all_mse, bins=50, edgecolor='black', alpha=0.7, color='steelblue')
    axes[0].axvline(all_mse.mean(), color='red', linestyle='--',
                    label=f'Mean={all_mse.mean():.4f}')
    axes[0].set_title('Temporal Prediction MSE\n(z\'_{t+1} vs z_{t+1})')
    axes[0].set_xlabel('MSE')
    axes[0].set_ylabel('Count')
    axes[0].legend()

    axes[1].hist(all_cosine, bins=50, edgecolor='black', alpha=0.7, color='green')
    axes[1].axvline(all_cosine.mean(), color='red', linestyle='--',
                    label=f'Mean={all_cosine.mean():.4f}')
    axes[1].set_title('Temporal Prediction Cosine Similarity\n(z\'_{t+1} vs z_{t+1})')
    axes[1].set_xlabel('Cosine Similarity')
    axes[1].set_ylabel('Count')
    axes[1].legend()

    fig.suptitle('Phase 1 World Model Quality', fontsize=14)
    path = os.path.join(save_dir, 'temporal_prediction.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    log.info(f"Saved: {path}")

    return {
        'mse_mean': float(all_mse.mean()),
        'mse_std': float(all_mse.std()),
        'cosine_mean': float(all_cosine.mean()),
        'cosine_std': float(all_cosine.std()),
    }


def eval_representation_stats(z, save_dir):
    """Basic statistics of the representation space."""
    log.info("Computing representation statistics...")

    z_tensor = torch.from_numpy(z).float()

    # Per-dimension variance
    var = z_tensor.var(dim=0)
    # Effective rank (how many dimensions are actually used)
    svd_vals = torch.linalg.svdvals(z_tensor - z_tensor.mean(dim=0))
    svd_normalized = svd_vals / svd_vals.sum()
    entropy = -(svd_normalized * torch.log(svd_normalized + 1e-10)).sum()
    effective_rank = torch.exp(entropy).item()

    # Isotropy: how uniformly spread in all directions
    mean_var = var.mean().item()
    std_var = var.std().item()

    log.info(f"Representation Statistics:")
    log.info(f"  Dimension: {z.shape[1]}")
    log.info(f"  Effective Rank: {effective_rank:.1f} / {z.shape[1]}")
    log.info(f"  Mean Variance: {mean_var:.4f}")
    log.info(f"  Std of Variance: {std_var:.4f}")
    log.info(f"  Norm (mean): {np.linalg.norm(z, axis=1).mean():.4f}")

    # Plot variance per dimension
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].bar(range(len(var)), var.numpy(), alpha=0.7)
    axes[0].set_title(f'Per-Dimension Variance\n(Effective Rank: {effective_rank:.1f}/{z.shape[1]})')
    axes[0].set_xlabel('Dimension')
    axes[0].set_ylabel('Variance')

    # Singular value spectrum
    axes[1].plot(svd_vals.numpy()[:50], 'o-', markersize=3)
    axes[1].set_title('Singular Value Spectrum (top 50)')
    axes[1].set_xlabel('Component')
    axes[1].set_ylabel('Singular Value')
    axes[1].set_yscale('log')

    path = os.path.join(save_dir, 'representation_stats.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    log.info(f"Saved: {path}")

    return {
        'effective_rank': effective_rank,
        'total_dim': z.shape[1],
        'mean_variance': mean_var,
        'norm_mean': float(np.linalg.norm(z, axis=1).mean()),
    }


def main():
    parser = argparse.ArgumentParser(description='Phase 1 Representation Evaluation')
    parser.add_argument('--checkpoint', type=str, required=True, help='Phase 1 checkpoint path')
    parser.add_argument('--config', type=str, default='conf/config.yaml')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--max_samples', type=int, default=2000)
    parser.add_argument('--num_trajectories', type=int, default=20)
    parser.add_argument('--save_dir', type=str, default='eval_results/phase1')
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    with open(args.config) as f:
        config = yaml.safe_load(f)

    device = torch.device(args.device)

    # Load model
    log.info("Loading model...")
    model = load_model(config, args.checkpoint, device)

    # Load dataset
    log.info("Loading dataset...")
    dataset = StateVLADataset(
        data_directory=config['data']['data_directory'],
        action_dim=config['model']['action_dim'],
        action_seq_len=config['model']['action_seq_len'],
        demos_per_task=config['data']['demos_per_task'],
        max_len_data=config['data']['max_len_data'],
        image_size=config['model'].get('image_size', 224),
        camera_names=config['cameras']['names'],
    )

    # Get task names from HDF5 filenames
    data_dir = config['data']['data_directory']
    task_names = sorted([
        f.replace('_demo.hdf5', '').replace('.hdf5', '')
        for f in os.listdir(data_dir) if f.endswith('.hdf5')
    ])
    log.info(f"Tasks: {task_names}")

    # 1. Temporal prediction (CORE Phase 1 objective)
    log.info("\n=== [CORE] Temporal Prediction MSE ===")
    temporal_pred = eval_temporal_prediction(
        model, dataset, device, args.save_dir, max_samples=args.max_samples, batch_size=64)

    # 2. Extract representations
    log.info("\n=== Extracting Representations ===")
    z, task_labels, num_tasks = extract_representations(
        model, dataset, device, max_samples=args.max_samples)

    # 3. Representation statistics
    log.info("\n=== Representation Statistics ===")
    stats = eval_representation_stats(z, args.save_dir)

    # 4. t-SNE visualization
    log.info("\n=== t-SNE Visualization ===")
    eval_tsne(z, task_labels, args.save_dir, task_names=task_names)

    # 5. Temporal smoothness
    log.info("\n=== Temporal Smoothness ===")
    trajectories = extract_trajectory_representations(
        model, dataset, device, num_trajectories=args.num_trajectories)
    smoothness = eval_temporal_smoothness(trajectories, args.save_dir)

    # 6. Linear probing
    log.info("\n=== Linear Probing ===")
    probe = eval_linear_probe(z, task_labels, args.save_dir)

    # 7. Nearest neighbor
    log.info("\n=== Nearest Neighbor Analysis ===")
    nn_results = eval_nearest_neighbor(z, task_labels, args.save_dir)

    # Summary
    log.info("\n" + "=" * 60)
    log.info("EVALUATION SUMMARY")
    log.info("=" * 60)
    log.info(f"[CORE] Temporal Pred MSE:  {temporal_pred['mse_mean']:.4f} ± {temporal_pred['mse_std']:.4f}  (낮을수록 좋음)")
    log.info(f"[CORE] Temporal Pred Cos:  {temporal_pred['cosine_mean']:.4f} ± {temporal_pred['cosine_std']:.4f}  (1에 가까울수록 좋음)")
    log.info(f"Effective Rank:            {stats['effective_rank']:.1f} / {stats['total_dim']}  (높을수록 좋음)")
    log.info(f"Temporal Smoothness Cos:   {smoothness['cosine_mean']:.4f} ± {smoothness['cosine_std']:.4f}")
    log.info(f"Linear Probe Accuracy:     {probe['accuracy']:.4f}")
    log.info(f"NN Same-Task Ratio:        {nn_results['same_task_ratio']:.4f} (baseline: {nn_results['random_baseline']:.4f})")
    log.info(f"\nResults saved to: {args.save_dir}")


if __name__ == '__main__':
    main()
