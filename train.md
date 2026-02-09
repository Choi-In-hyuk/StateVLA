# StateVLA JEPA Training Guide

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    Input Tokenization                       │
│  [agentview] [eye_in_hand] [language] [robot_state]         │
│       ↓           ↓            ↓           ↓                │
│   ViT패치     ViT패치       토큰        토큰                │
│   (196개)    (196개)       (1개)       (1개)                │
└───────────────────────┬─────────────────────────────────────┘
                        ↓
              [전체 토큰: ~394개]
                        ↓
                 랜덤 마스킹 (50%)
        ┌───────────────┴───────────────┐
        ↓                               ↓
┌───────────────┐               ┌───────────────┐
│Context Encoder│               │Target Encoder │
│  (Mamba SSM)  │               │  (Mamba EMA)  │
│  visible만    │               │   전체 토큰    │
└───────┬───────┘               └───────┬───────┘
        ↓                               ↓
┌───────────────┐               ┌───────────────┐
│   Predictor   │──── Loss ────│  타겟 latent   │
│ (masked 예측)  │   (MSE+VICReg)│               │
└───────────────┘               └───────────────┘
        ↓
      z_t (CLS token)
        ↓
┌─────────────────────────────────────┐
│          Action Policy              │
│  Position/Rotation: Flow Matching   │
│  Gripper: Binary Classification     │
└─────────────────────────────────────┘
        ↓
     action [7 dims]
```

## Model Configuration

| Component | Model | Dimension | Status |
|-----------|-------|-----------|--------|
| Vision Encoder | SigLIP (google/siglip-base-patch16-224) | 768 → 256 | Frozen |
| Language Encoder | Pre-computed Qwen embeddings | 3584 → 256 | Frozen |
| Mamba Encoder | 12 layers, d_state=16, d_conv=4 | 256 | Trainable |
| JEPA Predictor | 6 layers | 192 | Trainable |
| Action Policy | Flow Matching + Gripper Classifier | 256 | Trainable |

### Key Features
- **Gripper**: 별도 Binary Classification Head (BCE Loss)
- **Position/Rotation**: Flow Matching (MSE Loss)
- **Action Normalization**: pos/rot만 정규화, gripper는 그대로 (-1, 1)

## Dataset

### LIBERO-Object
- **Location**: `data/libero/libero_object`
- **Tasks**: 10개 pick-and-place tasks
- **Trajectories**: 500개 (각 태스크당 50개)
- **Total samples**: ~70,000개
- **Cameras**: agentview, eye_in_hand (224x224)

---

## Single GPU Training

```bash
python train.py --config conf/config_libero_object.yaml
```

### Options
```bash
# Resume from checkpoint
python train.py --config conf/config_libero_object.yaml \
    --checkpoint checkpoints/libero_object/jepa_xxx/checkpoint_latest.pt

# Override batch size (GPU memory 부족시)
python train.py --config conf/config_libero_object.yaml --batch_size 64

# Override data directory
python train.py --config conf/config_libero_object.yaml \
    --data_directory /path/to/data
```

---

## Multi-GPU Training (DDP)

```bash
# 2 GPUs
torchrun --nproc_per_node=2 train.py --config conf/config_libero_object.yaml

# 4 GPUs
torchrun --nproc_per_node=4 train.py --config conf/config_libero_object.yaml

# 8 GPUs
torchrun --nproc_per_node=8 train.py --config conf/config_libero_object.yaml
```

---

## Training on Another Computer

### 1. Clone Repository
```bash
git clone https://github.com/Choi-In-hyuk/StateVLA.git
cd StateVLA
```

### 2. Install Dependencies
```bash
pip install torch torchvision
pip install mamba-ssm
pip install transformers
pip install pyyaml tqdm numpy imageio h5py
```

### 3. Prepare Data
```bash
# Option 1: Symlink
ln -s /path/to/libero/data data

# Option 2: Copy
cp -r /source/data ./data

# Data structure:
# data/
# └── libero/
#     ├── libero_object/
#     │   ├── pick_up_the_butter_and_place_it_in_the_basket_demo.hdf5
#     │   ├── pick_up_the_chocolate_pudding_and_place_it_in_the_basket_demo.hdf5
#     │   └── ...
#     └── language_embeddings/
#         └── libero_object.pkl
```

### 4. Run Training
```bash
# Single GPU
python train.py --config conf/config_libero_object.yaml

# Multi-GPU (DDP)
torchrun --nproc_per_node=4 train.py --config conf/config_libero_object.yaml
```

---

## GPU Memory Guide

| Batch Size | GPU Memory (Approx) |
|------------|---------------------|
| 256        | ~40GB               |
| 128        | ~24GB               |
| 64         | ~16GB               |
| 32         | ~10GB               |

---

## Training Configuration

### Current Settings (`conf/config_libero_object.yaml`)
```yaml
training:
  batch_size: 256
  num_epochs: 2000
  learning_rate: 1.0e-4
  weight_decay: 0.05
  gradient_clip: 1.0

  # Loss weights
  jepa_loss_weight: 1.0
  action_loss_weight: 1.0

  # VICReg regularization
  variance_weight: 1.0
  covariance_weight: 0.04

  # EMA target encoder
  ema_momentum: 0.996
  ema_momentum_schedule: "cosine"

  # Flow Matching
  sampling_steps: 4

model:
  action_dim: 7
  action_seq_len: 8
  mask_ratio: 0.5
  masking_strategy: "modality_aware"
```

---

## Checkpoints

저장 위치: `checkpoints/libero_object/jepa_YYYYMMDD_HHMMSS/`

| File | Description |
|------|-------------|
| `checkpoint_latest.pt` | 최신 체크포인트 |
| `checkpoint_best.pt` | 최저 loss 체크포인트 |
| `checkpoint_epoch_N.pt` | N 에포크 체크포인트 |

### Checkpoint Structure
```python
{
    'epoch': int,
    'model_state_dict': dict,
    'optimizer_state_dict': dict,
    'loss': float,
    'config': dict
}
```

---

## Evaluation

### Offline Evaluation (MSE)
```bash
python eval.py --checkpoint checkpoints/libero_object/jepa_xxx/checkpoint_best.pt
```

### LIBERO Simulation Evaluation
```bash
# Clone LIBERO (if not exists)
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git

# Run simulation evaluation
python run_libero_eval.py \
    --checkpoint checkpoints/libero_object/jepa_xxx/checkpoint_best.pt \
    --task_suite libero_object \
    --num_trials 50
```

---

## Troubleshooting

### 1. Out of Memory
```bash
# Reduce batch size
python train.py --config conf/config_libero_object.yaml --batch_size 64
```

### 2. CUDA Version Mismatch
```bash
# Check CUDA version
nvcc --version

# Install matching PyTorch
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### 3. Mamba-SSM Build Error
```bash
# Build from source
pip install mamba-ssm --no-build-isolation
```

### 4. Missing Language Embeddings
```bash
# Generate embeddings (requires Qwen model)
python scripts/generate_lang_embeddings.py --task_suite libero_object
```

---

## Key Files

| File | Description |
|------|-------------|
| `train.py` | Main training script |
| `eval.py` | Offline evaluation (MSE) |
| `run_libero_eval.py` | LIBERO simulation evaluation |
| `statevla_model.py` | StateVLA + JEPA architecture |
| `action_policy.py` | Flow Matching + Gripper Classifier |
| `dataloader.py` | Dataset loading with normalization |
| `jepa/` | JEPA modules (tokenizer, encoder, predictor, masking) |
| `conf/config_libero_object.yaml` | Training configuration |
