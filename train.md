# StateVLA Two-Phase Training Guide

## Architecture Overview

### Token Sequence (Mamba Causal Order)

```
[Lang(1)] → [Robot(1)] → [Agentview(196)] → [Eye-in-hand(196)] → [CLS(1)]

  Mamba hidden state flow:
  ①  Language  → "pick up the butter" 인코딩
  ②  Robot     → "팔이 어디에 있는지" 추가
  ③  Vision    → lang+robot context를 가진 채로 이미지 특징 추출
  ④  CLS       → 전체 통합 representation z_t (256 dim)
```

### Phase 1: Temporal JEPA (표현 학습)

```
┌─────────────────────────────────────────────────────────────────────┐
│                         Phase 1: Temporal JEPA                      │
│                                                                     │
│  "현재 상황(z_t)에서 이 액션(a_t)을 하면 미래가 어떻게 될까?"       │
│                                                                     │
│  obs_t ──→ Tokenizer ──→ Mamba Encoder (12L) ──→ z_t (256D)        │
│                                                    │                │
│                                      ┌─────────────┘                │
│                                      ↓                              │
│                           ┌──────────────────────┐                  │
│                           │  Temporal Predictor   │                  │
│                           │  z_t(256) ──→ proj(512)                 │
│                           │  a_t(7)  ──→ proj(512)                  │
│                           │  concat(1024) ──→ MLP ──→ delta(256)    │
│                           │  z'_{t+1} = z_t + delta (residual)      │
│                           └──────────┬───────────┘                  │
│                                      ↓                              │
│                                  z'_{t+1} ───── MSE Loss            │
│                                      ↑              ↓               │
│                                  (compare)     + VICReg             │
│                                      ↓         (variance +         │
│  obs_{t+1} ──→ Target Encoder (EMA) ──→ z_{t+1}   covariance)     │
│                                                                     │
│  Trainable: Context Encoder + Temporal Predictor                    │
│  EMA:       Target Encoder (cosine momentum 0.996 → 1.0)           │
└─────────────────────────────────────────────────────────────────────┘
```

### Phase 2: Flow Matching (정책 학습)

```
┌─────────────────────────────────────────────────────────────────────┐
│                      Phase 2: Flow Matching                         │
│                                                                     │
│  obs_t ──→ Frozen Encoder ──→ z_t (256D, no gradient)              │
│                                 │                                   │
│                    ┌────────────┴────────────┐                      │
│                    ↓                         ↓                      │
│          ┌─────────────────┐      ┌──────────────────┐             │
│          │  Flow Matching  │      │ Gripper Classifier│             │
│          │  (Mamba 3L)     │      │ (MLP 3L)         │             │
│          │                 │      │                   │             │
│          │  Input seq:     │      │ z_t → 256 → 256  │             │
│          │  [σ, z_t,       │      │   → 10 logits    │             │
│          │   a_noisy×10]   │      │                   │             │
│          │     ↓           │      │ Loss: BCE         │             │
│          │  Mamba backbone │      │ {-1,1} → {0,1}   │             │
│          │     ↓           │      └────────┬─────────┘             │
│          │  velocity (6D)  │               │                        │
│          │                 │               │                        │
│          │  Loss: MSE      │               │                        │
│          │  (v, noise-x_0) │               │                        │
│          └────────┬────────┘               │                        │
│                   │                        │                        │
│                   └──────┬─────────────────┘                        │
│                          ↓                                          │
│                [pos_rot(6), gripper(1)] × 10 steps                  │
│                                                                     │
│  Trainable: Flow Matching Policy + Gripper Classifier               │
│  Frozen:    Encoder (from Phase 1)                                  │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Model Components

| Component | Model | Params | Description |
|-----------|-------|--------|-------------|
| Image Tokenizer | Conv2d (16×16 patches) | ~0.6M | 224×224 → 196 patches × 256D (per camera) |
| Language Tokenizer | Linear (512 → 256) | ~131K | Pre-computed embedding → 1 token |
| Robot State Tokenizer | MLP (9 → 256) | ~68K | joint(7) + gripper(2) → 1 token |
| Context Encoder | Mamba SSM × 12 layers | ~12.3M | d_model=256, d_state=16, d_conv=4 |
| Target Encoder | EMA copy | shared | No gradient, momentum update |
| Temporal Predictor | MLP (1024 → 512 → 512 → 256) | ~1.1M | Residual: z_{t+1} = z_t + delta |
| Flow Matching Policy | Mamba SSM × 3 layers | ~1.8M | d_model=256, denoising |
| Gripper Classifier | MLP (256 → 256 → 10) | ~132K | Binary per timestep |
| **Total** | | **~14.9M** | |

---

## Data Format

### LIBERO-Object Dataset

- **Location**: `/path/to/libero_object/`
- **Tasks**: 10 pick-and-place tasks
- **Demos**: 50 per task (500 total)
- **Total samples**: ~70,000

### Observation (Encoder Input)

| Field | Shape | Description |
|-------|-------|-------------|
| `agentview_rgb` | [T, 224, 224, 3] | Third-person camera |
| `eye_in_hand_rgb` | [T, 224, 224, 3] | Wrist camera |
| `joint_states` | [T, 7] | Joint positions |
| `gripper_states` | [T, 2] | Left/right finger positions (continuous) |

→ `robot_state = concat(joint_states, gripper_states)` → **9D**

### Action (Policy Output)

| Dim | Range | Description | Training |
|-----|-------|-------------|----------|
| 0-5 | continuous | Position + Rotation (6D) | Flow Matching (normalized) |
| 6 | {-1, +1} | Gripper open/close (binary) | BCE classifier (not normalized) |

---

## Training Commands

### Phase 1: Temporal JEPA

```bash
python train.py --config conf/config.yaml --phase 1
```

**What it learns**: 현재 상태 z_t에서 action a_t를 수행하면 다음 상태 z_{t+1}이 어떻게 되는지 예측

**Monitoring**:
- `jepa_mse`: 예측 정확도 (낮을수록 좋음)
- `jepa_variance`: representation collapse 방지 (0에 가까울수록 좋음)
- `jepa_covariance`: 차원 간 독립성 (0에 가까울수록 좋음)

**완료 기준**: jepa_mse가 수렴하고, variance가 0 근처로 안정화

### Phase 2: Flow Matching

```bash
python train.py --config conf/config.yaml --phase 2 \
    --phase1_checkpoint checkpoints/phase1_temporal_jepa/checkpoint_best.pt
```

**What it learns**: 부드러운 연속 action 생성 (encoder는 frozen)

**Monitoring**:
- `pos_rot_loss`: position/rotation Flow Matching loss
- `gripper_loss`: gripper BCE loss
- `action_loss`: total = pos_rot + gripper

### Resume Training

```bash
# Phase 1 resume
python train.py --config conf/config.yaml --phase 1 \
    --checkpoint checkpoints/phase1_temporal_jepa/checkpoint_latest.pt

# Phase 2 resume
python train.py --config conf/config.yaml --phase 2 \
    --checkpoint checkpoints/phase2_flow_matching/checkpoint_latest.pt
```

### Multi-GPU (DDP)

```bash
torchrun --nproc_per_node=2 train.py --config conf/config.yaml --phase 1
torchrun --nproc_per_node=4 train.py --config conf/config.yaml --phase 1
```

---

## Configuration

### `conf/config.yaml`

```yaml
data:
  data_directory: "/path/to/libero_object"
  demos_per_task: 50
  max_len_data: 260
  train_split: 0.9

model:
  image_size: 224
  patch_size: 16                # 14×14 = 196 patches per image
  embed_dim: 256                # Token embedding dimension
  lang_emb_dim: 512             # Language embedding input dim
  robot_state_dim: 9            # joint(7) + gripper(2)
  encoder_depth: 12             # Mamba encoder layers
  d_state: 16                   # Mamba state dimension
  d_conv: 4                     # Mamba convolution width
  expand: 2                     # Mamba expansion factor
  state_dim: 256                # State representation z_t dim
  action_dim: 7                 # 6 pos/rot + 1 gripper
  action_seq_len: 10            # Action chunk length
  policy_layers: 3              # Flow Matching Mamba layers
  policy_embed_dim: 256         # Flow Matching hidden dim

training:
  batch_size: 64
  learning_rate: 1.0e-4
  weight_decay: 0.05
  gradient_clip: 1.0
  ema_momentum: 0.996
  ema_momentum_schedule: "cosine"
  save_interval: 100
  log_interval: 10

  phase1:
    num_epochs: 1000
    learning_rate: 1.0e-4
    temporal_predictor_hidden_dim: 512

  phase2:
    num_epochs: 1000
    learning_rate: 5.0e-5
```

---

## GPU Memory Guide

| Batch Size | GPU Memory (approx) |
|------------|---------------------|
| 256 | ~40GB |
| 128 | ~24GB |
| 64 | ~16GB |
| 32 | ~10GB |

---

## Checkpoints

저장 위치: `checkpoints/phase{1,2}_{name}/`

| File | Description |
|------|-------------|
| `checkpoint_latest.pt` | 최신 체크포인트 |
| `checkpoint_best.pt` | 최저 loss 체크포인트 |
| `checkpoint_epoch_N.pt` | N 에포크 체크포인트 (save_interval마다) |

### Checkpoint Structure

```python
{
    'epoch': int,
    'model_state_dict': dict,     # StateVLATrainer state
    'optimizer_state_dict': dict,
    'loss': float,
    'config': dict,
    'phase': int,                 # 1 or 2
}
```

---

## Evaluation

### Offline (MSE)

```bash
python eval.py --checkpoint checkpoints/phase2_flow_matching/checkpoint_best.pt
```

출력: per-dimension MSE (x, y, z, rx, ry, rz, gripper)

### LIBERO Simulation

```bash
python run_libero_eval.py \
    --checkpoint checkpoints/phase2_flow_matching/checkpoint_best.pt \
    --task_suite libero_object \
    --num_trials 50
```

---

## Troubleshooting

### Out of Memory

```bash
python train.py --config conf/config.yaml --phase 1 --batch_size 32
```

### Mamba-SSM Build Error

```bash
pip install mamba-ssm --no-build-isolation
```

### Representation Collapse (Phase 1)

`jepa_variance` loss가 계속 높으면 → representation이 collapse되고 있음
- VICReg weights 확인: `variance_weight: 1.0`, `covariance_weight: 0.04`
- CLS 토큰이 시퀀스 끝에 있는지 확인 (Mamba causal)

### Phase 2 Loss Not Decreasing

- Phase 1 체크포인트가 제대로 로드되었는지 확인
- Encoder가 frozen인지 확인 (학습 로그에 parameter count)

---

## Key Files

| File | Description |
|------|-------------|
| `train.py` | Training script (`--phase 1` / `--phase 2`) |
| `eval.py` | Offline evaluation (MSE) |
| `run_libero_eval.py` | LIBERO simulation evaluation |
| `statevla_model.py` | StateVLA + StateVLATrainer (two-phase routing) |
| `state_encoder.py` | JEPAStateEncoder (tokenizer + encoder + temporal predictor) |
| `action_policy.py` | FlowMatchingPolicy + GripperClassifier |
| `jepa/tokenizer.py` | Multi-modal tokenizer (Lang → Robot → Vision → CLS) |
| `jepa/encoder.py` | Context Encoder (Mamba) + Target Encoder (EMA) |
| `jepa/temporal_predictor.py` | z_t + a_t → z'_{t+1} + VICReg loss |
| `dataloader.py` | Dataset loading + action normalization (pos/rot only) |
| `conf/config.yaml` | Training configuration |
