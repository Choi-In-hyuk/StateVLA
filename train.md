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

### Phase 2: GoalPredictor + Flow Matching (정책 학습)

```
┌─────────────────────────────────────────────────────────────────────┐
│              Phase 2: Goal-Conditioned Flow Matching                 │
│                                                                     │
│  obs_t ──→ Frozen Encoder ──→ z_t (256D, no gradient)              │
│                                 │                                   │
│                                 ↓                                   │
│                      ┌──────────────────┐                           │
│                      │  GoalPredictor    │                           │
│                      │  MLP(512 hidden)  │                           │
│                      │  z_t → z_goal     │ ← "H step 후 상태 예측"  │
│                      └────────┬─────────┘                           │
│                               │                                     │
│                 concat([z_t, z_goal]) = 512D                        │
│                               │                                     │
│                               ↓                                     │
│              ┌────────────────────────────┐                         │
│              │  Flow Matching Policy       │                         │
│              │  (Mamba 3L + Cross-Attn)    │                         │
│              │                             │                         │
│              │  Input seq:                 │  ← spatial_features     │
│              │  [σ_emb(1), state_emb(1),   │     (image patches)     │
│              │   a_noisy(10)]              │                         │
│              │     ↓                       │                         │
│              │  Mamba backbone (3L)        │                         │
│              │     ↓                       │                         │
│              │  Cross-Attn to patches      │                         │
│              │     ↓                       │                         │
│              │  velocity [B, 10, 7]        │                         │
│              │  Loss: MSE (all 7 dims)     │                         │
│              └────────────────────────────┘                         │
│                               │                                     │
│  Goal Loss (weight=0.1):      │                                     │
│    z_goal_target = TargetEncoder(obs_{t+H})                         │
│    L_goal = MSE(z_goal, z_goal_target.detach())                     │
│                                                                     │
│  Total Loss: L_flow + 0.1 × L_goal                                  │
│                                                                     │
│  Trainable: GoalPredictor + Flow Matching Policy                    │
│  Frozen:    Encoder + Temporal Predictor (from Phase 1)             │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Model Components

| Component | Model | Params | Description |
|-----------|-------|--------|-------------|
| Image Tokenizer | SigLIP ViT-B/16 (frozen) | ~86M | 224×224 → 196 patches × 256D (per camera) |
| Language Tokenizer | CLIP ViT-B/32 (frozen) | ~87M | Pre-computed embedding → 1 token × 256D |
| Robot State Tokenizer | MLP (9 → 256) | ~68K | joint(7) + gripper(2) → 1 token |
| Context Encoder | Mamba SSM × 12 layers | ~12.3M | d_model=256, d_state=16, d_conv=4 |
| Target Encoder | EMA copy | shared | No gradient, momentum update |
| Temporal Predictor | MLP (1024 → 512 → 512 → 256) | ~1.1M | Residual: z_{t+1} = z_t + delta |
| GoalPredictor | MLP (256 → 512 → 512 → 256) | ~0.5M | z_t → z_goal (H step ahead in latent space) |
| Flow Matching Policy | Mamba 3L + Cross-Attn | ~1.9M | Conditioned on [z_t, z_goal], all 7D |
| **Total Trainable** | | **~14.9M** | |

---

## Data Format

### LIBERO-Object Dataset

- **Location**: `/home/choi/libero_object_temp/`
- **Language Embeddings**: `/home/choi/StateVLA/data/libero/language_embeddings/libero_object.pkl`
- **Tasks**: 10 pick-and-place tasks
- **Demos**: 50 per task (500 total)
- **Training**: 전체 데이터 사용 (val split 없음)

### Observation (Encoder Input)

| Field | Shape | Description |
|-------|-------|-------------|
| `agentview_rgb` | [T, 224, 224, 3] | Third-person camera |
| `eye_in_hand_rgb` | [T, 224, 224, 3] | Wrist camera |
| `joint_states` | [T, 7] | Joint positions |
| `gripper_states` | [T, 2] | Left/right finger positions (continuous) |
| `lang_emb` | [1, 512] | Pre-computed CLIP task embedding |

→ `robot_state = concat(joint_states, gripper_states)` → **9D**

### Action (Policy Output)

| Dim | Range | Description | Normalization |
|-----|-------|-------------|---------------|
| 0-5 | [-1, 1] | Position + Rotation (6D) | Min-max → [-1, 1] |
| 6 | [-1, 1] | Gripper open/close | Min-max → [-1, 1] (unified flow matching) |

**Action Normalization**: min-max scaling to [-1, 1] for all 7 dimensions.

```python
# dataloader.py
normalized = (action - action_min) / (action_max - action_min) * 2.0 - 1.0
```

---

## Training Commands

### Phase 1: Temporal JEPA

Single GPU:
```bash
CUDA_VISIBLE_DEVICES=1 python /home/choi/StateVLA/train.py \
  --config /home/choi/StateVLA/conf/config.yaml \
  --phase 1 \
  --device cuda:0
```

Multi-GPU (DDP, 2x GPU, batch 128/GPU = effective 256):
```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 /home/choi/StateVLA/train.py \
  --config /home/choi/StateVLA/conf/config.yaml \
  --phase 1 \
  --batch_size 128 \
  2>&1 | tee /home/choi/StateVLA/phase1_train_ddp.log
```

**Monitoring**:
- `jepa_mse`: 예측 정확도 (낮을수록 좋음)
- `jepa_variance`: representation collapse 방지 (0에 가까울수록 좋음)
- `jepa_covariance`: 차원 간 독립성 (0에 가까울수록 좋음)

**완료 기준**: jepa_mse가 수렴하고, variance가 0 근처로 안정화

### Phase 2: Goal-Conditioned Flow Matching

```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 /home/choi/StateVLA/train.py \
  --config /home/choi/StateVLA/conf/config.yaml \
  --phase 2 \
  --batch_size 128 \
  --phase1_checkpoint checkpoints/phase1_XXXXXXXX_XXXXXX/checkpoint_best.pt \
  2>&1 | tee /home/choi/StateVLA/phase2_train_ddp.log
```

**Monitoring**:
- `action`: flow matching loss (all 7 dims, 낮을수록 좋음)
- `goal`: goal predictor loss (GoalPredictor가 얼마나 정확하게 목표 latent를 예측하는지)

### Resume Training

```bash
# Phase 1 resume
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 /home/choi/StateVLA/train.py \
  --config /home/choi/StateVLA/conf/config.yaml \
  --phase 1 \
  --batch_size 128 \
  --checkpoint checkpoints/phase1_XXXXXXXX_XXXXXX/checkpoint_latest.pt \
  2>&1 | tee /home/choi/StateVLA/phase1_resume_ddp.log

# Phase 2 resume
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 /home/choi/StateVLA/train.py \
  --config /home/choi/StateVLA/conf/config.yaml \
  --phase 2 \
  --batch_size 128 \
  --phase1_checkpoint checkpoints/phase1_XXXXXXXX_XXXXXX/checkpoint_best.pt \
  --checkpoint checkpoints/phase2_XXXXXXXX_XXXXXX/checkpoint_latest.pt \
  2>&1 | tee /home/choi/StateVLA/phase2_resume_ddp.log

# Phase 2 resume with LR reset (when changing LR or optimizer state is stale)
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 /home/choi/StateVLA/train.py \
  --config /home/choi/StateVLA/conf/config.yaml \
  --phase 2 \
  --batch_size 128 \
  --phase1_checkpoint checkpoints/phase1_XXXXXXXX_XXXXXX/checkpoint_best.pt \
  --checkpoint checkpoints/phase2_XXXXXXXX_XXXXXX/checkpoint_latest.pt \
  --reset_optimizer \
  2>&1 | tee /home/choi/StateVLA/phase2_resume_reset_ddp.log
```

---

## Configuration (`conf/config.yaml`)

```yaml
data:
  data_directory: "/home/choi/libero_object_temp"
  language_embedding_path: "/home/choi/StateVLA/data/libero/language_embeddings/libero_object.pkl"
  demos_per_task: 50
  max_len_data: 260

model:
  image_size: 224
  patch_size: 16                # 14×14 = 196 patches per image
  embed_dim: 256                # Token embedding dimension
  use_pretrained_vision: true   # SigLIP backbone (frozen)
  use_pretrained_language: true # CLIP backbone (frozen)
  vision_model_name: "google/siglip-base-patch16-224"
  language_model_name: "ViT-B/32"
  lang_emb_dim: 512             # Language embedding input dim (CLIP output)
  robot_state_dim: 9            # joint(7) + gripper(2)
  encoder_depth: 12             # Mamba encoder layers
  d_state: 16                   # Mamba state dimension
  d_conv: 4                     # Mamba convolution width
  expand: 2                     # Mamba expansion factor
  state_dim: 256                # State representation z_t dim
  action_dim: 7                 # pos(3) + rot(3) + gripper(1), unified 7D
  action_seq_len: 10            # Action chunk length
  goal_predictor_hidden_dim: 512  # GoalPredictor hidden dim
  policy_layers: 3              # Flow Matching Mamba layers
  policy_embed_dim: 256         # Flow Matching hidden dim

training:
  batch_size: 128
  learning_rate: 1.0e-4
  weight_decay: 0.05
  gradient_clip: 1.0
  ema_momentum: 0.996
  ema_momentum_schedule: "cosine"
  world_model_loss_weight: 0.0  # Disabled (replaced by GoalPredictor)
  goal_loss_weight: 0.1         # Phase 2: GoalPredictor target loss weight
  use_lr_scheduler: true        # CosineAnnealingLR
  min_lr: 5.0e-6
  val_interval: 1               # Validate every epoch
  save_interval: 200            # epoch checkpoint 저장 간격

  phase1:
    num_epochs: 1000
    learning_rate: 1.0e-4
    temporal_predictor_hidden_dim: 512

  phase2:
    num_epochs: 3000
    learning_rate: 1.0e-5
    scheduler_epochs: 3000      # CosineAnnealingLR T_max
```

---

## GPU Memory Guide

| Batch Size (per GPU) | GPU Memory (approx) |
|----------------------|---------------------|
| 256 | ~40GB |
| 128 | ~24GB |
| 64 | ~16GB |
| 32 | ~10GB |

---

## Checkpoints

저장 위치: `checkpoints/phase{1,2}_{timestamp}/`

| File | Description |
|------|-------------|
| `checkpoint_latest.pt` | 매 epoch 덮어씌우는 최신 체크포인트 |
| `checkpoint_best.pt` | train_loss 기준 최저 체크포인트 |
| `checkpoint_epoch_N.pt` | N epoch 체크포인트 (save_interval=200마다) |

### Checkpoint Structure

```python
{
    'epoch': int,
    'model_state_dict': dict,       # StateVLATrainer state
    'optimizer_state_dict': dict,
    'scheduler_state_dict': dict,   # CosineAnnealingLR state (resume 시 LR 복원)
    'loss': float,                  # 해당 epoch train loss
    'best_train_loss': float,       # 지금까지 최저 train loss (best 판단 기준)
    'config': dict,
}
```

---

## Evaluation

### LIBERO Simulation

```bash
python run_libero_eval.py \
    --checkpoint checkpoints/phase2_XXXXXXXX_XXXXXX/checkpoint_best.pt \
    --task_suite libero_object \
    --num_trials 50
```

---

## Troubleshooting

### Out of Memory

```bash
# DDP: batch_size를 GPU당 64로 줄이기
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 train.py \
  --config conf/config.yaml --phase 1 --batch_size 64
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
- Encoder가 frozen인지 확인 (학습 로그에 trainable parameter count)
- LR이 너무 크면 val loss가 발산할 수 있음 → `phase2.learning_rate: 1.0e-5`

### Phase 2 Val Loss Increasing After Resume

Adam optimizer state (v_t)가 이전 LR 기준으로 calibrated되어 있어 LR 변경 시 instability 발생.

```bash
# --reset_optimizer 플래그로 optimizer state 초기화
torchrun ... --reset_optimizer
```

---

## Key Files

| File | Description |
|------|-------------|
| `train.py` | Training script (`--phase 1` / `--phase 2`), DDP 지원, `--reset_optimizer` |
| `run_libero_eval.py` | LIBERO simulation evaluation |
| `statevla_model.py` | StateVLA + StateVLATrainer (two-phase routing) + GoalPredictor |
| `state_encoder.py` | JEPAStateEncoder (tokenizer + encoder + temporal predictor) |
| `action_policy.py` | FlowMatchingPolicy (goal-conditioned, all 7D) |
| `jepa/tokenizer.py` | Multi-modal tokenizer (SigLIP + CLIP + Robot → Mamba) |
| `jepa/encoder.py` | Context Encoder (Mamba) + Target Encoder (EMA) |
| `jepa/temporal_predictor.py` | z_t + a_t → z'_{t+1} + VICReg loss |
| `dataloader.py` | Dataset loading + min-max action normalization (all 7D) |
| `conf/config.yaml` | Training configuration |
