# StateVLA: State-based Vision-Language-Action Model

**Physics-aware Representation Learning via Temporal JEPA and Goal-Conditioned Smooth Action Generation via Flow Matching**

---

## Abstract

StateVLA is a lightweight yet high-performance Vision-Language-Action (VLA) model designed for real-time robotic control. Unlike traditional end-to-end VLA architectures that directly map observations to actions, StateVLA separates **world representation learning** and **policy learning**, enabling more stable training, smoother actions, and significantly faster inference.

The model first learns **physics-aware latent state representations** using Temporal Joint Embedding Predictive Architecture (Temporal JEPA), and then generates continuous robot trajectories using **goal-conditioned Flow Matching**. A lightweight **GoalPredictor** module bridges the two phases by predicting the expected latent state after the action chunk executes, conditioning the policy on both the current state and the goal state. By leveraging pretrained SigLIP vision encoders, CLIP language embeddings, and an efficient **Mamba State Space Model (SSM)** backbone, StateVLA achieves real-time capable inference while maintaining strong manipulation performance.

---

## Method Overview

### Two-Phase Training Strategy

```
Phase 1: Representation Learning
obs_t, action_t  →  latent dynamics learning (Temporal JEPA)

Phase 2: Policy Learning
latent state z_t  →  GoalPredictor → z_goal
[z_t, z_goal]     →  continuous trajectory generation (Flow Matching)
```

This decoupled training stabilizes optimization and ensures the encoder captures physical causality before policy learning begins.

---

## Architecture

### Token Processing Order (Causal Mamba)

```
[Language] → [Robot State] → [Agent View Patches]
           → [Eye-in-Hand Patches] → [CLS]
```

* Language tokens appear first to condition all visual processing
* CLS token is placed at the end so the final position observes the full context
* Total sequence length: ~395 tokens
* Embedding dimension: 256

---

### Full Model Diagram

```
                ┌────────────────────────────┐
                │   Frozen Vision Encoder     │
                │        (SigLIP)             │
                └─────────────┬───────────────┘
                              │
obs_t ──► visual tokens ───────┤
                              │
                ┌─────────────▼───────────────┐
                │   Frozen Language Encoder    │
                │            (CLIP)            │
                └─────────────┬───────────────┘
                              │
                              ▼
                    ┌─────────────────────┐
                    │ Mamba Context Encoder│
                    │   (State Encoder)    │
                    └─────────┬───────────┘
                              │
                     latent state z_t
                              │
        ┌─────────────────────┴─────────────────────┐
        │                                           │
Phase 1 │ Temporal Predictor (JEPA)                 │
        │                                           │
        │ z_t + action_t → predict z'_{t+1}         │
        │                                           │
        └─────────────────────┬─────────────────────┘
                              │
Phase 2                       ▼
                    GoalPredictor (MLP)
                    z_t → z_goal (H steps ahead)
                              │
                    [z_t, z_goal] concat (512D)
                              │
                              ▼
                    Flow Matching Policy
                    (Mamba + Cross-Attn, 7D unified)
                              │
                              ▼
                    Smooth Robot Actions [B, 10, 7]
```

---

## Phase 1 — Temporal JEPA

> "현재 상태 z_t에서 action a_t를 수행하면, 다음 상태 z_{t+1}은 어떻게 될까?"

### Forward Pass

```
obs_t ──→ Tokenizer ──→ Context Encoder (Mamba 12L) ──→ CLS ──→ State Proj ──→ z_t
                                                                                 │
                                                                   ┌─────────────┘
                                                                   ▼
                                                        ┌────────────────────┐
                                                        │ Temporal Predictor  │
                                                        │                    │
                                                        │ z_t ──→ Linear(256→512)
                                                        │ a_t ──→ Linear(7→512)
                                                        │ concat(1024) ──→ MLP ──→ delta(256)
                                                        │                    │
                                                        │ z'_{t+1} = z_t + delta  (residual)
                                                        └──────────┬─────────┘
                                                                   ▼
                                                               z'_{t+1} (predicted)
                                                                   │
                                                              (compare)
                                                                   │
obs_{t+1} ──→ Tokenizer ──→ Target Encoder (EMA, no grad) ──→ z_{t+1} (target)
```

Residual prediction을 사용하는 이유: 작은 action은 작은 state 변화를 일으키므로 delta를 학습하는 것이 절대값 예측보다 안정적이고 효율적.

### Loss Function (Phase 1)

$$\mathcal{L}_{\text{phase1}} = \mathcal{L}_{\text{MSE}} + \lambda_v \cdot \mathcal{L}_{\text{var}} + \lambda_c \cdot \mathcal{L}_{\text{cov}}$$

---

#### 1. MSE Loss (Invariance) — 미래 상태 예측 정확도

$$\mathcal{L}_{\text{MSE}} = \frac{1}{D} \sum_{i=1}^{D} \left( z'_{t+1,i} - \bar{z}_{t+1,i} \right)^2$$

| Symbol | Shape | Description |
|--------|-------|-------------|
| $z'_{t+1}$ | `[B, 256]` | Temporal Predictor가 예측한 다음 상태 |
| $\bar{z}_{t+1}$ | `[B, 256]` | Target Encoder가 인코딩한 실제 다음 상태 (detached, no gradient) |

**핵심 역할**: "현재 상태 + action을 알면, 미래 상태를 예측할 수 있어야 한다"는 물리적 인과성을 학습.

```python
# jepa/temporal_predictor.py:116
mse_loss = F.mse_loss(z_next_pred, z_next_target.detach())
```

---

#### 2. Variance Loss (VICReg) — Representation Collapse 방지

$$\mathcal{L}_{\text{var}} = \frac{1}{D} \sum_{j=1}^{D} \max\left(0,\ 1 - \sigma_j\right)$$

where $\sigma_j = \text{std}(z'_{t+1, :, j})$ is the standard deviation of dimension $j$ across the batch.

**핵심 역할**: 모든 입력이 동일한 representation으로 매핑되는 trivial solution을 방지.

**Weight**: $\lambda_v = 1.0$

---

#### 3. Covariance Loss (VICReg) — 차원 간 독립성 확보

$$\mathcal{L}_{\text{cov}} = \frac{1}{D} \sum_{i \neq j} C_{ij}^2$$

**핵심 역할**: 256개 차원이 각각 다른 정보를 담도록 강제.

**Weight**: $\lambda_c = 0.04$

---

### EMA Target Encoder

```
θ_target = m · θ_target + (1 - m) · θ_context

Cosine momentum schedule:
  m(t) = 1.0 - (1.0 - 0.996) × (1 + cos(π · t/T)) / 2

  t=0:  m ≈ 0.996  (target이 context를 빠르게 추적)
  t=T:  m → 1.0    (target이 거의 고정, 매우 안정적)
```

---

## Phase 2 — Goal-Conditioned Flow Matching

> "현재 상태 z_t에서, H step 후 목표 z_goal을 향해 어떤 action trajectory를 생성해야 할까?"

Encoder와 Temporal Predictor는 Phase 1 가중치로 **완전히 freeze**하고, GoalPredictor와 Flow Matching Policy만 학습.

### GoalPredictor

```
z_t (256D) ──→ MLP (256 → 512 → 512 → 256) ──→ z_goal (256D)
```

GoalPredictor는 현재 latent state에서 action_seq_len (H=10) step 후의 latent state를 예측. 이를 `z_goal`이라 하며, Flow Matching Policy가 "어디로 가야 하는지"를 알 수 있도록 conditioning을 제공.

**Goal Loss:**

$$\mathcal{L}_{\text{goal}} = \text{MSE}\left(z_{\text{goal}},\ \bar{z}_{t+H}\right)$$

where $\bar{z}_{t+H}$ is the EMA target encoder applied to `obs_{t+H}` (the observation H steps ahead, detached).

### Forward Pass

```
obs_t ──→ Frozen Encoder ──→ z_t (256D, no gradient)
                               │
                               ↓
                    GoalPredictor (MLP)
                               │
                           z_goal (256D)
                               │
          concat([z_t, z_goal]) = z_state (512D)
                               │
                               ↓
         ┌──────────────────────────────────────┐
         │  Flow Matching Policy (Mamba 3L)      │
         │                                      │
         │  Input tokens:                       │
         │  [σ_emb(1), state_emb(1),            │
         │   a_noisy × 10]                      │
         │  = 12 tokens                         │
         │                                      │
         │  Mamba 3L backbone                   │
         │     ↓                                │
         │  Cross-Attn to image patches         │  ← spatial_features [B, N, 256]
         │     ↓                                │
         │  velocity [B, 10, 7]                 │
         │  Loss: MSE (all 7 dims unified)      │
         └──────────────────────────────────────┘
```

### Loss Function (Phase 2)

$$\mathcal{L}_{\text{phase2}} = \mathcal{L}_{\text{flow}} + \lambda_{\text{goal}} \cdot \mathcal{L}_{\text{goal}}$$

$$\mathcal{L}_{\text{phase2}} = \mathcal{L}_{\text{flow}} + 0.1 \cdot \mathcal{L}_{\text{goal}}$$

---

#### 1. Flow Matching Loss — 통합 7D Action 생성

모든 7차원 action (pos/rot 6D + gripper 1D)을 단일 Flow Matching으로 처리.

**학습 과정 (Training):**

```python
# Step 1: Diffusion timestep 샘플링
σ ~ Uniform(0, 1)                           # [B]

# Step 2: Gaussian noise 샘플링
ε ~ N(0, I)                                 # [B, 10, 7]

# Step 3: Linear interpolation으로 noisy action 생성
x_noisy = (1 - σ) · x_0 + σ · ε            # x_0 = ground truth actions (normalized) [B, 10, 7]

# Step 4: Target velocity 계산
v_target = ε - x_0

# Step 5: Policy가 velocity 예측
v_pred = FlowMatchingPolicy(σ, z_state, x_noisy, spatial_features)  # [B, 10, 7]
```

**Loss:**

$$\mathcal{L}_{\text{flow}} = \text{MSE}(v_{\text{pred}},\ v_{\text{target}}) = \frac{1}{10 \times 7} \sum \left( v_{\text{pred}} - (\varepsilon - x_0) \right)^2$$

| Symbol | Shape | Description |
|--------|-------|-------------|
| $x_0$ | `[B, 10, 7]` | Ground truth actions (min-max normalized to [-1, 1]) |
| $\varepsilon$ | `[B, 10, 7]` | Sampled Gaussian noise |
| $\sigma$ | `[B]` | Diffusion timestep (0=clean, 1=noise) |
| $z_{\text{state}}$ | `[B, 512]` | concat([z_t, z_goal]) — current + goal state |

---

#### 2. Goal Prediction Loss — GoalPredictor 학습

GoalPredictor가 예측한 z_goal과 실제 H step 후 Target Encoder 출력을 비교.

$$\mathcal{L}_{\text{goal}} = \text{MSE}\left(z_{\text{goal}},\ \bar{z}_{t+H}\right)$$

**Weight**: $\lambda_{\text{goal}} = 0.1$

GoalPredictor를 통해 policy는 단순히 현재 상태를 모방하는 것을 넘어, 미래 목표 상태를 향해 행동하도록 유도됨.

---

#### Action Normalization

모든 7차원 action을 min-max scaling으로 [-1, 1]로 정규화:

```python
# 정규화 (dataloader)
normalized = (action - action_min) / (action_max - action_min) * 2.0 - 1.0

# 역정규화 (inference)
denormalized = (action + 1.0) / 2.0 * (action_max - action_min) + action_min
```

기존 z-score 대비 장점: 액션이 항상 [-1, 1] 범위에 있어 flow matching의 초기 noise scale과 잘 맞음.

---

#### Inference (Action Generation)

```python
# Pure noise에서 시작
x = N(0, I)                                 # [B, 10, 7]

# GoalPredictor: 목표 latent 예측
z_goal = GoalPredictor(z_t)                 # [B, 256]
z_state = concat([z_t, z_goal], dim=-1)    # [B, 512]

for t in [1.0, 0.75, 0.5, 0.25]:            # 4 denoising steps
    v = FlowMatchingPolicy(t, z_state, x, spatial_features)
    x = x - (1/4) · v                       # Euler step

# Min-max 역정규화
action = denormalize(x)                     # [B, 10, 7]
```

| Step | $t$ | State of $x$ |
|------|-----|-------------|
| 0 | - | Pure Gaussian noise |
| 1 | 1.00 | Rough structure emerges |
| 2 | 0.75 | Trajectory shape forms |
| 3 | 0.50 | Fine details appear |
| 4 | 0.25 | Clean action output |

---

## Model Components

| Component          | Model                  | Parameters | Trainable |
| ------------------ | ---------------------- | ---------- | --------- |
| Vision Backbone    | SigLIP ViT-B/16        | ~86M       | Frozen    |
| Language Backbone  | CLIP ViT-B/32          | ~87M       | Frozen    |
| Context Encoder    | Mamba SSM (12L)        | ~12.3M     | Phase 1   |
| Temporal Predictor | MLP                    | ~1.1M      | Phase 1   |
| GoalPredictor      | MLP (256→512→512→256)  | ~0.5M      | Phase 2   |
| Flow Policy        | Mamba SSM (3L) + CrossAttn | ~1.9M  | Phase 2   |
| Total Trainable    | —                      | ~15.8M     | —         |

---

## Why Not Generative VLMs?

Large generative VLMs (e.g., RT-2, Qwen-VL) are powerful planners but inefficient for servo-level motor control due to:

* Low inference frequency (1–5 Hz)
* Discrete token outputs
* High computational cost

StateVLA instead functions as a **low-level controller**, enabling:

* > 100 Hz inference
* Continuous trajectory generation
* Lightweight deployment

Future systems can combine:

```
High-level Planner (VLM)
          ↓
StateVLA Controller
          ↓
Robot
```

---

## Installation

```bash
git clone https://github.com/Choi-In-hyuk/StateVLA.git
cd StateVLA

conda create -n statevla python=3.10
conda activate statevla

pip install torch torchvision
pip install mamba-ssm causal-conv1d
pip install transformers numpy einops tqdm imageio h5py
```

Install LIBERO (optional):

```bash
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git
cd LIBERO
pip install -e .
cd ..
```

---

## Training

### Phase 1 — Temporal JEPA

Single GPU:
```bash
CUDA_VISIBLE_DEVICES=1 python train.py --config conf/config.yaml --phase 1 --device cuda:0
```

Multi-GPU (DDP):
```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 train.py \
    --config conf/config.yaml \
    --phase 1 \
    --batch_size 128
```

### Phase 2 — Goal-Conditioned Flow Matching

```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 train.py \
    --config conf/config.yaml \
    --phase 2 \
    --batch_size 128 \
    --phase1_checkpoint checkpoints/phase1_XXXXXXXX_XXXXXX/checkpoint_best.pt
```

### Resume Training

```bash
# Resume phase 2 (with optimizer reset when changing LR)
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 train.py \
    --config conf/config.yaml \
    --phase 2 \
    --batch_size 128 \
    --phase1_checkpoint checkpoints/phase1_XXXXXXXX_XXXXXX/checkpoint_best.pt \
    --checkpoint checkpoints/phase2_XXXXXXXX_XXXXXX/checkpoint_latest.pt \
    --reset_optimizer
```

---

## Evaluation

LIBERO simulation:

```bash
python run_libero_eval.py \
    --checkpoint checkpoints/phase2_XXXXXXXX_XXXXXX/checkpoint_best.pt \
    --task_suite libero_object \
    --num_trials 50
```

---

## Citation

```
@article{statevla2025,
  title={StateVLA: State-based Vision-Language-Action Model},
  author={Choi, In-hyuk},
  year={2025}
}
```

---

## License

MIT License
