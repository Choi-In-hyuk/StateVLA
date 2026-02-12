# StateVLA: State-based Vision-Language-Action Model

**Physics-aware Representation Learning via Temporal JEPA and Smooth Action Generation via Flow Matching**

---

## Abstract

StateVLA is a lightweight yet high-performance Vision-Language-Action (VLA) model designed for real-time robotic control. Unlike traditional end-to-end VLA architectures that directly map observations to actions, StateVLA separates **world representation learning** and **policy learning**, enabling more stable training, smoother actions, and significantly faster inference.

The model first learns **physics-aware latent state representations** using Temporal Joint Embedding Predictive Architecture (Temporal JEPA), and then generates continuous robot trajectories using **Flow Matching** conditioned on the learned latent state. By leveraging pretrained SigLIP vision encoders, CLIP language embeddings, and an efficient **Mamba State Space Model (SSM)** backbone, StateVLA achieves real-time capable inference while maintaining strong manipulation performance.

---

## Method Overview

### Two-Phase Training Strategy

```
Phase 1: Representation Learning
obs_t, action_t  →  latent dynamics learning (Temporal JEPA)

Phase 2: Policy Learning
latent state z_t → continuous trajectory generation (Flow Matching)
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
                    Flow Matching Policy
                      (Trajectory Model)
                              │
                              ▼
                    Smooth Robot Actions
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

**핵심 역할**: "현재 상태 + action을 알면, 미래 상태를 예측할 수 있어야 한다"는 물리적 인과성을 학습. 이 loss가 메인 학습 신호.

```python
# jepa/temporal_predictor.py:116
mse_loss = F.mse_loss(z_next_pred, z_next_target.detach())
```

---

#### 2. Variance Loss (VICReg) — Representation Collapse 방지

$$\mathcal{L}_{\text{var}} = \frac{1}{D} \sum_{j=1}^{D} \max\left(0,\ 1 - \sigma_j\right)$$

where $\sigma_j = \text{std}(z'_{t+1, :, j})$ is the standard deviation of dimension $j$ across the batch.

| Condition | Meaning |
|-----------|---------|
| $\sigma_j \geq 1$ | Dimension $j$ has enough spread → loss = 0 (good) |
| $\sigma_j < 1$ | Dimension $j$ is collapsing → penalized by $1 - \sigma_j$ (bad) |
| All $\sigma_j = 0$ | Complete collapse: all samples map to same point |

**핵심 역할**: 모든 입력이 동일한 representation으로 매핑되는 trivial solution을 방지. 이 loss 없이는 encoder가 상수 함수를 학습해서 MSE = 0이 되는 shortcut을 찾을 수 있음.

```python
# jepa/temporal_predictor.py:119-120
std_pred = z_next_pred.std(dim=0)          # [D] — 배치 내 각 차원의 표준편차
var_loss = F.relu(1.0 - std_pred).mean()   # std < 1인 차원에 penalty
```

**Weight**: $\lambda_v = 1.0$

**모니터링**: `jepa_variance` → 0에 가까울수록 collapse 없이 잘 학습되고 있음

---

#### 3. Covariance Loss (VICReg) — 차원 간 독립성 확보

$$\mathcal{L}_{\text{cov}} = \frac{1}{D} \sum_{i \neq j} C_{ij}^2$$

where $C$ is the covariance matrix:

$$C = \frac{1}{B-1} \hat{z}'^{\,T} \hat{z}', \quad \hat{z}' = z' - \text{mean}(z', \text{dim}=0)$$

| Condition | Meaning |
|-----------|---------|
| $C_{ij} = 0\ (i \neq j)$ | Dimension $i$와 $j$는 독립적 → loss = 0 (good) |
| $C_{ij} \neq 0\ (i \neq j)$ | Dimension $i$와 $j$가 상관관계 → penalized (bad) |
| Diagonal $C_{ii}$ | 무시 (각 차원의 분산, variance loss가 관리) |

**핵심 역할**: 256개 차원이 각각 다른 정보를 담도록 강제. 이 loss 없이는 여러 차원이 동일한 feature를 중복 인코딩하여 representation capacity가 낭비됨.

```python
# jepa/temporal_predictor.py:122-126
pred_centered = z_next_pred - z_next_pred.mean(dim=0, keepdim=True)
cov = (pred_centered.T @ pred_centered) / (B - 1 + 1e-8)   # [D, D] 공분산 행렬
off_diag = cov - torch.diag(torch.diag(cov))                # 대각선 제거
cov_loss = (off_diag ** 2).sum() / D                         # 비대각 원소의 제곱합
```

**Weight**: $\lambda_c = 0.04$ (MSE 대비 낮은 가중치 — 보조 정규화 역할)

**모니터링**: `jepa_covariance` → 0에 가까울수록 차원 간 redundancy가 없음

---

#### Why VICReg? (Variance + Covariance)

Self-supervised latent prediction은 **representation collapse** 위험이 있음:

```
Without VICReg:
  encoder → 모든 입력을 동일 벡터로 매핑 → MSE = 0 (trivial solution)
  결과: 아무것도 학습하지 않음

With VICReg:
  Variance  → 각 차원이 충분히 퍼져야 함 (collapse 방지)
  Covariance → 각 차원이 서로 다른 정보를 담아야 함 (redundancy 방지)
  결과: 256D 공간을 최대한 활용하는 의미 있는 representation 학습
```

Contrastive learning (SimCLR 등)과 달리 negative pair가 필요 없어, 배치 내 모든 샘플을 활용하여 더 안정적으로 학습.

### EMA Target Encoder

```
θ_target = m · θ_target + (1 - m) · θ_context

Cosine momentum schedule:
  m(t) = 1.0 - (1.0 - 0.996) × (1 + cos(π · t/T)) / 2

  t=0:  m ≈ 0.996  (target이 context를 빠르게 추적)
  t=T:  m → 1.0    (target이 거의 고정, 매우 안정적)
```

| Property | Description |
|----------|-------------|
| Gradient | Target encoder는 gradient를 **절대 받지 않음** |
| Update | Context encoder의 EMA copy로만 업데이트 |
| Early training | 낮은 momentum → target이 context를 빠르게 따라감 |
| Late training | 높은 momentum → target이 거의 고정되어 안정적 학습 target 제공 |

---

## Phase 2 — Flow Matching Policy

> "주어진 상태 z_t에서, 어떤 action trajectory를 생성해야 할까?"

Encoder는 Phase 1에서 학습된 가중치로 **완전히 freeze**하고, action 생성 모듈만 학습.

### Forward Pass

```
obs_t ──→ Frozen Encoder ──→ z_t (256D, no gradient)
                               │
                  ┌────────────┴────────────┐
                  ▼                         ▼
       ┌─────────────────┐      ┌──────────────────┐
       │  Flow Matching   │      │ Gripper Classifier│
       │  (Mamba 3L)      │      │ (MLP)            │
       │                  │      │                   │
       │  Input tokens:   │      │ z_t               │
       │  [σ_emb(1),      │      │   → Linear(256)   │
       │   z_emb(1),      │      │   → SiLU           │
       │   a_noisy(10)]   │      │   → Linear(256)   │
       │  = 12 tokens     │      │   → SiLU           │
       │                  │      │   → Linear(10)    │
       │  Mamba 3L        │      │  = 10 logits      │
       │  → MLP head      │      │                   │
       │  → velocity(6D)  │      │ Loss: BCE          │
       │                  │      │                   │
       │  Loss: MSE       │      └────────┬─────────┘
       └────────┬─────────┘               │
                └──────┬──────────────────┘
                       ▼
             [pos_rot(6D), gripper(1D)] × 10 steps = action [B, 10, 7]
```

### Loss Function (Phase 2)

$$\mathcal{L}_{\text{phase2}} = \mathcal{L}_{\text{flow}} + \mathcal{L}_{\text{gripper}}$$

---

#### 1. Flow Matching Loss — Position/Rotation (6D) 연속 action 생성

Flow Matching은 noise에서 시작하여 ground truth action으로 향하는 **velocity field**를 학습.

**학습 과정 (Training):**

```python
# Step 1: Diffusion timestep 샘플링
σ ~ Uniform(0, 1)                           # [B]

# Step 2: Gaussian noise 샘플링
ε ~ N(0, I)                                 # [B, 10, 6]

# Step 3: Linear interpolation으로 noisy action 생성
x_noisy = (1 - σ) · x_0 + σ · ε            # x_0 = ground truth pos/rot [B, 10, 6]
#  σ=0 → x_noisy = x_0 (clean)
#  σ=1 → x_noisy = ε   (pure noise)

# Step 4: Target velocity 계산 (optimal transport 방향)
v_target = ε - x_0                          # noise에서 data로 향하는 방향의 반대

# Step 5: Mamba policy가 velocity 예측
v_pred = FlowMatchingPolicy(σ, z_t, x_noisy)  # [B, 10, 6]
```

**Loss:**

$$\mathcal{L}_{\text{flow}} = \text{MSE}(v_{\text{pred}},\ v_{\text{target}}) = \frac{1}{10 \times 6} \sum \left( v_{\text{pred}} - (\varepsilon - x_0) \right)^2$$

```python
# statevla_model.py:361-362
target_velocity = noise - pos_rot_actions
pos_rot_loss = F.mse_loss(velocity_pred, target_velocity)
```

| Symbol | Shape | Description |
|--------|-------|-------------|
| $x_0$ | `[B, 10, 6]` | Ground truth pos/rot actions (normalized) |
| $\varepsilon$ | `[B, 10, 6]` | Sampled Gaussian noise |
| $\sigma$ | `[B]` | Diffusion timestep (0=clean, 1=noise) |
| $x_{\text{noisy}}$ | `[B, 10, 6]` | Interpolated noisy actions |
| $v_{\text{target}}$ | `[B, 10, 6]` | Target velocity = $\varepsilon - x_0$ |
| $v_{\text{pred}}$ | `[B, 10, 6]` | Mamba policy predicted velocity |

**핵심 역할**: "임의의 noise level에서, noise를 제거하는 방향(velocity)을 정확히 예측"하는 것을 학습. 이를 통해 inference 시 pure noise에서 출발하여 iterative하게 clean action을 생성할 수 있음.

---

#### 2. Gripper BCE Loss — Gripper (1D) 이산 action 분류

Gripper는 open(-1) / close(+1) 두 상태만 존재하므로 Flow Matching이 아닌 **binary classification**으로 처리.

$$\mathcal{L}_{\text{gripper}} = -\frac{1}{10} \sum_{k=1}^{10} \left[ y_k \log \hat{p}_k + (1 - y_k) \log(1 - \hat{p}_k) \right]$$

| Symbol | Shape | Description |
|--------|-------|-------------|
| Gripper GT | `[B, 10]` | Ground truth: {-1, +1} per timestep |
| $y_k$ | `[B, 10]` | Binary target: $(gripper > 0)$ → {0, 1} |
| Logits | `[B, 10]` | MLP classifier raw output |
| $\hat{p}_k$ | `[B, 10]` | $\text{sigmoid}(\text{logit}_k)$ |

```python
# statevla_model.py:364-368
gripper_binary = (gripper_targets > 0).float()          # {-1,1} → {0,1}
gripper_loss = F.binary_cross_entropy_with_logits(
    gripper_logits, gripper_binary
)
```

**Gripper를 Flow Matching에서 분리한 이유:**
- Gripper는 이산 값 {open, close} → 연속 denoising과 맞지 않음
- Flow Matching은 연속 공간에서 smooth trajectory를 만드는 데 최적화 → 이산 binary action에는 비효율
- BCE classifier가 더 직관적이고 안정적

---

#### Inference (Action Generation)

학습된 velocity field를 사용하여 pure noise에서 clean action을 **iterative하게 복원**:

```python
# Euler method sampling (4 steps)
x = N(0, I)                                 # Start from pure noise [B, 10, 6]

for t in [1.0, 0.75, 0.5, 0.25]:            # 4 denoising steps
    v = FlowMatchingPolicy(t, z_t, x)       # predict velocity at noise level t
    x = x - (1/4) · v                       # Euler step: move toward clean data

# Gripper: independent binary prediction (no denoising needed)
logits = GripperClassifier(z_t)              # [B, 10]
gripper = where(logits > 0, +1, -1)         # [B, 10, 1]

# Combine
action = concat(x, gripper, dim=-1)         # [B, 10, 7]
```

| Step | $t$ | State of $x$ |
|------|-----|-------------|
| 0 | - | Pure Gaussian noise |
| 1 | 1.00 | Rough structure emerges |
| 2 | 0.75 | Trajectory shape forms |
| 3 | 0.50 | Fine details appear |
| 4 | 0.25 | Clean action output |

Steps를 늘리면 quality 향상, 줄이면 inference 속도 향상 (default: 4 steps)

---

## Model Components

| Component          | Model           | Parameters | Trainable |
| ------------------ | --------------- | ---------- | --------- |
| Vision Backbone    | SigLIP ViT-B/16 | ~86M       | Frozen    |
| Language Backbone  | CLIP ViT-B/32   | ~87M       | Frozen    |
| Context Encoder    | Mamba SSM (12L) | ~12.3M     | Yes       |
| Temporal Predictor | MLP             | ~1.1M      | Yes       |
| Flow Policy        | Mamba SSM (3L)  | ~1.8M      | Yes       |
| Total Trainable    | —               | ~15M       | —         |

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

```bash
python train.py --config conf/config.yaml --phase 1
```

### Phase 2 — Flow Matching

```bash
python train.py \
    --config conf/config.yaml \
    --phase 2 \
    --phase1_checkpoint checkpoints/phase1_temporal_jepa/checkpoint_best.pt
```

Multi-GPU:

```bash
torchrun --nproc_per_node=4 train.py --config conf/config.yaml --phase 1
```

---

## Evaluation

Offline evaluation:

```bash
python eval.py --checkpoint checkpoints/phase2_flow_matching/checkpoint_best.pt
```

LIBERO simulation:

```bash
python run_libero_eval.py \
    --checkpoint checkpoints/phase2_flow_matching/checkpoint_best.pt \
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
