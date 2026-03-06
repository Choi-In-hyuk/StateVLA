# StateVLA Model Architecture

StateVLA는 2단계 학습으로 구성된 Vision-Language-Action 모델입니다.

- **Phase 1 (Temporal JEPA)**: Encoder가 "이 액션을 하면 미래 상태가 어떻게 될까?"를 학습
- **Phase 2 (Flow Matching)**: 고정된 Encoder 위에서 Action Policy를 학습

---

## 목차

1. [입력 데이터](#1-입력-데이터)
2. [멀티모달 토크나이저](#2-멀티모달-토크나이저)
3. [Phase 1 — Temporal JEPA](#3-phase-1--temporal-jepa)
4. [Phase 2 — Flow Matching Policy](#4-phase-2--flow-matching-policy)
5. [추론 (Inference)](#5-추론-inference)
6. [전체 파라미터](#6-전체-파라미터)
7. [설정값 (config.yaml)](#7-설정값-configyaml)

---

## 1. 입력 데이터

### 1.1 이미지

- **카메라**: `agentview` (3인칭), `eye_in_hand` (손목)
- **원본 크기**: 128×128 (HDF5 저장)
- **전처리**: `/ 255.0` → [0, 1] 범위, `TF.resize` → 224×224
- **형식**: `[B, 3, 224, 224]` float32

```
※ SigLIP 권장 정규화 mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5] 는 미적용 상태.
  두 encoder 모두 동일하게 처리되므로 내부 일관성은 유지됨.
```

### 1.2 언어 임베딩

- **출처**: CLIP ViT-B/32로 사전 계산 → `data/libero/language_embeddings/libero_object.pkl`
- **형식**: `{task_name: Tensor[1, 512]}`
- **예시**: `"pick_up_the_butter_and_place_it_in_the_bowl"` → [1, 512] float32

학습 시 각 task의 고정 임베딩이 모든 demo에 동일하게 사용됨 (online CLIP 추론 없음).

### 1.3 로봇 상태

- **구성**: `joint_states` (7) + `gripper_states` (2) = **9D**
- **단위**: 관절 라디안 [-3.14, 3.14], 그리퍼 미터 [0, 0.04]
- **형식**: `[B, 9]` float32
- **정규화**: 미적용 (토크나이저 MLP가 학습 중 보정)

### 1.4 액션

- **구성**: `pos(3) + rot(3) + gripper(1)` = **7D**
- **정규화**: 전체 데이터셋의 mean/std로 z-score 정규화 (모든 7차원 동일 적용)

```python
# 데이터셋 로드 시 한 번 계산
action_mean = all_actions.mean(dim=0)   # [7]
action_std  = all_actions.std(dim=0)    # [7], min clamp 1e-6
actions_normalized = (actions - action_mean) / action_std
```

---

## 2. 멀티모달 토크나이저

파일: `jepa/tokenizer.py` — `MultiModalTokenizer`

### 2.1 토큰 순서 (Mamba 인과 설계)

```
[Lang(1)] → [Robot(1)] → [AgentView(196)] → [EyeInHand(196)] → [CLS(1)]
 총 395 토큰, 각 D=256
```

Mamba는 단방향(causal)이므로 Language가 맨 앞에 위치해야
이후 Vision 토큰들이 Language 정보를 조건으로 받을 수 있음.
CLS는 마지막에 위치해 모든 토큰을 집계.

### 2.2 이미지 토크나이저 (SigLIP)

```
config: use_pretrained_vision=True, vision_model_name="google/siglip-base-patch16-224"
        freeze_vision=True (gradient 없음)
```

```
이미지 [B, 3, 224, 224]
  └─ SigLIP vision_model (frozen)
       └─ [B, 196, 768]   (patch embedding + attention)
  └─ Linear(768, 256)     (학습 가능 projection)
  └─ + pos_embed[1, 196, 256]
  └─ + modality_embed[AGENTVIEW or WRIST]
  = [B, 196, 256]  per camera
```

agentview와 eye_in_hand는 **동일한 SigLIP backbone**을 공유하고
각자 별도의 `Linear(768, 256)` projection 레이어를 가짐.

### 2.3 언어 토크나이저

```
config: use_pretrained_language=True (사전계산 임베딩 사용)
```

```
lang_emb [B, 512]  (pre-computed CLIP)
  └─ Linear(512, 256)   (학습 가능)
  └─ reshape [B, 1, 256]
  └─ + pos_embed[1, 1, 256]
  └─ + modality_embed[LANGUAGE]
  = [B, 1, 256]
```

### 2.4 로봇 상태 토크나이저

```
robot_state [B, 9]
  └─ MLP: Linear(9, 256) → GELU → Linear(256, 256)
  └─ reshape [B, 1, 256]
  └─ + pos_embed[1, 1, 256]
  └─ + modality_embed[ROBOT_STATE]
  = [B, 1, 256]
```

### 2.5 CLS 토큰

```
cls_token: nn.Parameter [1, 1, 256], trunc_normal_(std=0.02)
  마지막에 append → Mamba 끝에서 모든 토큰 정보를 집계
```

### 2.6 Modality Embedding

각 토큰 그룹에 4종의 modality embedding 추가 (nn.Embedding):

| ID | 의미 |
|----|------|
| 0 | AGENTVIEW |
| 1 | EYE_IN_HAND (WRIST) |
| 2 | LANGUAGE |
| 3 | ROBOT_STATE |

---

## 3. Phase 1 — Temporal JEPA

### 3.1 개요

"현재 상태 z_t에서 액션 a_t를 하면 미래 z_{t+1}이 어떻게 될지" 예측하는 world model 학습.

```
obs_t   → Tokenizer → ContextEncoder → CLS → state_proj → z_t       [B, 256]
obs_t+1 → Tokenizer → TargetEncoder  → CLS → state_proj → z_{t+1}   [B, 256]  (no grad)

z_t + a_t → TemporalPredictor → z'_{t+1}   [B, 256]

Loss: MSE(z'_{t+1}, z_{t+1}) + VICReg(z'_{t+1})
```

### 3.2 Context Encoder

파일: `jepa/encoder.py` — `ContextEncoder`

```
tokens [B, 395, 256]
  └─ MixerModel (Mamba-12L)
       ssm_cfg: layer=Mamba1, d_state=16, d_conv=4, expand=2
       d_intermediate=0  (순수 Mamba, MLP 없음)
  └─ LayerNorm(256)
  └─ hidden [B, 395, 256]

CLS output = hidden[:, -1]   [B, 256]  (마지막 토큰)
features   = hidden[:, :-1]  [B, 394, 256]
```

Mamba의 causal 특성상 마지막 위치의 CLS 토큰이 앞선 394개 토큰을 모두 본 상태.
이 CLS 출력이 관찰의 전역 표현 z_t가 됨.

### 3.3 State Projection

```
CLS [B, 256]
  └─ Linear(256, 256)
  └─ LayerNorm(256)
  = z_t [B, 256]
```

CLS → state_dim 변환. LayerNorm으로 표현 안정화.

### 3.4 Target Encoder (EMA)

파일: `jepa/encoder.py` — `TargetEncoder`

Context Encoder의 deep copy. **파라미터 고정 (requires_grad=False)**.
매 step마다 EMA로 Context Encoder를 따라 업데이트:

```
target_param = momentum * target_param + (1 - momentum) * context_param
```

**EMA Momentum 스케줄 (cosine)**:

```
초기:  momentum ≈ 0.996  (천천히 업데이트)
후기:  momentum → 1.0    (거의 고정)
```

z_{t+1} target은 이 encoder에서 gradient 없이 추출됨 → 안정적인 학습 신호.

### 3.5 Temporal Predictor

파일: `jepa/temporal_predictor.py` — `TemporalPredictor`

```
z_t [B, 256]  → Linear(256, 512) → state_emb  [B, 512]
a_t [B, 7]    → Linear(7, 512)   → action_emb [B, 512]

concat [B, 1024]
  └─ Linear(1024, 512) → SiLU → LayerNorm(512)
  └─ Linear(512, 512)  → SiLU → LayerNorm(512)
  └─ Linear(512, 256)
  = delta [B, 256]

z'_{t+1} = z_t + delta   (Residual Prediction)
```

**Residual prediction 이유**: 대부분의 액션에서 상태 변화는 작음.
절대값 예측보다 변화량(delta)를 예측하는 게 학습 안정적.
마지막 레이어 weight/bias를 0으로 초기화 → 학습 초반 항등 함수처럼 시작.

### 3.6 Phase 1 손실 함수

파일: `jepa/temporal_predictor.py` — `compute_temporal_jepa_loss`

#### MSE Loss (Invariance)

```
L_mse = MSE(z'_{t+1}, z_{t+1})
```

예측이 target과 방향/크기 모두 일치하도록.

#### VICReg — Variance Loss

```
std_per_dim = z'_{t+1}.std(dim=0)   [256]
L_var = mean(ReLU(1.0 - std_per_dim))
```

각 차원의 분산이 1 이상 되도록 강제.
**이 loss가 없으면 표현 붕괴(collapse) — 모든 샘플이 동일한 z로 수렴.**

#### VICReg — Covariance Loss

```
z_centered = z'_{t+1} - mean(z'_{t+1})
C = z_centered.T @ z_centered / (B-1)   [256, 256]
L_cov = sum(off_diagonal(C)^2) / 256
```

차원 간 상관관계를 제거해 각 차원이 독립적인 정보를 담도록.
**이 loss가 없으면 정보 중복 — 유효 rank 감소.**

#### 총 Phase 1 Loss

```
L_phase1 = L_mse + 1.0 * L_var + 0.04 * L_cov
```

| 항 | 역할 | 가중치 |
|----|------|--------|
| L_mse | temporal prediction 정확도 | 1.0 |
| L_var | 표현 붕괴 방지 | 1.0 |
| L_cov | 차원 독립성 (decorrelation) | 0.04 |

### 3.7 Phase 1 학습 설정

```
옵티마이저:  AdamW (lr=1e-4, weight_decay=0.05)
스케줄러:    CosineAnnealingLR (T_max=1000, eta_min=5e-6)
Epochs:      1000
Batch size:  256 (DDP: 128 per GPU × 2 GPU)
Grad clip:   1.0
Save:        checkpoint_best.pt (train_loss 기준), checkpoint_epoch_N.pt (200 epoch마다)
```

---

## 4. Phase 2 — Flow Matching Policy

### 4.1 개요

Phase 1에서 학습된 Encoder를 **완전 고정(freeze)**하고,
z_t → 액션 시퀀스를 생성하는 Flow Matching Policy를 학습.

```
obs_t → FrozenEncoder → z_t [B, 256]
  └─ FlowMatchingPolicy
       입력: z_t, noisy_actions, sigma
       출력: velocity [B, 10, 7]
  └─ L_flow = MSE(velocity_pred, velocity_target)

선택적:
obs_{t+1} → TargetEncoder → z_{t+1} (no grad)
z_t + â_t → TemporalPredictor → ẑ_{t+1}
  └─ L_world = MSE(ẑ_{t+1}, z_{t+1})
```

### 4.2 Encoder Freeze

```python
for param in self.state_encoder.parameters():
    param.requires_grad = False
```

Phase 1 checkpoint 로드 후 encoder 파라미터 고정.
학습되는 파라미터: `FlowMatchingPolicy` + `CorrectionMLP`만.

Phase 2에서 z_t 계산 시:
```python
with torch.no_grad():
    state_outputs = self.state_encoder(obs_dict)
z_t = state_outputs["z_t"]
```

### 4.3 Flow Matching 원리

Flow Matching은 노이즈 → 클린 액션으로의 확률적 흐름을 학습.

**Forward process (학습용 노이즈 추가)**:
```
noise ~ N(0, I)      [B, 10, 7]
sigma ~ Uniform(0,1) [B]        (diffusion timestep)

noisy_actions = (1 - sigma) * gt_actions + sigma * noise
```

sigma=0: 클린 액션, sigma=1: 순수 노이즈.

**학습 목표 (velocity)**:
```
target_velocity = noise - gt_actions
```

모델이 이 velocity를 예측하면, Euler integration으로 노이즈에서 클린 액션을 복원 가능.

**손실**:
```
L_flow = MSE(velocity_pred, target_velocity)
```

### 4.4 FlowMatchingPolicy

파일: `action_policy.py` — `FlowMatchingPolicy`

```
입력:
  z_next_pred [B, 256]   → state_proj Linear(256, 256) → [B, 1, 256]
  sigma       [B]        → TimeEmbedding(256)            → [B, 1, 256]
  noisy_actions [B,10,7] → action_emb Linear(7, 256)    → [B, 10, 256]

concat: [sigma_emb, state_emb, action_emb] = [B, 12, 256]
  └─ + pos_emb [1, 12, 256]
  └─ MixerModel (Mamba-3L)
       ssm_cfg: d_state=64, d_conv=4, expand=2
  └─ output [B, 12, 256]
  └─ 마지막 10 토큰 [B, 10, 256]
  └─ MLP(256→256→7) action_pred
  = velocity [B, 10, 7]
```

Mamba 순서: sigma가 먼저 → 모든 action 토큰이 diffusion 단계를 조건으로 받음.
state가 두 번째 → action 생성이 현재 상태를 조건으로.

### 4.5 Inference (Action 생성)

학습된 velocity를 Euler integration으로 적분:

```
actions ~ N(0, I) [B, 10, 7]   (초기 노이즈)

for t in [1.0, 0.75, 0.5, 0.25]:   (sample_steps=4)
    sigma = [t, t, ..., t]
    velocity = FlowMatchingPolicy(z_t, actions, sigma)
    actions = actions - (1/4) * velocity

return actions  (denormalized)
```

4스텝으로 노이즈 → 클린 액션 복원. step 수 늘릴수록 품질 향상 (속도 저하).

### 4.6 World Model Consistency Loss

Phase 1에서 학습한 Temporal Predictor를 Phase 2 학습에 재활용.

**아이디어**: "Policy가 생성한 액션이 물리적으로 타당한가?"를 world model로 검증.

```
# Flow matching 예측에서 클린 액션 추정
noisy_a = (1-σ)*gt_a + σ*noise
x_0_pred = noisy_a - σ * velocity_pred    # [B, 10, 7]
â_t = x_0_pred[:, 0, :]                   # 첫 번째 action [B, 7]

# World model forward (temporal predictor는 frozen)
ẑ_{t+1} = TemporalPredictor(z_t, â_t)    # [B, 256]

# 실제 다음 상태 (target encoder, no grad)
z_{t+1} = TargetEncoder(obs_{t+1})        # [B, 256]

L_world = MSE(ẑ_{t+1}, z_{t+1})
```

**Gradient 흐름**: `L_world → ẑ_{t+1} → â_t → velocity_pred → policy 파라미터`
Temporal Predictor와 Encoder는 frozen이므로 gradient가 policy로만 흐름.

**효과**: Policy가 물리적으로 불가능한 액션(너무 큰 이동, 충돌 등)을 생성하지 않도록 제약.

### 4.7 총 Phase 2 Loss

```
L_phase2 = L_flow + 0.1 * L_world
```

| 항 | 역할 | 가중치 |
|----|------|--------|
| L_flow | flow matching velocity 예측 | 1.0 |
| L_world | 물리적 타당성 (world model 일관성) | 0.1 |

### 4.8 Phase 2 학습 설정

```
옵티마이저:  AdamW (lr=5e-5, weight_decay=0.05)
스케줄러:    CosineAnnealingLR (T_max=3000, eta_min=5e-6)
Epochs:      3000
Batch size:  256 (DDP: 128 per GPU × 2 GPU)
Grad clip:   1.0
Frozen:      state_encoder 전체 (tokenizer, context_encoder, target_encoder, temporal_predictor, state_proj)
학습 가능:   action_policy (FlowMatchingPolicy + CorrectionMLP if enabled)
```

---

## 5. 추론 (Inference)

```python
# 1. 관찰 인코딩
z_t = model.state_encoder.encode(obs_dict)   # [B, 256]

# 2. Flow matching으로 액션 생성 (4스텝 Euler)
actions_normalized = model.action_policy.generate_actions(
    z_t, z_t, error=zeros, sample_steps=4
)   # [B, 10, 7]

# 3. 역정규화
actions = actions * action_std + action_mean   # [B, 10, 7]

# 4. 실행할 액션 선택 (첫 번째)
action_t = actions[:, 0, :]   # [B, 7]
```

**Receding Horizon**: 10개 액션을 생성하지만 매 step 재계획.
현재 코드: `action_idx=-1` (마지막 액션 사용) — 실험적 설정, 0이 더 일반적.

---

## 6. 전체 파라미터

| 컴포넌트 | 파라미터 수 | Phase 1 학습 | Phase 2 학습 |
|---------|-----------|-------------|-------------|
| SigLIP vision_model (frozen) | ~86M | ✗ | ✗ |
| image_projections (2 cameras) | ~0.4M | ✓ | ✗ |
| LanguageTokenizer (proj) | ~0.1M | ✓ | ✗ |
| RobotStateTokenizer (MLP) | ~0.1M | ✓ | ✗ |
| ContextEncoder (Mamba-12L) | ~12M | ✓ | ✗ |
| TargetEncoder (EMA, frozen) | ~12M | ✗ | ✗ |
| TemporalPredictor (MLP) | ~0.8M | ✓ | ✗ |
| state_proj (Linear+LN) | ~0.1M | ✓ | ✗ |
| FlowMatchingPolicy (Mamba-3L) | ~2M | ✗ | ✓ |
| **Phase 1 학습 파라미터 합계** | **~13.5M** | | |
| **Phase 2 학습 파라미터 합계** | **~2M** | | |

---

## 7. 설정값 (config.yaml)

```yaml
model:
  image_size: 224
  patch_size: 16            # 196 patches per image
  embed_dim: 256            # 토큰 차원
  lang_emb_dim: 512         # CLIP ViT-B/32 출력
  robot_state_dim: 9        # joint(7) + gripper(2)
  state_dim: 256            # z_t 차원
  action_dim: 7             # pos(3) + rot(3) + gripper(1)
  action_seq_len: 10        # 예측할 액션 시퀀스 길이

  use_pretrained_vision: true
  vision_model_name: "google/siglip-base-patch16-224"
  freeze_vision: true

  use_pretrained_language: true
  language_model_name: "ViT-B/32"
  freeze_language: true

  encoder_depth: 12         # Mamba context encoder 레이어 수
  d_state: 16               # Mamba SSM state 차원
  d_conv: 4                 # Mamba convolution 크기
  expand: 2                 # Mamba expansion factor

  policy_layers: 3          # FlowMatchingPolicy Mamba 레이어 수
  policy_embed_dim: 256

training:
  phase1:
    num_epochs: 1000
    learning_rate: 1.0e-4
    temporal_predictor_hidden_dim: 512

  phase2:
    num_epochs: 3000
    learning_rate: 5.0e-5

  variance_weight: 1.0      # VICReg variance
  covariance_weight: 0.04   # VICReg covariance
  world_model_loss_weight: 0.1
  ema_momentum: 0.996
  ema_momentum_schedule: "cosine"
  sampling_steps: 4         # inference Euler steps
  gradient_clip: 1.0
  weight_decay: 0.05
```

---

## 8. 알려진 제한점

| 항목 | 현황 | 영향 |
|------|------|------|
| SigLIP 입력 정규화 | [0,1] 사용 (권장: [-1,1]) | 피처 품질 소폭 저하 |
| Robot state 정규화 | 미적용 (joint vs gripper 스케일 차이) | MLP가 보정, 낮은 영향 |
| 단방향 Mamba | 이미지 패치 간 양방향 attention 불가 | 공간 이해력 한계 |
| Language 1토큰 | 긴 instruction의 세부 정보 손실 | task 구분에는 충분 |
| Compounding Error | 학습: 전문가 궤적, 추론: 자기 액션 | 긴 horizon에서 오차 누적 |
