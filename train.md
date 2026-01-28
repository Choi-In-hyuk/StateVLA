# StateVLA Training Guide

## Overview

StateVLA 모델을 LIBERO-Object 데이터셋으로 학습하는 가이드입니다.

## Model Configuration

### Vision Encoder
- **Model**: Eagle2-1B (nvidia/Eagle2-1B)
- **Base**: SigLIP vision encoder
- **Hidden size**: 1152
- **Status**: Frozen (학습 안 됨)
- **Image size**: 384x384

### Language Encoder
- **Model**: Qwen2-7B-Instruct (Qwen/Qwen2-7B-Instruct)
- **Embedding dimension**: 3584
- **Status**: Frozen (학습 안 됨)
- **Language embeddings**: Pre-generated, stored in `data/libero/language_embeddings/libero_object.pkl`

### State Encoder (Multimodal Fusion)
- **Type**: MLP (기본) 또는 Cross-Attention (선택 가능)
- **역할**: Vision, Language, Robot State 융합
- **MLP**: 단순 concatenation + MLP (빠름, 간단)
- **Cross-Attention**: Transformer attention으로 융합 (느림, 표현력 높음)

### Model Statistics
- **Total parameters**: 7,507.93M
- **Trainable parameters**: 8.75M (vision과 language encoder는 frozen)
- **Architecture**: StateVLA with Mamba-SSM backbone

## Dataset

### LIBERO-Object
- **Location**: `/home/choi/StateVLA/data/libero/libero_object`
- **Tasks**: 10개 pick-and-place tasks
  - pick_up_the_butter_and_place_it_in_the_basket
  - pick_up_the_chocolate_pudding_and_place_it_in_the_basket
  - pick_up_the_salad_dressing_and_place_it_in_the_basket
  - pick_up_the_ketchup_and_place_it_in_the_basket
  - pick_up_the_milk_and_place_it_in_the_basket
  - pick_up_the_alphabet_soup_and_place_it_in_the_basket
  - pick_up_the_orange_juice_and_place_it_in_the_basket
  - pick_up_the_cream_cheese_and_place_it_in_the_basket
  - pick_up_the_bbq_sauce_and_place_it_in_the_basket
  - pick_up_the_tomato_sauce_and_place_it_in_the_basket
- **Trajectories**: 500개 (각 태스크당 50개)
- **Total samples**: 70,007개
- **Cameras**: agentview, eye_in_hand

## Training Configuration

### Hyperparameters
```yaml
batch_size: 32
num_epochs: 2000
learning_rate: 0.0001
weight_decay: 0.05
gradient_clip: 1.0
action_loss_weight: 1.0
state_loss_weight: 0.1
```

### Training Settings
```yaml
enable_ema: true
ema_decay_rate: 0.995
sampling_steps: 4
save_interval: 100  # 100 에포크마다 체크포인트 저장
checkpoint_dir: checkpoints/libero_object
```

### Model Architecture
```yaml
latent_dim: 256
state_encoder_type: "mlp"  # mlp or cross_attention
state_dim: 256
action_dim: 7
action_seq_len: 10
obs_tok_len: 2
policy_layers: 3
policy_embed_dim: 256
state_predictor_layers: 4
use_correction: true
dropout: 0.1
```

## Environment Setup

### GPU
- **Device**: NVIDIA RTX PRO 6000 Blackwell Max-Q (sm_120)
- **CUDA**: 12.9
- **Note**: Blackwell GPU는 PyTorch nightly 버전 필요

### Python Environment
```bash
# PyTorch nightly (CUDA 12.8)
pip install torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu128

# Mamba-SSM (소스에서 빌드 필요)
cd /tmp
git clone https://github.com/state-spaces/mamba.git
cd mamba
CUDA_HOME=/usr/local/cuda-12.9 pip install -e . --no-build-isolation

# Other dependencies
pip install transformers accelerate peft h5py ftfy tiktoken
```

### Flash Attention 비활성화
Blackwell GPU에서 Flash Attention 호환성 문제로 인해 eager attention 사용:
- `train.py`에서 transformers monkey-patch 적용
- 환경변수: `TRANSFORMERS_NO_FLASH_ATTN=1`

## Running Training

### Basic Command
```bash
bash run_train.sh
```

### With Logging
```bash
bash run_train.sh 2>&1 | tee training.log
```

### Advanced Options

#### State Encoder 선택
```bash
# MLP (기본값 - 빠름)
python train.py --config conf/config_libero_object.yaml

# Cross-Attention (더 표현력 높음, 약간 느림)
python train.py --config conf/config_libero_object.yaml --cross_attention

# 또는
python train.py --config conf/config_libero_object.yaml --state_encoder_type cross_attention
```

#### 기타 옵션
```bash
# 데이터 경로 override
python train.py --config conf/config.yaml --data_directory /path/to/data

# 디바이스 선택
python train.py --config conf/config.yaml --device cuda

# 체크포인트에서 재개
python train.py --config conf/config.yaml --checkpoint checkpoints/checkpoint_latest.pt

# 옵션 조합
python train.py \
    --config conf/config_libero_object.yaml \
    --cross_attention \
    --checkpoint checkpoints/run_20260126/checkpoint_epoch_100.pt
```

### Monitor Training
```bash
# 실시간 로그 확인
tail -f training.log

# GPU 사용률 확인
watch -n 1 nvidia-smi
```

## Training Progress

### Expected Performance
- **Speed**: ~3.3초/배치
- **Epoch time**: ~2시간/에포크 (2188 배치)
- **Loss**: 초기 ~1.3 → 0.6-0.7로 감소

### Checkpoints
체크포인트는 `checkpoints/libero_object/run_YYYYMMDD_HHMMSS/`에 저장됨:
- `checkpoint_latest.pt`: 최신 체크포인트
- `checkpoint_epoch_N.pt`: N 에포크 체크포인트
- `checkpoint_best.pt`: 최고 성능 체크포인트

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

## Resume Training

체크포인트에서 학습 재개:
```bash
python train.py \
    --config conf/config_libero_object.yaml \
    --checkpoint checkpoints/libero_object/run_YYYYMMDD_HHMMSS/checkpoint_latest.pt
```

## Key Files

- `run_train.sh`: 학습 실행 스크립트
- `train.py`: 메인 학습 스크립트
- `conf/config_libero_object.yaml`: 학습 설정
- `dataloader.py`: 데이터 로딩
- `model_factory.py`: 모델 생성
- `statevla_model.py`: StateVLA 아키텍처
- `train_policy.py`: Training wrapper with flow matching
- `backbone/eagle2/eagle2_vision_encoder.py`: Eagle2 vision encoder
- `backbone/qwen/qwen_lang_encoder.py`: Qwen language encoder

## Troubleshooting

### Common Issues

1. **Flash Attention Error**
   - 해결: `train.py`의 monkey-patch가 자동으로 처리
   - 환경변수 `TRANSFORMERS_NO_FLASH_ATTN=1` 설정됨

2. **CUDA Version Mismatch**
   - PyTorch nightly 필요: `torch==2.11.0.dev`
   - Mamba-SSM 소스 빌드 필요

3. **Out of Memory**
   - Batch size 줄이기: `config_libero_object.yaml`에서 `batch_size` 조정
   - Image size 줄이기: `image_size` 조정 (현재 384)

4. **Slow Training**
   - 정상: Eagle2-1B + Qwen2-7B는 큰 모델
   - 배치당 ~3초는 예상 속도

## Model Improvements

현재 모델을 개선할 수 있는 방법들:

### 1. Cross-Attention Fusion (구현 완료)
```bash
python train.py --config conf/config_libero_object.yaml --cross_attention
```
- 멀티모달 융합 개선
- 언어가 vision에 "어디를 봐야 하는지" 알려줌
- 약간 느려지지만 성능 향상 가능

### 2. Vision Token 수 증가
```yaml
# config에서 변경
model:
  obs_tok_len: 8  # 현재 2 → 8로 증가
```
- 공간 정보 손실 감소
- 더 풍부한 visual representation

### 3. History 추가
```yaml
# state_predictor에서
use_history: true
history_len: 5-10
```
- 궤적의 momentum 이해
- 시간적 context 활용

### 4. Data Augmentation
- Random crop, color jitter
- Action noise injection
- Temporal dropout

### 5. LoRA Fine-tuning
- Vision/Language encoder를 약간 fine-tune
- 메모리: +1-2GB, 파라미터: +10M
- 로봇 도메인에 특화

### 6. Flow Matching 개선
- Runge-Kutta solver 사용
- Adaptive step size
- 더 적은 step으로 더 나은 품질

## Architecture Details

### 전체 구조
```
Frozen Encoders (7.5B params):
  ├─ Eagle2-1B (Vision) → 1152dim features
  └─ Qwen2-7B (Language) → 3584dim embeddings

Trainable Components (8.75M params):
  ├─ Projection Layers → 256dim latent
  ├─ State Encoder (MLP or Cross-Attention) → 멀티모달 융합
  ├─ State Predictor (Mamba-SSM) → 미래 상태 예측
  └─ Action Policy (Flow Matching + Mamba) → 행동 생성
```

### State Encoder 비교

**MLP (기본)**:
- Concatenation + 3-layer MLP
- 빠름: ~3.3초/배치
- 간단하고 안정적
- 8.75M 파라미터

**Cross-Attention (선택)**:
- Transformer-style attention
- 약간 느림: ~3.5-3.8초/배치
- 더 나은 멀티모달 이해
- ~9M 파라미터

### Flow Matching vs Diffusion

현재 사용 중인 Flow Matching:
- Velocity 예측 (vs Diffusion의 noise 예측)
- 4 steps로 충분 (vs Diffusion의 50+ steps)
- 더 빠르고 안정적
- 로봇 제어에 적합

## Notes

- Vision과 language encoder는 frozen 상태로 학습됨
- 실제 학습되는 파라미터는 8.75M (전체의 0.1%)
- Language embeddings는 사전 생성되어 캐싱됨
- 데이터는 on-the-fly로 augmentation 없이 로딩됨
- Mamba-SSM은 Transformer 대체 (O(n) vs O(n²) complexity)
