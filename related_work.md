# Related Work

## 1. Model Size Comparison

| Model | Parameters | Architecture | LIBERO Avg |
|---|---|---|---|
| **StateVLA (Ours)** | **15M** | Mamba encoder + Flow Matching | TBD |
| MambaVLA (ResNet) | ~25M | ResNet18 + Mamba + BC | - |
| MambaVLA (Eagle) | ~1.5B | Eagle(LLaMA/Qwen2) + Mamba + BC | - |
| SmolVLA | 450M | SmolVLM2 + Transformer | 61.9% (L-90) |
| π0-small | 470M | ViT (no VLM init) | - |
| TinyVLA | 422M–1.3B | Pythia VLM + Diffusion | > OpenVLA |
| VLA-0-Smol | 500M | SmolVLM2 + Flow Matching | 94.1% |
| SimVLA | 500M+ | SmolVLM-0.5B + Action head | 98.6% |
| VLA-Adapter | 597M | Qwen2.5-0.5B + Policy (97M) | 97.3–98.5% |
| MiniVLA | ~1B | Qwen2.5-0.5B + ViT | 82% (L-90) |
| GR00T N1 | 2.2B | NVIDIA Eagle-2 + Flow Matching | - |
| OpenVLA | 7B | LLaMA + ViT | ~62% (L-90) |
| LAPA | 7B | LWM backbone | - |

**StateVLA는 최소 크기 경쟁자(SmolVLA 450M)보다 30배 작다.**
Phase 2에서 실제로 학습되는 파라미터는 action_policy **2.66M** 뿐이며 encoder는 frozen이다.

---

## 2. Closest Related Papers

### FLARE: Robot Learning with Implicit World Modeling (2025.05, arXiv:2505.15659)
**가장 유사한 구조.**

| 항목 | FLARE | StateVLA |
|---|---|---|
| EMA target encoder | ✓ (ρ=0.995) | ✓ (ρ=0.996) |
| Flow matching | ✓ | ✓ |
| Two-phase training | ✓ | ✓ |
| JEPA temporal prediction | ✗ | ✓ |
| 데이터 규모 | 2,000시간 대규모 | 소규모 demo only |
| 모델 크기 | 대형 | 15M 경량 |

FLARE는 Phase 1에서 대규모 robot video로 action-aware embedding을 학습하지만, JEPA 방식의 명시적 temporal prediction (z_t + a_t → z_{t+1})이 없다. StateVLA는 소규모 demo 데이터만으로도 temporal JEPA를 통해 causal representation을 학습한다.

---

### V-JEPA 2 (2025.06, arXiv:2506.09985)
**JEPA 측면에서 가장 유사.**

| 항목 | V-JEPA 2 | StateVLA |
|---|---|---|
| JEPA + EMA target | ✓ | ✓ |
| Action-conditioned predictor | ✓ | ✓ |
| Two-phase | ✓ | ✓ |
| Flow matching | ✗ (CEM planning) | ✓ |
| 데이터 | 1M시간 인터넷 비디오 | 소규모 demo |
| 도메인 | 비디오 이해 + planning | robot manipulation |

V-JEPA 2는 pixel-level video prediction을 기반으로 하며 action generation에 flow matching이 아닌 Cross-Entropy Method(CEM)를 사용한다. StateVLA는 latent space에서만 예측하여 계산 효율을 높이고, flow matching으로 smooth continuous action을 생성한다.

---

### GR-1: Unleashing Large-Scale Video Generative Pre-training (2023)
**개념적으로 가장 유사한 선행 연구.**

- GPT 기반으로 다음 프레임 예측을 사전학습 → manipulation policy fine-tuning
- StateVLA와 동일한 "temporal 예측 → policy" 2단계 구조
- **차이**: pixel-level 재구성 (계산 비용 큼), large backbone, flow matching 아님
- StateVLA는 latent space JEPA로 pixel 재구성 없이 표현만 학습 → 더 경량·효율적

---

### LAPA: Latent Action Pretraining from Videos (2024.10, arXiv:2410.11758)

- VQ-VAE로 discrete latent action 학습 → 대형 VLM fine-tuning
- 3단계 학습 (latent quantization → language pretraining → action fine-tuning)
- **차이**: action label 없는 인터넷 비디오 활용이 핵심, JEPA 아님, flow matching 아님
- StateVLA보다 30배+ 큰 모델

---

### TD-MPC2 (2023)

- latent world model + temporal difference learning으로 policy 학습
- world model 예측으로 policy를 regularize → StateVLA의 L_world와 유사한 아이디어
- **차이**: RL 기반 (StateVLA는 imitation learning), JEPA 아님

---

### Diffusion Policy (2023) / FlowPolicy (2024)

- Flow matching / diffusion으로 robot action 생성
- StateVLA Phase 2의 action policy 부분과 유사
- **차이**: 표현 학습 단계 없음, 단일 단계 학습

---

## 3. StateVLA의 Novelty

기존 연구들과 비교했을 때 StateVLA의 고유한 조합:

1. **JEPA-style temporal prediction** (latent space, VICReg + MSE, EMA target encoder)
2. **Flow matching action policy** (continuous 7D action)
3. **Two-phase training** (Phase 1: JEPA, Phase 2: frozen encoder + flow matching)
4. **World model consistency loss** (L_world: 예측 action의 물리적 타당성 regularization)
5. **Spatial patch cross-attention** (image patch tokens → action tokens, Phase 2만)
6. **15M 초경량** (소규모 demo 데이터만 사용)

이 조합 전체를 동시에 사용한 논문은 2026년 3월 기준으로 확인되지 않았다.

---

## 4. 차별점 요약

| 비교 축 | 기존 연구 | StateVLA |
|---|---|---|
| 표현 학습 | pixel 재구성 또는 대형 VLM pretrain | latent JEPA (경량, 소규모 데이터) |
| Action 생성 | BC, diffusion, autoregressive | Flow matching (연속, 부드러움) |
| 물리 일관성 | 없음 (대부분) | L_world (temporal predictor 재활용) |
| 데이터 요구량 | 수천 시간 (FLARE, V-JEPA 2) | demo data만 |
| 모델 크기 | 450M–7B | **15M** |
