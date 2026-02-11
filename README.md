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

The model learns causal world dynamics in latent space:

[
z'_{t+1} = z_t + \Delta(z_t, a_t)
]

Loss:

[
\mathcal{L} = \text{MSE}(z'*{t+1}, z*{t+1})

* \lambda_v \mathcal{L}_{variance}
* \lambda_c \mathcal{L}_{covariance}
  ]

Target encoder parameters are updated using exponential moving average (EMA).

This stage produces **physics-consistent state embeddings** suitable for control.

---

## Phase 2 — Flow Matching Policy

The encoder is frozen and used as a state representation provider:

```
obs_t → encoder → z_t → Flow Matching → action trajectory
```

Outputs:

* 6D pose trajectory (continuous)
* Binary gripper command (classifier)

Final action tensor:

```
[B, horizon=10, action_dim=7]
```

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
