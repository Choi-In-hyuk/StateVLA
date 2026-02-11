# StateVLA: State-based Vision-Language-Action Model

StateVLA is a Vision-Language-Action model that learns **physics-aware state representations** via Temporal JEPA, then generates smooth robot actions via Flow Matching.

The two-phase design ensures the encoder learns "what happens when I do this action" before training the policy. We leverage **Pretrained Vision (SigLIP)** and **Language (CLIP)** backbones with a lightweight **Mamba SSM** to achieve both high performance and fast inference.

## Key Innovation

**Traditional VLA**: `obs → action` (no world model, heavy computation)

**StateVLA**: Learn world dynamics first with efficient state-space models.

Phase 1 (Temporal JEPA): "이 액션을 하면 세상이 어떻게 변할까?" (World Model Learning) Phase 2 (Flow Matching): "원하는 결과를 위해 어떤 액션을 해야 할까?" (Policy Learning)


## Architecture

### Token Sequence (Mamba Causal Order)

[Lang(1)] → [Robot(1)] → [Agentview(196)] → [Eye-in-hand(196)] → [CLS(1)] ↑ Language FIRST: Mamba hidden state carries task context All info aggregated while processing vision patches ("뭘 찾아야 하는지 알고 봄")


- **Vision Backbone**: Google SigLIP (ViT-B/16) - Frozen
- **Language Backbone**: OpenAI CLIP (ViT-B/32) - Frozen
- **Sequence**: Total ~395 tokens x 256 dim
- **CLS Token**: Placed at the **end** (Mamba is causal: only the last position sees the full context)

### Phase 1: Temporal JEPA (Representation Learning)

obs_t → [SigLIP/CLIP] → [Lang,Robot,Vision,CLS] → Mamba Encoder (12L) → z_t ↓ z_t + a_t → TemporalPredictor → z'{t+1} ↓ obs{t+1} → [SigLIP/CLIP] → [Lang,Robot,Vision,CLS] → Target Encoder (EMA) → z_{t+1} MSE + VICReg ↑ ↓ (compare) ←──── z'_{t+1}


- **Input**: Pretrained features from SigLIP (Frozen) + CLIP (Frozen)
- **TemporalPredictor**: Residual prediction `z'_{t+1} = z_t + delta` (small action → small change)
- **Loss**: MSE(prediction, target) + Variance regularization + Covariance decorrelation
- **EMA**: Target encoder updated via exponential moving average (cosine schedule)

### Phase 2: Flow Matching (Policy Learning)

obs_t → Frozen Mamba Encoder → z_t ──┬──→ Flow Matching (Mamba 3L) → pos/rot (6D) └──→ Gripper Classifier (MLP) → gripper (1D) ↓ action [B, 10, 7]


- Mamba Context Encoder is **frozen** from Phase 1.
- **Position/Rotation** (6D): Iterative denoising from noise (4 Euler steps).
- **Gripper** (1D): Binary classifier (BCE loss), separated from continuous Flow Matching.

### Model Components

| Component | Model | Parameters | Learnable? |
|-----------|-------|-----------|------------|
| **Vision Backbone** | Google SigLIP (ViT-B/16) | ~86M | ❄️ Frozen |
| **Lang Backbone** | OpenAI CLIP (ViT-B/32) | ~87M | ❄️ Frozen |
| **Context Encoder** | Mamba SSM (12 layers, d=256) | ~12.3M | 🔥 Trainable |
| **Target Encoder** | EMA copy of Context Encoder | shared | ❄️ EMA Update |
| **Temporal Predictor** | MLP (3 layers, hidden=512) | ~1.1M | 🔥 Trainable (Phase 1) |
| **Flow Policy** | Mamba SSM (3 layers, d=256) | ~1.8M | 🔥 Trainable (Phase 2) |
| **Total Trainable** | | **~15M** | (Lightweight!) |

### Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| **Backbone** | Mamba SSM | O(N) linear complexity for long sequences (395 tokens) |
| **Tokenizer** | SigLIP + CLIP | Leverage massive pretrained knowledge (better than training from scratch) |
| **Token Order** | Lang → Robot → Vision | Mamba is causal; language must condition vision processing |
| **Temporal Predictor** | Residual (z_t + delta) | Easier to learn changes ($\delta$) than full states |
| **Gripper** | Separate Classifier | Discrete actions (open/close) don't fit Flow Matching |
| **Two-phase** | Phase 1 → freeze → Phase 2 | Establish stable world model before learning policy |

## Data Format

### LIBERO Dataset

| Field | Dimension | Description |
|-------|-----------|-------------|
| `obs/agentview_rgb` | [T, 224, 224, 3] | Third-person camera |
| `obs/eye_in_hand_rgb` | [T, 224, 224, 3] | Wrist camera |
| `obs/joint_states` | [T, 7] | Joint positions |
| `obs/gripper_states` | [T, 2] | Left/right finger positions |
| `actions` | [T, 7] | 6D pos/rot + 1D binary gripper {-1, 1} |

## Installation

```bash
git clone [https://github.com/Choi-In-hyuk/StateVLA.git](https://github.com/Choi-In-hyuk/StateVLA.git)
cd StateVLA

conda create -n statevla python=3.10
conda activate statevla

# Core dependencies
pip install torch torchvision
pip install mamba-ssm causal-conv1d
pip install transformers pyyaml tqdm numpy imageio h5py einops

# LIBERO (for evaluation)
git clone [https://github.com/Lifelong-Robot-Learning/LIBERO.git](https://github.com/Lifelong-Robot-Learning/LIBERO.git)
cd LIBERO && pip install -e . && cd ..
Training
Phase 1: Temporal JEPA (World Model)
Bash
python train.py --config conf/config.yaml --phase 1
Goal: Learn physics and causality.

Input: obs_t, action_t

Target: obs_{t+1} (in latent space)

Note: First run will download SigLIP/CLIP models automatically.

Phase 2: Flow Matching (Policy)
Bash
python train.py --config conf/config.yaml --phase 2 \
    --phase1_checkpoint checkpoints/phase1_temporal_jepa/checkpoint_best.pt
Goal: Learn to act based on the frozen world model.

Input: obs_t

Output: action_sequence (smooth trajectory)

Multi-GPU (DDP)
Bash
torchrun --nproc_per_node=4 train.py --config conf/config.yaml --phase 1
Evaluation
Bash
# Offline (MSE Check)
python eval.py --checkpoint checkpoints/phase2_flow_matching/checkpoint_best.pt

# LIBERO Simulation (Success Rate)
python run_libero_eval.py \
    --checkpoint checkpoints/phase2_flow_matching/checkpoint_best.pt \
    --task_suite libero_object --num_trials 50
Configuration (conf/config.yaml)
YAML
model:
  # Pretrained Backbones
  use_pretrained_vision: true
  vision_model_name: "google/siglip-base-patch16-224"
  use_pretrained_language: true
  
  # Architecture Dimensions
  embed_dim: 256              
  encoder_depth: 12           
  state_dim: 256              
  action_dim: 7               

training:
  phase1:
    num_epochs: 1000
    learning_rate: 1.0e-4
  phase2:
    num_epochs: 1000
    learning_rate: 5.0e-5
Citation
Code snippet
@article{statevla2025,
  title={StateVLA: State-based Vision-Language-Action Model with Temporal JEPA},
  author={Choi, In-hyuk},
  year={2025}
}
License
MIT License
