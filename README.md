# StateVLA: State-based Vision-Language-Action Model

StateVLA is a Vision-Language-Action model that learns **physics-aware state representations** via Temporal JEPA, then generates smooth robot actions via Flow Matching. The two-phase design ensures the encoder learns "what happens when I do this action" before training the policy.

## Key Innovation

**Traditional VLA**: `obs → action` (no world model)

**StateVLA**: Learn world dynamics first, then act.

```
Phase 1 (Temporal JEPA):  "이 액션을 하면 세상이 어떻게 변할까?"
Phase 2 (Flow Matching):  "원하는 결과를 위해 어떤 액션을 해야 할까?"
```

## Architecture

### Token Sequence (Mamba Causal Order)

```
[Lang(1)] → [Robot(1)] → [Agentview(196)] → [Eye-in-hand(196)] → [CLS(1)]
                                                                     ↑
  Language FIRST: Mamba hidden state carries task context     All info aggregated
  while processing vision patches ("뭘 찾아야 하는지 알고 봄")
```

- Total: **395 tokens x 256 dim**
- Each token has a **modality embedding** (camera0, camera1, language, robot_state)
- CLS at **end** (Mamba is causal: only last position sees everything)

### Phase 1: Temporal JEPA (Representation Learning)

```
obs_t → Tokenizer → [Lang,Robot,Vision,CLS] → Mamba Encoder (12L) → z_t
                                                                      ↓
                                                         z_t + a_t → TemporalPredictor → z'_{t+1}
                                                                                            ↓
obs_{t+1} → Tokenizer → [Lang,Robot,Vision,CLS] → Target Encoder (EMA) → z_{t+1}    MSE + VICReg
                                                                            ↑              ↓
                                                                      (compare) ←──── z'_{t+1}
```

- **TemporalPredictor**: Residual prediction `z'_{t+1} = z_t + delta` (small action → small change)
- **Loss**: MSE(prediction, target) + Variance regularization + Covariance decorrelation
- **EMA**: Target encoder updated via exponential moving average (cosine schedule)

### Phase 2: Flow Matching (Policy Learning)

```
obs_t → Frozen Encoder → z_t ──┬──→ Flow Matching (Mamba 3L) → pos/rot (6D)
                                └──→ Gripper Classifier (MLP)  → gripper (1D)
                                                                      ↓
                                                              action [B, 10, 7]
```

- Encoder **frozen** from Phase 1
- **Position/Rotation** (6D): Iterative denoising from noise (4 Euler steps)
- **Gripper** (1D): Binary classifier (BCE loss), separate from continuous Flow Matching

### Model Components

| Component | Model | Parameters | File |
|-----------|-------|-----------|------|
| Tokenizer | Conv2d patches + Linear projections | ~1.5M | `jepa/tokenizer.py` |
| Context Encoder | Mamba SSM (12 layers, d=256) | ~12.3M | `jepa/encoder.py` |
| Target Encoder | EMA copy of Context Encoder | shared | `jepa/encoder.py` |
| Temporal Predictor | MLP (3 layers, hidden=512) | ~1.1M | `jepa/temporal_predictor.py` |
| State Projection | Linear + LayerNorm | ~66K | `state_encoder.py` |
| Flow Matching Policy | Mamba SSM (3 layers, d=256) | ~1.8M | `action_policy.py` |
| Gripper Classifier | MLP (3 layers) | ~132K | `action_policy.py` |
| **Total** | | **~14.9M** | |

### Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Backbone | Mamba SSM | O(N) linear complexity for 395-token sequences |
| Token order | Lang → Robot → Vision → CLS | Mamba causal: language conditions vision processing |
| CLS position | End of sequence | Only last position aggregates all information in Mamba |
| Temporal Predictor | Residual (z_t + delta) | Small actions produce small state changes |
| Gripper | Separate classifier (BCE) | Discrete open/close, not suitable for Flow Matching |
| VICReg | Variance + Covariance | Prevents representation collapse |
| Two-phase | Phase 1 → freeze → Phase 2 | Mature representations before policy learning |

## Data Format

### LIBERO Dataset

| Field | Dimension | Description |
|-------|-----------|-------------|
| `obs/agentview_rgb` | [T, 224, 224, 3] | Third-person camera |
| `obs/eye_in_hand_rgb` | [T, 224, 224, 3] | Wrist camera |
| `obs/joint_states` | [T, 7] | Joint positions |
| `obs/gripper_states` | [T, 2] | Left/right finger positions |
| `actions` | [T, 7] | 6D pos/rot + 1D binary gripper {-1, 1} |

- **Observation robot_state** (9D): `joint(7) + gripper_fingers(2)` → tokenized as encoder input
- **Action pos/rot** (6D): Normalized (zero-mean, unit-std) → Flow Matching
- **Action gripper** (1D): Binary {-1, 1}, not normalized → BCE classifier

## Installation

```bash
git clone https://github.com/Choi-In-hyuk/StateVLA.git
cd StateVLA

conda create -n statevla python=3.10
conda activate statevla

pip install torch torchvision
pip install mamba-ssm
pip install transformers pyyaml tqdm numpy imageio h5py einops

# LIBERO (for evaluation)
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git
cd LIBERO && pip install -e . && cd ..
```

## Training

### Phase 1: Temporal JEPA

```bash
python train.py --config conf/config.yaml --phase 1
```

Learns: "Given state z_t and action a_t, what will z_{t+1} look like?"

### Phase 2: Flow Matching

```bash
python train.py --config conf/config.yaml --phase 2 \
    --phase1_checkpoint checkpoints/phase1_temporal_jepa/checkpoint_best.pt
```

Learns: Smooth action generation conditioned on frozen z_t.

### Multi-GPU (DDP)

```bash
torchrun --nproc_per_node=4 train.py --config conf/config.yaml --phase 1
```

## Evaluation

```bash
# Offline (MSE)
python eval.py --checkpoint checkpoints/phase2_flow_matching/checkpoint_best.pt

# LIBERO Simulation
python run_libero_eval.py \
    --checkpoint checkpoints/phase2_flow_matching/checkpoint_best.pt \
    --task_suite libero_object --num_trials 50
```

## Project Structure

```
StateVLA/
├── jepa/
│   ├── tokenizer.py            # Multi-modal tokenizer (Conv2d + projections)
│   ├── encoder.py              # Context Encoder (Mamba) + Target Encoder (EMA)
│   ├── temporal_predictor.py   # z_t + a_t → z'_{t+1} (Phase 1)
│   ├── predictor.py            # Legacy JEPA predictor (unused)
│   └── masking.py              # Legacy modality masking (unused)
├── mamba/                      # Mamba SSM backbone
├── utils/                      # MLP, TimeEmbedding, etc.
├── state_encoder.py            # JEPAStateEncoder (tokenizer + encoder + predictor)
├── statevla_model.py           # StateVLA + StateVLATrainer (two-phase)
├── action_policy.py            # FlowMatchingPolicy + GripperClassifier
├── train.py                    # Training script (--phase 1 / --phase 2)
├── eval.py                     # Offline evaluation
├── run_libero_eval.py          # LIBERO simulation evaluation
├── dataloader.py               # Dataset with action normalization
└── conf/config.yaml            # Configuration
```

## Configuration

```yaml
model:
  embed_dim: 256              # Token/state dimension
  encoder_depth: 12           # Mamba encoder layers
  state_dim: 256              # State representation dimension
  action_dim: 7               # 6 pos/rot + 1 gripper
  action_seq_len: 10          # Action chunk length
  robot_state_dim: 9          # 7 joints + 2 gripper fingers

training:
  phase1:
    num_epochs: 1000
    learning_rate: 1.0e-4
    temporal_predictor_hidden_dim: 512
  phase2:
    num_epochs: 1000
    learning_rate: 5.0e-5
```

## Citation

```bibtex
@article{statevla2025,
  title={StateVLA: State-based Vision-Language-Action Model with Temporal JEPA},
  author={Choi, In-hyuk},
  year={2025}
}
```

## License

MIT License
