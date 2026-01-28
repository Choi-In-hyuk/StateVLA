# StateVLA: State-based Vision-Language-Action Model

StateVLA is a novel Vision-Language-Action model that introduces **state prediction** as an intermediate step before action generation. Unlike traditional VLA models that directly map observations to actions, StateVLA predicts the next world state and uses this prediction (along with error correction) to generate more robust actions.

## Key Innovation

**Traditional VLA**: `obs → action` (one-shot, no feedback)

**StateVLA**: `obs → state → next_state prediction → action` (predictive + feedback loop)

```
┌─────────────────────────────────────────────────────────────────────────┐
│                            StateVLA Architecture                        │
└─────────────────────────────────────────────────────────────────────────┘

  obs_t + lang + a_{t-1} ──► StateEncoder ──► z_t (current state)
                                               │
                                               ▼
                              StatePredictor (Mamba) ──► z_{t+1}^pred
                                               │
                           ┌───────────────────┴───────────────────┐
                           │                                       │
                           ▼                                       ▼
                    Base Policy                              Correction
                   (Flow Matching)                              (MLP)
                           │                                       │
                           ▼                                       ▼
                       a_base          +                         Δa
                           └───────────────┬───────────────────────┘
                                           ▼
                                    a_t = a_base + Δa

  Error Feedback: error = z_{t+1}^pred (previous) - z_{t+1}^actual (current)
```

## Features

- **State Prediction**: Predicts next world state before generating actions
- **Residual Action Policy**: Base policy + correction for robustness
- **Flow Matching**: Diffusion-based action generation
- **Mamba Backbone**: Efficient state-space model for sequence processing
- **Closed-loop Feedback**: Uses prediction errors to improve future actions

## Installation

```bash
# Clone the repository
git clone https://github.com/Choi-In-hyuk/StateVLA.git
cd StateVLA

# Create conda environment (recommended)
conda create -n statevla python=3.10
conda activate statevla

# Install dependencies
pip install -r requirements.txt

# Install StateVLA
pip install -e .

# Clone and install LIBERO (required for training/evaluation)
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git
cd LIBERO
pip install -e .
cd ..
```

## Quick Start

### Training

```bash
# Train with default config
python train.py --config conf/config.yaml --data_directory /path/to/your/data

# Resume from checkpoint
python train.py --config conf/config.yaml --checkpoint checkpoints/checkpoint_latest.pt
```

### Evaluation

```bash
python eval.py --checkpoint checkpoints/checkpoint_best.pt --data_directory /path/to/data
```

### Using StateVLA in Your Code

```python
from StateVLA import create_statevla_model, create_training_model

# Create model
model = create_statevla_model(
    camera_names=['agentview', 'eye_in_hand'],
    state_dim=256,
    action_dim=7,
    action_seq_len=10,
    device='cuda'
)

# For training
training_model, ema = create_training_model(model, use_ema=True)

# Forward pass (training)
outputs = training_model.forward(
    obs_dict=obs,
    prev_action=prev_action,
    gt_actions=actions
)
loss = outputs['loss']

# Inference
actions = training_model.predict(obs_dict, prev_action)
```

## Project Structure

```
StateVLA/
├── StateVLA/                    # Main package
│   ├── __init__.py
│   ├── statevla_model.py       # Core StateVLA model
│   ├── state_encoder.py        # State fusion module
│   ├── state_predictor.py      # Mamba-based state predictor
│   ├── action_policy.py        # Residual action policy
│   ├── model_factory.py        # Model creation utilities
│   ├── train_policy.py         # Training wrapper
│   ├── mamba/                  # Mamba backbone
│   ├── backbone/               # Vision & language encoders
│   └── utils/                  # Utilities
├── conf/
│   └── config.yaml             # Configuration file
├── train.py                    # Training script
├── eval.py                     # Evaluation script
├── dataloader.py               # Data loading utilities
├── requirements.txt
├── setup.py
└── README.md
```

## Configuration

Key configuration options in `conf/config.yaml`:

```yaml
model:
  state_dim: 256              # State latent dimension
  action_dim: 7               # Robot action dimension
  action_seq_len: 10          # Action sequence length
  state_predictor_layers: 4   # Mamba layers for state prediction
  policy_layers: 3            # Mamba layers for action policy
  use_correction: true        # Enable residual correction

training:
  batch_size: 64
  learning_rate: 1.0e-4
  action_loss_weight: 1.0     # Weight for action loss
  state_loss_weight: 0.1      # Weight for state prediction loss
  enable_ema: true
```

## Model Components

### 1. StateEncoder
Fuses multi-modal inputs (vision, language, previous action) into a unified state representation.

### 2. StatePredictor
Uses Mamba (State Space Model) to predict the next state from the current state.

### 3. ActionPolicy (Residual)
- **Base Policy**: Flow Matching generates base action from predicted next state
- **Correction MLP**: Adjusts action based on current state and prediction error
- **Final Action**: `a = a_base + Δa`

## Loss Functions

```
L_total = L_action + λ × L_state

where:
  L_action = Flow Matching loss (action prediction)
  L_state = MSE(z_{t+1}^pred, z_{t+1}^actual)
```

## Citation

```bibtex
@article{statevla2024,
  title={StateVLA: State-based Vision-Language-Action Model with Predictive Feedback},
  author={Your Name},
  year={2024}
}
```

## Acknowledgments

- Based on [MambaVLA](https://github.com/...) architecture
- Uses [Mamba](https://github.com/state-spaces/mamba) for efficient sequence modeling
- Flow Matching inspired by [Flow Matching for Generative Modeling](https://arxiv.org/abs/...)

## License

MIT License
