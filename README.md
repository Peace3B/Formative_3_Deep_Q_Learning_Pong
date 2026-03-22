# Formative 3: Deep Q-Learning on Atari Pong

Train and evaluate a DQN agent to play Atari Pong using Stable Baselines3 and Gymnasium.

## Objective

Implement hyperparameter tuning by running **10 distinct experiments per group member** and comparing DQN performance across different hyperparameter combinations.

## Project Structure

```
.
├── train.py              # Training script
├── play.py               # Evaluation/play script (Group 3 branded UI)
├── experiments/
│   ├── damour/
│   │   ├── config.py     # 10 experiments (complete)
│   │   ├── logs/         # training_log.csv per experiment
│   │   ├── models/       # best_model.zip per experiment
│   │   └── results/      # hyperparameter_table.csv
│   ├── daniel/
│   ├── peace/
│   └── musembi/
└── pyproject.toml
```

## Key Features

- **Policy Comparison**: Uses `CnnPolicy` for Atari visual inputs
- **Hyperparameter Tuning**: Each member runs 10 experiments with varied lr, gamma, batch_size, epsilon
- **Comprehensive Logging**: Episode rewards, lengths, timesteps saved to CSV
- **Model Evaluation**: `play.py` with greedy policy and customizable playback (15 FPS default)

---

## 1. Setup

```bash
uv venv
source .venv/bin/activate
uv sync
```

Or manually:

```bash
uv pip install 'stable-baselines3[extra]' 'gymnasium[atari]' 'autorom[accept-rom-license]' ale-py
autorom --accept-license
```

---

## 2. Hyperparameter Experiments (10 per Member)

Each member defines **10 experiments** in their config file.

### Config Template

```python
MEMBER_NAME = "YourName"
EXPERIMENTS = [
    {
        "id": 1,
        "name": "Baseline",
        "lr": 0.0001,
        "gamma": 0.99,
        "batch_size": 32,
        "eps_start": 1.0,
        "eps_end": 0.01,
        "eps_decay_fraction": 0.10,
        "buffer_size": 50000,
        "learning_starts": 10000,
        "noted_behavior": "Description of observed behavior"
    },
    # ... 9 more experiments
]
```

### Hyperparameter Tuning Dimensions

- **Learning Rate (lr)**: 0.00005 – 0.001
- **Gamma (discount factor)**: 0.85 – 0.9995
- **Batch Size**: 16 – 256
- **Epsilon Decay**: 0.02 – 0.50 (fraction of total timesteps)
- **Buffer Size**: 1000 – 200000
- **Learning Starts**: 100 – 50000 (timesteps before training begins)

### Output Table Format

`results/<member>/hyperparameter_table.csv`:

| Member | Exp# | Name     | lr     | gamma | batch_size | ... | Avg Reward | Noted Behavior     |
| ------ | ---- | -------- | ------ | ----- | ---------- | --- | ---------- | ------------------ |
| daniel | 1    | Baseline | 0.0001 | 0.99  | 32         | ... | -18.5      | Stable learning    |
| daniel | 2    | Fast LR  | 0.001  | 0.99  | 32         | ... | -21.0      | Unstable, diverges |

---

## 3. Main Scripts

### train.py — Training

Trains DQN agents per member's config.

**Usage:**

```bash
python train.py --member damour
```

**Output:**

- `logs/<member>/experiment_<id>/training_log.csv` — Episode data
- `models/<member>/experiment_<id>/best_model.zip` — Trained model
- `results/<member>/hyperparameter_table.csv` — Summary table

### play.py — Evaluation

Loads trained model and runs greedy evaluation.

**Usage:**

```bash
python play.py --model best_model.zip --episodes 5 --fps 15
```

**Options:**

- `--model`: Path to model (default: `best_model.zip`)
- `--episodes`: Number of play episodes (default: 5)
- `--fps`: Playback speed (default: 15, ALE-like pacing)
- `--no-render`: Disable rendering
- `--no-color`: Disable colored output

---

## 4. Run Training

### Interactive Menu

```bash
python main.py
```

### Train One Member

```bash
python train.py --member damour
```

Valid members: `damour`, `daniel`, `peace`, `musembi`

---

## 5. Outputs

Per-member structure:

- **Logs**: `experiments/<member>/logs/experiment_<id>/training_log.csv`
- **Models**: `experiments/<member>/models/experiment_<id>/best_model.zip`
- **Results**: `experiments/<member>/results/hyperparameter_table.csv`

---

## 6. Hyperparameter Tuning & Results

In this project, each team member ran 10 unique hyperparameter configurations to explore the impact on our DQN agent's performance in Atari Pong. We adjusted parameters such as the learning rate, discount factor (gamma), batch size, and epsilon decay rates.

Here is a summary of our team's findings:

- **Learning Rate Impact:** We found that moderating the learning rate around `0.0001` to `0.00025` provided the best balance of convergence speed and stability. Extremely high learning rates often led to divergence and erratic reward curves.
- **Gamma (Discount Factor):** Higher gamma values (e.g., `0.999` vs `0.99`) allowed the agent to optimize for longer-term returns, which is critical in Pong where the reward is delayed until a point is scored. However, this occasionally slowed down early-stage learning.
- **Batch Size Effect:** Increasing the batch size to `64` or `128` stabilized the neural network's gradient updates and reduced variance across reward episodes, paring well with moderate learning rates.
- **Exploration vs Exploitation:** A steady, well-calibrated epsilon decay allowed our agents to explore the environment state space thoroughly before fully committing to their learned policies, avoiding getting stuck in sub-optimal local minima.

_Detailed numerical results for every experiment can be found directly within each member's directory:_ `experiments/<member>/results/hyperparameter_table.csv`.

---

## 7. Project Structure & Contributions

This repository contains our team's complete training code, model artifacts, and evaluation scripts.

- `train.py` & `play.py`: Our core scripts for training continuous DQN agents and displaying their evaluation through a custom Pygame user interface.
- **Experiments Directory:** Features individual workspaces for `damour`, `daniel`, `peace`, and `musembi`.
  - Each contains a `config.py` governing their 10 distinct hyperparameter experiments.
  - Generates respective Tensorboard logs, best model checkpoints (`.zip`), and structured CSV results matrices.

---

## 8. Video Demonstration

Our submission video demonstrates our best-trained RL agent successfully playing Atari Pong, alongside a brief overview of our environment setup and a discussion of our hyperparameter tuning strategy.

**Watch our project video here:**

<video src="video.mp4" controls="controls" width="100%">
</video>
