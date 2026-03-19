# Formative 3: Deep Q-Learning on Atari Pong

Train and compare DQN experiments for Atari Pong, grouped by team member.

## Project Structure

```text
.
├── main.py
├── train.py
├── experiments/
│   ├── damour/config.py
│   ├── daniel/config.py
│   ├── peace/config.py
│   └── musembi/config.py
├── logs/
├── models/
├── results/
└── pyproject.toml
```

## What This Project Does

- Compares `CnnPolicy` vs `MlpPolicy`.
- Runs member-specific hyperparameter experiments.
- Logs episode reward and episode length.
- Saves best model per member as `dqn_model.zip`.

## 1. Setup

From project root:

```bash
uv venv
source .venv/bin/activate
uv sync
```

If you prefer manual install:

```bash
uv pip install 'stable-baselines3[extra]' 'gymnasium[atari]' ale-py 'autorom[accept-rom-license]' numpy
```

Notes:

- Use single quotes for packages with `[]` in zsh.
- If ROM prompts appear, run: `AutoROM --accept-license`.
- Atari training is compute-heavy and can take a long time.

## 2. Member Configs

Each member edits only their own config file:

- `experiments/damour/config.py`
- `experiments/daniel/config.py`
- `experiments/peace/config.py`
- `experiments/musembi/config.py`

Each config must define:

```python
MEMBER_NAME = "YourName"
EXPERIMENTS = [
    {
        "id": 1,
        "name": "Baseline",
        "lr": 1e-4,
        "gamma": 0.99,
        "batch_size": 32,
        "eps_start": 1.0,
        "eps_end": 0.01,
        "eps_decay": 0.1,
        "noted_behavior": "Your observation"
    }
]
```

If a member has not started yet, keep:

```python
EXPERIMENTS = []
```

The trainer will skip empty configs safely.

## 3. Run Training

### Interactive Menu

```bash
python main.py
```

Choose one member or all members (sequential).

### Run One Member Directly

```bash
python main.py damour
```

or

```bash
python train.py --member damour
```

Valid members:

- `damour`
- `daniel`
- `peace`
- `musembi`

## 4. Outputs (Grouped by Member)

For `damour`:

- Logs: `logs/damour/experiment_<id>/training_log.csv`
- Model checkpoints: `models/damour/experiment_<id>/`
- Best final model: `models/damour/dqn_model.zip`
- Result table: `results/damour/hyperparameter_table.csv`
- TensorBoard: `pong_tensorboard/damour/`

Other members follow the same structure.

## 5. Run from Notebook (Jupyter)

### Cell 1

```python
from train import run_training
```

### Cell 2

```python
member = "damour"  # replace with: "daniel", "peace", or "musembi"
result = run_training(member_name=member)
result
```

### What You Replace in Notebook

- Replace `member = "damour"` with your name.
- In `experiments/<member>/config.py`, replace `EXPERIMENTS = []` with your experiment list.
- Optionally edit hyperparameters in that config file.

## 6. TensorBoard

```bash
tensorboard --logdir ./pong_tensorboard/
```

## 7. Troubleshooting

### `zsh: no matches found: stable-baselines3[extra]`

Use quotes around bracket extras:

```bash
uv pip install 'stable-baselines3[extra]' 'gymnasium[atari]' 'autorom[accept-rom-license]'
```

### `ModuleNotFoundError` for `numpy`, `ale_py`, or `stable_baselines3`

```bash
source .venv/bin/activate
uv sync
```

### Member is skipped

This means `EXPERIMENTS = []` in that member config. Add experiments in:

- `experiments/daniel/config.py`
- `experiments/peace/config.py`
- `experiments/musembi/config.py`
