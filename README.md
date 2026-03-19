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
└── pyproject.toml
```

## What This Project Does

## 1. Setup

From the project root:

```bash
uv venv
source .venv/bin/activate
uv sync
```

If you need to install dependencies manually:

```bash
uv pip install 'stable-baselines3[extra]' 'gymnasium[atari]' ale-py 'autorom[accept-rom-license]' numpy
```

Notes:

## 2. Team Workflow (Per Member)

Each member config lives in:

Each config file must define:

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

Members who have not added experiments should keep:

```python
EXPERIMENTS = []
```

## 3. Run Training

### Interactive menu

```bash
python main.py
```

This lets you choose:

### Run one member directly

```bash
python main.py damour
```

or

```bash
python train.py --member damour
```

Valid members:

## 4. Output Locations (Grouped by Member)

For member `damour`, outputs are written to:

Same structure applies to other members.

## 5. Run in Notebook (Jupyter)

If you want to run this project from a notebook, use this flow.

### Cell 1: Import

```python
from train import run_training
```

### Cell 2: Choose member and run

```python
member = "damour"  # replace with: "daniel", "peace", or "musembi"
result = run_training(member_name=member)
result
```

### What to replace before running in notebook

- `MEMBER_NAME = "Daniel"` if needed
- `EXPERIMENTS = []` with your experiment list.

If `EXPERIMENTS` is empty, training is skipped by design.

## 6. Optional: TensorBoard

```bash
tensorboard --logdir ./pong_tensorboard/
```

Then open the local TensorBoard URL in your browser.

## 7. Troubleshooting

### `zsh: no matches found: stable-baselines3[extra]`

Use quotes:

```bash
uv pip install 'stable-baselines3[extra]'
```

### `ModuleNotFoundError` (for `numpy`, `ale_py`, `stable_baselines3`)

```bash
source .venv/bin/activate
```

```bash
uv sync
```

# Formative 3: Deep Q-Learning for Atari Pong

### `zsh: no matches found: stable-baselines3[extra]`

Use single quotes around bracket packages:

```bash
uv pip install 'stable-baselines3[extra]' 'gymnasium[atari]' 'gymnasium[accept-rom-license]' ale-py autorom numpy
```

### `ModuleNotFoundError: No module named 'numpy'`

Install numpy inside the active `.venv`:

```bash
uv pip install numpy
```

### Member skipped / no experiments

Add experiments to:

- `experiments/daniel/config.py`
- `experiments/peace/config.py`
- `experiments/musembi/config.py`

## 8. Team Workflow Suggestion

1. Each person edits only their own `experiments/<name>/config.py`.
2. Run your own training first: `python main.py <name>`.
3. When everyone is ready, run all members from menu in `main.py`.
