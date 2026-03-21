# ============================================================
# MUSEMBI'S CONFIG — Hyperparameter Experiments
# ============================================================
# Structured to match the assignment requirement of 10 experiments.

MEMBER_NAME = "Musembi"

EXPERIMENTS = [
	# --------------------------------------------------------
	# Experiment 1 — BASELINE
	# --------------------------------------------------------
	{
		"id": 1,
		"name": "Baseline",
		"lr": 1e-4,
		"gamma": 0.99,
		"batch_size": 32,
		"eps_start": 1.0,
		"eps_end": 0.01,
		"eps_decay": 0.1,
		"noted_behavior": (
			"Stable baseline run. Agent starts near random performance and "
			"gradually improves over training."
		),
	},
	# --------------------------------------------------------
	# Experiment 2 — HIGH LEARNING RATE
	# --------------------------------------------------------
	{
		"id": 2,
		"name": "High LR",
		"lr": 1e-3,
		"gamma": 0.99,
		"batch_size": 32,
		"eps_start": 1.0,
		"eps_end": 0.01,
		"eps_decay": 0.1,
		"noted_behavior": (
			"Updates are aggressive; reward tends to fluctuate and learning "
			"can become unstable."
		),
	},
	# --------------------------------------------------------
	# Experiment 3 — LOW LEARNING RATE
	# --------------------------------------------------------
	{
		"id": 3,
		"name": "Low LR",
		"lr": 1e-5,
		"gamma": 0.99,
		"batch_size": 32,
		"eps_start": 1.0,
		"eps_end": 0.01,
		"eps_decay": 0.1,
		"noted_behavior": (
			"Learning is slow and conservative; may require far more timesteps "
			"to reach strong performance."
		),
	},
	# --------------------------------------------------------
	# Experiment 4 — LOW GAMMA
	# --------------------------------------------------------
	{
		"id": 4,
		"name": "Low Gamma",
		"lr": 1e-4,
		"gamma": 0.80,
		"batch_size": 32,
		"eps_start": 1.0,
		"eps_end": 0.01,
		"eps_decay": 0.1,
		"noted_behavior": (
			"Agent becomes short-sighted by prioritizing immediate rewards, "
			"often hurting long-term rally strategy."
		),
	},
	# --------------------------------------------------------
	# Experiment 5 — MEDIUM GAMMA
	# --------------------------------------------------------
	{
		"id": 5,
		"name": "Medium Gamma",
		"lr": 1e-4,
		"gamma": 0.95,
		"batch_size": 32,
		"eps_start": 1.0,
		"eps_end": 0.01,
		"eps_decay": 0.1,
		"noted_behavior": (
			"Balances immediate and future reward, usually better than very low "
			"gamma but often below 0.99 in Atari tasks."
		),
	},
	# --------------------------------------------------------
	# Experiment 6 — LARGE BATCH SIZE
	# --------------------------------------------------------
	{
		"id": 6,
		"name": "Large Batch",
		"lr": 1e-4,
		"gamma": 0.99,
		"batch_size": 128,
		"eps_start": 1.0,
		"eps_end": 0.01,
		"eps_decay": 0.1,
		"noted_behavior": (
			"Gradient estimates are smoother and often more stable, but learning "
			"may progress slower early on."
		),
	},
	# --------------------------------------------------------
	# Experiment 7 — SMALL BATCH SIZE
	# --------------------------------------------------------
	{
		"id": 7,
		"name": "Small Batch",
		"lr": 1e-4,
		"gamma": 0.99,
		"batch_size": 16,
		"eps_start": 1.0,
		"eps_end": 0.01,
		"eps_decay": 0.1,
		"noted_behavior": (
			"Noisier updates can improve early adaptation but often increase "
			"reward variance across episodes."
		),
	},
	# --------------------------------------------------------
	# Experiment 8 — FAST EPSILON DECAY
	# --------------------------------------------------------
	{
		"id": 8,
		"name": "Fast Epsilon Decay",
		"lr": 1e-4,
		"gamma": 0.99,
		"batch_size": 32,
		"eps_start": 1.0,
		"eps_end": 0.01,
		"eps_decay": 0.05,
		"noted_behavior": (
			"Exploration ends quickly; agent may prematurely exploit weak "
			"strategies and plateau early."
		),
	},
	# --------------------------------------------------------
	# Experiment 9 — SLOW EPSILON DECAY
	# --------------------------------------------------------
	{
		"id": 9,
		"name": "Slow Epsilon Decay",
		"lr": 1e-4,
		"gamma": 0.99,
		"batch_size": 32,
		"eps_start": 1.0,
		"eps_end": 0.01,
		"eps_decay": 0.5,
		"noted_behavior": (
			"Long exploration phase improves state coverage, but reward gains "
			"usually appear later in training."
		),
	},
	# --------------------------------------------------------
	# Experiment 10 — BEST COMBINED CANDIDATE
	# --------------------------------------------------------
	{
		"id": 10,
		"name": "Best Combined",
		"lr": 5e-4,
		"gamma": 0.99,
		"batch_size": 64,
		"eps_start": 1.0,
		"eps_end": 0.01,
		"eps_decay": 0.1,
		"noted_behavior": (
			"Candidate best mix of faster convergence and stable gradients; "
			"intended to produce the strongest final reward."
		),
	},
]
