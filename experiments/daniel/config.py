# ============================================================
# DANIEL'S CONFIG — Hyperparameter Experiments
# ============================================================
# 10 Hyperparameter Experiments (IDs 21-30)
# Testing learning rate, gamma, batch size, epsilon decay, buffer size

MEMBER_NAME = "Daniel"

EXPERIMENTS = [
    {
        "id": 21,
        "name": "Daniel Baseline",
        "lr": 0.0001,
        "gamma": 0.99,
        "batch_size": 32,
        "eps_start": 1.0,
        "eps_end": 0.01,
        "eps_decay_fraction": 0.10,
        "buffer_size": 50000,
        "learning_starts": 10000,
        "noted_behavior": "Baseline with proper warmup (10k learning starts). Should show clear learning progression from -21 toward -15."
    },
    {
        "id": 22,
        "name": "Very Fast LR",
        "lr": 0.001,
        "gamma": 0.99,
        "batch_size": 32,
        "eps_start": 1.0,
        "eps_end": 0.01,
        "eps_decay_fraction": 0.10,
        "buffer_size": 50000,
        "learning_starts": 10000,
        "noted_behavior": "10x higher learning rate - likely to diverge or become unstable. Q-values may explode."
    },
    {
        "id": 23,
        "name": "Moderate Fast LR",
        "lr": 0.00025,
        "gamma": 0.99,
        "batch_size": 32,
        "eps_start": 1.0,
        "eps_end": 0.01,
        "eps_decay_fraction": 0.10,
        "buffer_size": 50000,
        "learning_starts": 10000,
        "noted_behavior": "2.5x baseline LR - should learn faster while remaining stable. Good candidate for best config."
    },
    {
        "id": 24,
        "name": "Extreme Short Horizon",
        "lr": 0.0001,
        "gamma": 0.85,
        "batch_size": 32,
        "eps_start": 1.0,
        "eps_end": 0.01,
        "eps_decay_fraction": 0.10,
        "buffer_size": 50000,
        "learning_starts": 10000,
        "noted_behavior": "gamma=0.85 - very short-sighted. Should perform poorly as agent cannot plan rallies longer than ~10 steps."
    },
    {
        "id": 25,
        "name": "Very Long Horizon",
        "lr": 0.0001,
        "gamma": 0.9995,
        "batch_size": 32,
        "eps_start": 1.0,
        "eps_end": 0.01,
        "eps_decay_fraction": 0.10,
        "buffer_size": 50000,
        "learning_starts": 10000,
        "noted_behavior": "gamma near 1.0 - values future rewards heavily. Should perform well but may need more steps to converge."
    },
    {
        "id": 26,
        "name": "Extra Large Batch",
        "lr": 0.0001,
        "gamma": 0.99,
        "batch_size": 256,
        "eps_start": 1.0,
        "eps_end": 0.01,
        "eps_decay_fraction": 0.10,
        "buffer_size": 100000,
        "learning_starts": 10000,
        "noted_behavior": "Very large batch - very stable gradients but slower learning per step. Should show smooth reward curve."
    },
    {
        "id": 27,
        "name": "Very Fast Epsilon Decay",
        "lr": 0.0001,
        "gamma": 0.99,
        "batch_size": 32,
        "eps_start": 1.0,
        "eps_end": 0.01,
        "eps_decay_fraction": 0.02,
        "buffer_size": 50000,
        "learning_starts": 10000,
        "noted_behavior": "Epsilon decays in first 2% of training (10k steps). Agent becomes greedy too early - may plateau at poor policy."
    },
    {
        "id": 28,
        "name": "Very Slow Epsilon Decay",
        "lr": 0.0001,
        "gamma": 0.99,
        "batch_size": 32,
        "eps_start": 1.0,
        "eps_end": 0.01,
        "eps_decay_fraction": 0.50,
        "buffer_size": 50000,
        "learning_starts": 10000,
        "noted_behavior": "Epsilon decays over 50% of training (250k steps). Explores too long - delayed learning but potentially better final policy."
    },
    {
        "id": 29,
        "name": "Large Buffer + Early Learning",
        "lr": 0.0001,
        "gamma": 0.99,
        "batch_size": 32,
        "eps_start": 1.0,
        "eps_end": 0.01,
        "eps_decay_fraction": 0.10,
        "buffer_size": 200000,
        "learning_starts": 5000,
        "noted_behavior": "Large buffer (200k) with early learning start (5k). More diverse replay experiences - should learn faster."
    },
    {
        "id": 30,
        "name": "Daniel Best Combined",
        "lr": 0.00025,
        "gamma": 0.99,
        "batch_size": 128,
        "eps_start": 1.0,
        "eps_end": 0.01,
        "eps_decay_fraction": 0.12,
        "buffer_size": 100000,
        "learning_starts": 10000,
        "noted_behavior": "Combination: moderate LR (2.5e-4), larger batch (128), balanced epsilon decay (12%). Expected best performance."
    },
]
