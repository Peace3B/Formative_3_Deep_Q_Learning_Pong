# ============================================================
# DAMOUR'S CONFIG — Hyperparameter Experiments
# ============================================================

MEMBER_NAME = "Damour"

EXPERIMENTS = [
    # --------------------------------------------------------
    # Experiment 1 — BASELINE
    # Standard DQN values from the original DeepMind paper.
    # This is our reference point for all comparisons.
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
            "Stable learning. Agent slowly improves. "
            "Reward starts at -21 and gradually increases. "
            "Good reference point."
        )
    },
    # --------------------------------------------------------
    # Experiment 2 — HIGH LEARNING RATE
    # Effect: Network updates are too aggressive.
    # Expected: Reward fluctuates wildly, unstable training.
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
            "Unstable training. High LR causes the network "
            "to overshoot optimal weights. Reward oscillates "
            "and does not converge steadily."
        )
    },
    # --------------------------------------------------------
    # Experiment 3 — LOW LEARNING RATE
    # Effect: Network updates are too small.
    # Expected: Very slow improvement, may not learn much
    # within 200k steps.
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
            "Very slow learning. Agent barely improves within "
            "200k steps. LR too low to make meaningful weight "
            "updates. Would need millions of steps."
        )
    },
    # --------------------------------------------------------
    # Experiment 4 — LOW GAMMA (short-sighted agent)
    # Effect: Agent discounts future rewards heavily.
    # Expected: Agent focuses on immediate rewards only,
    # poor long-term strategy in Pong.
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
            "Short-sighted behavior. Agent struggles to plan "
            "multi-step strategies. Reward improvement is "
            "slower than baseline. Future rewards barely "
            "influence current decisions."
        )
    },
    # --------------------------------------------------------
    # Experiment 5 — MEDIUM GAMMA
    # Effect: Moderate future reward consideration.
    # Expected: Slightly worse than 0.99 but better than 0.80.
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
            "Moderate future planning. Performance between "
            "low gamma and baseline. Agent shows reasonable "
            "improvement but not as good as gamma=0.99."
        )
    },
    # --------------------------------------------------------
    # Experiment 6 — LARGE BATCH SIZE
    # Effect: More stable gradient updates per step.
    # Expected: More stable but slower per-update learning.
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
            "More stable gradient updates. Less noisy reward "
            "curve than baseline. Slightly slower improvement "
            "per timestep but more consistent progress."
        )
    },
    # --------------------------------------------------------
    # Experiment 7 — SMALL BATCH SIZE
    # Effect: Noisier but more frequent gradient updates.
    # Expected: Faster early improvement but more variance.
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
            "Noisy reward curve. Smaller batch means high "
            "variance in gradient updates. Some early speed "
            "gains but unstable overall compared to baseline."
        )
    },
    # --------------------------------------------------------
    # Experiment 8 — FAST EPSILON DECAY
    # Effect: Agent stops exploring very early.
    # Expected: Agent gets stuck in suboptimal strategy
    # because it exploits too early before learning enough.
    # --------------------------------------------------------
    {
        "id": 8,
        "name": "Fast Epsilon Decay",
        "lr": 1e-4,
        "gamma": 0.99,
        "batch_size": 32,
        "eps_start": 1.0,
        "eps_end": 0.01,
        "eps_decay": 0.05,  # decays in first 5% of steps
        "noted_behavior": (
            "Agent exploits too early. Stops exploring before "
            "learning good strategies. Gets stuck in a "
            "suboptimal policy. Lower final reward than "
            "baseline."
        )
    },
    # --------------------------------------------------------
    # Experiment 9 — SLOW EPSILON DECAY
    # Effect: Agent explores for much longer.
    # Expected: Better exploration of state space but slower
    # to start exploiting and improving reward.
    # --------------------------------------------------------
    {
        "id": 9,
        "name": "Slow Epsilon Decay",
        "lr": 1e-4,
        "gamma": 0.99,
        "batch_size": 32,
        "eps_start": 1.0,
        "eps_end": 0.01,
        "eps_decay": 0.5,  # decays over first 50% of steps
        "noted_behavior": (
            "More thorough exploration. Agent takes longer "
            "to start improving reward because it stays "
            "random for longer. May lead to better final "
            "policy given enough timesteps."
        )
    },
    # --------------------------------------------------------
    # Experiment 10 — BEST COMBINED CONFIG
    # Effect: Combines best settings found across experiments.
    # Expected: Best overall performance.
    # This is the config we use for the final model.
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
            "Best configuration. Slightly higher LR speeds "
            "up convergence. Larger batch size stabilizes "
            "gradients. Combined effect produces the highest "
            "and most consistent reward improvement."
        )
    },
]
