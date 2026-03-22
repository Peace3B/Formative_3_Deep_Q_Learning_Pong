# ============================================================
# PEACE'S CONFIG — Hyperparameter Experiments
# ============================================================
# 10 Hyperparameter Experiments (IDs 11-20)
# Testing learning rate, gamma, batch size, epsilon decay, buffer size

MEMBER_NAME = "Peace"

EXPERIMENTS = [
    {
        "id": 11,
        "name": "M2 Baseline",
        "lr": 1e-4,
        "gamma": 0.99,
        "batch_size": 32,
        "eps_start": 1.0,
        "eps_end": 0.01,
        "eps_decay_fraction": 0.10,
        "buffer_size": 10000,
        "learning_starts": 5000,
        "noted_behavior": "Proper baseline with 500k steps and correct warmup. Agent begins improving after ~50k steps once replay buffer is filled. Reward rises from -21 toward -18 to -15 range. This is the reference point for Member 2 comparisons."
    },
    {
        "id": 12,
        "name": "Very Low LR",
        "lr": 5e-5,
        "gamma": 0.99,
        "batch_size": 32,
        "eps_start": 1.0,
        "eps_end": 0.01,
        "eps_decay_fraction": 0.10,
        "buffer_size": 1000,
        "learning_starts": 500,
        "noted_behavior": "Half the standard learning rate. Weight updates are too small to accumulate meaningful change in 500k steps. Loss decreases very slowly, reward barely moves above -21. Demonstrates that lr=1e-4 is already near the lower limit for Atari in this timestep budget."
    },
    {
        "id": 13,
        "name": "Aggressive LR",
        "lr": 5e-4,
        "gamma": 0.99,
        "batch_size": 32,
        "eps_start": 1.0,
        "eps_end": 0.01,
        "eps_decay_fraction": 0.10,
        "buffer_size": 1000,
        "learning_starts": 500,
        "noted_behavior": "5x higher learning rate causes rapid but unstable updates. Reward initially climbs faster than baseline but then oscillates significantly. Q-values likely overestimate due to large gradient steps. Shows classic instability of too-high LR in off-policy learning."
    },
    {
        "id": 14,
        "name": "Very Short Horizon",
        "lr": 1e-4,
        "gamma": 0.90,
        "batch_size": 32,
        "eps_start": 1.0,
        "eps_end": 0.01,
        "eps_decay_fraction": 0.10,
        "buffer_size": 1000,
        "learning_starts": 500,
        "noted_behavior": "gamma=0.90 makes future rewards decay steeply. A reward 10 steps away is worth only 35% of its value. Pong rallies last 50-200+ steps, so the agent struggles to learn defensive positioning. Reward ceiling notably lower than baseline."
    },
    {
        "id": 15,
        "name": "Large Batch + Higher LR",
        "lr": 2.5e-4,
        "gamma": 0.99,
        "batch_size": 128,
        "eps_start": 1.0,
        "eps_end": 0.01,
        "eps_decay_fraction": 0.10,
        "buffer_size": 1000,
        "learning_starts": 500,
        "noted_behavior": "Paired increase of batch size and learning rate. Larger batch produces lower-variance gradients; higher LR restores the effective update speed. Reward curve is noticeably smoother than the baseline and often achieves a higher final reward."
    },
    {
        "id": 16,
        "name": "Tiny Batch",
        "lr": 1e-4,
        "gamma": 0.99,
        "batch_size": 16,
        "eps_start": 1.0,
        "eps_end": 0.01,
        "eps_decay_fraction": 0.10,
        "buffer_size": 1000,
        "learning_starts": 500,
        "noted_behavior": "batch=16 produces extremely noisy gradients. Reward curve is highly erratic with large swings between episodes. The agent does make some progress but convergence is unreliable. Contrast with Exp 15: smaller batch requires much lower LR to stabilize."
    },
    {
        "id": 17,
        "name": "Fast Epsilon + Big Buffer",
        "lr": 1e-4,
        "gamma": 0.99,
        "batch_size": 32,
        "eps_start": 1.0,
        "eps_end": 0.01,
        "eps_decay_fraction": 0.05,
        "buffer_size": 1000,
        "learning_starts": 500,
        "noted_behavior": "Epsilon decays in the first 25k steps (5% of training). Agent locks into greedy policy very early, before it has learned reliable strategies. Reward plateaus earlier than baseline."
    },
    {
        "id": 18,
        "name": "Slow Epsilon Decay",
        "lr": 1e-4,
        "gamma": 0.99,
        "batch_size": 32,
        "eps_start": 1.0,
        "eps_end": 0.01,
        "eps_decay_fraction": 0.40,
        "buffer_size": 1000,
        "learning_starts": 500,
        "noted_behavior": "Epsilon decays over 40% of training. Agent remains highly exploratory for the first 200k steps, resulting in flat reward. However the diverse replay buffer contents lead to more stable later learning. Shows exploration-exploitation tradeoff clearly."
    },
    {
        "id": 19,
        "name": "Early Learning Start",
        "lr": 1e-4,
        "gamma": 0.99,
        "batch_size": 32,
        "eps_start": 1.0,
        "eps_end": 0.01,
        "eps_decay_fraction": 0.10,
        "buffer_size": 1000,
        "learning_starts": 100,
        "noted_behavior": "Learning starts immediately after 100 transitions. Early updates are very noisy so buffer is not diverse enough causing unstable early loss. Performance eventually stabilizes but initial training is much noisier than baseline."
    },
    {
        "id": 20,
        "name": "M2 Best Combined",
        "lr": 2.5e-4,
        "gamma": 0.99,
        "batch_size": 128,
        "eps_start": 1.0,
        "eps_end": 0.01,
        "eps_decay_fraction": 0.15,
        "buffer_size": 1000,
        "learning_starts": 500,
        "noted_behavior": "Combines all best findings: lr=2.5e-4 (fast learning), batch=128 (stable gradients), eps decay over 15% of training (balanced exploration). Reward climbs faster than baseline and reaches a higher ceiling."
    },
]
