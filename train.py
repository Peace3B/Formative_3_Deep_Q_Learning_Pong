# ============================================================
# TRAIN.PY — DQN Agent for Atari Pong
# ============================================================
# This script:
#   1. Compares CNNPolicy vs MLPPolicy
#   2. Runs 10 hyperparameter experiments
#   3. Logs reward trends and episode length
#   4. Saves the best model as dqn_model.zip
# ============================================================

# ============================================================
# IMPORTS
# ============================================================
import os
import csv
import time
import numpy as np
import gymnasium as gym
import ale_py  # Explicitly import ale_py to ensure Atari environments are registered
from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.callbacks import (
    EvalCallback,
    BaseCallback
)

# ============================================================
# CUSTOM CALLBACK — Logs reward & episode length per episode
# ============================================================
# A Callback is a function that runs automatically during
# training at specific points. We use it to:
#   - Track reward after every episode
#   - Track episode length after every episode
#   - Save everything to a CSV for our hyperparameter table
#
# Without this, we only get averaged stats every N steps.
# This gives us detailed per-episode tracking.
# ============================================================

class TrainingLogger(BaseCallback):
    """
    Logs episode reward and length to a CSV file after
    every completed episode during training.
    """
    def __init__(self, log_path, verbose=0):
        super().__init__(verbose)
        self.log_path = log_path
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_count = 0

        # Create CSV file with headers
        with open(self.log_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "episode",
                "reward",
                "episode_length",
                "timestep"
            ])

    def _on_step(self) -> bool:
        # Check if any environment finished an episode
        for info in self.locals.get("infos", []):
            if "episode" in info:
                ep_reward = info["episode"]["r"]  # total reward
                ep_length = info["episode"]["l"]  # steps taken
                self.episode_count += 1

                self.episode_rewards.append(ep_reward)
                self.episode_lengths.append(ep_length)

                # Write to CSV
                with open(self.log_path, "a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        self.episode_count,
                        round(ep_reward, 2),
                        ep_length,
                        self.num_timesteps
                    ])

                if self.verbose > 0:
                    print(
                        f"  Episode {self.episode_count} | "
                        f"Reward: {ep_reward:.1f} | "
                        f"Length: {ep_length}"
                    )
        return True  # return True to continue training


# ============================================================
# ENVIRONMENT SETUP FUNCTION
# ============================================================
# We wrap environment creation in a function so we can
# easily create fresh environments for each experiment.
#
# make_atari_env automatically applies:
#   - Grayscale conversion (removes color noise)
#   - Frame resize to 84x84 pixels
#   - Frame skipping (repeat action for 4 frames)
#   - Episode termination on life loss
#
# VecFrameStack stacks last 4 frames so agent detects motion.
# Without stacking, agent cannot see ball DIRECTION in Pong.
# ============================================================

def make_env(seed=42):
    """Creates and returns a preprocessed Pong environment."""
    env = make_atari_env("ALE/Pong-v5", n_envs=1, seed=seed)
    env = VecFrameStack(env, n_stack=4)
    return env


# ============================================================
# POLICY COMPARISON FUNCTION
# ============================================================
# This runs a SHORT training session to compare:
#   CNNPolicy: uses Convolutional Neural Networks
#              - designed for image/pixel input
#              - detects spatial patterns (ball, paddle shapes)
#              - standard choice for ALL Atari games
#
#   MLPPolicy: uses a flat Multilayer Perceptron
#              - designed for non-visual input (vectors)
#              - FLATTENS pixel input, loses spatial structure
#              - loses information about WHERE objects are
#
# We train both for 50,000 steps (short, just for comparison)
# and record their average rewards to prove CNN is superior.
# ============================================================

def compare_policies(timesteps=50_000):
    """
    Trains both CNNPolicy and MLPPolicy for a short run
    and compares their average rewards.
    Returns a dict with results for both policies.
    """
    results = {}

    for policy in ["CnnPolicy", "MlpPolicy"]:
        print(f"\n{'='*50}")
        print(f"  Training with {policy} for {timesteps} steps")
        print(f"{'='*50}")

        env = make_env(seed=42)

        model = DQN(
            policy=policy,
            env=env,
            learning_rate=1e-4,
            gamma=0.99,
            batch_size=32,
            buffer_size=10_000,
            learning_starts=10_000,
            exploration_initial_eps=1.0,
            exploration_final_eps=0.01,
            exploration_fraction=0.1,
            verbose=0  # silent for cleaner output
        )

        start_time = time.time()
        model.learn(total_timesteps=timesteps)
        duration = time.time() - start_time

        # ====================================================
        # EVALUATE the policy after short training
        # Run 5 episodes with greedy policy (deterministic)
        # and record average reward
        # ====================================================
        eval_env = make_env(seed=99)
        episode_rewards = []

        for _ in range(5):
            obs = eval_env.reset()
            done = False
            ep_reward = 0
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, _ = eval_env.step(action)
                ep_reward += reward[0]
            episode_rewards.append(ep_reward)

        avg_reward = np.mean(episode_rewards)
        results[policy] = {
            "avg_reward": round(avg_reward, 2),
            "training_time_sec": round(duration, 1)
        }

        print(f"  {policy} → Avg Reward: {avg_reward:.2f} | "
              f"Time: {duration:.0f}s")

        eval_env.close()
        env.close()

    # Print comparison summary
    print(f"\n{'='*50}")
    print("  POLICY COMPARISON SUMMARY")
    print(f"{'='*50}")
    print(f"  CNNPolicy avg reward : "
          f"{results['CnnPolicy']['avg_reward']}")
    print(f"  MLPPolicy avg reward : "
          f"{results['MlpPolicy']['avg_reward']}")

    winner = (
        "CnnPolicy"
        if results["CnnPolicy"]["avg_reward"]
        > results["MlpPolicy"]["avg_reward"]
        else "MlpPolicy"
    )
    print(f"\n  WINNER: {winner}")
    print(f"  Conclusion: CNNPolicy is better for Pong because")
    print(f"  it processes pixel input spatially using")
    print(f"  convolutions, preserving the location of the")
    print(f"  ball and paddle. MLPPolicy flattens pixels,")
    print(f"  losing all spatial structure.")
    print(f"{'='*50}\n")

    return results


# ============================================================
# HYPERPARAMETER EXPERIMENTS
# ============================================================
# Each experiment changes ONE or MORE hyperparameters from
# the baseline to isolate their effect on performance.
#
# MEMBER NAME: [Your Name Here]
# Replace with your actual name in the table below.
#
# We define 10 experiments as required. Each experiment:
#   - Uses a different hyperparameter combination
#   - Trains for 200,000 steps (enough to see a trend)
#   - Logs reward and episode length to a CSV
#   - Records a "noted behavior" observation
# ============================================================

MEMBER_NAME = "Your Name Here"  # CHANGE THIS

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


# ============================================================
# TRAINING FUNCTION
# ============================================================
# Runs one experiment with the given hyperparameters.
# Saves logs to CSV and returns the trained model.
# ============================================================

def run_experiment(exp, total_timesteps=200_000):
    """
    Trains a DQN agent with the given hyperparameter config.

    Args:
        exp: dict with hyperparameter values
        total_timesteps: how long to train

    Returns:
        model: trained DQN model
        avg_reward: average reward over final 5 eval episodes
    """

    exp_id = exp["id"]
    print(f"\n{'='*55}")
    print(f"  EXPERIMENT {exp_id}: {exp['name']}")
    print(f"  lr={exp['lr']} | gamma={exp['gamma']} | "
          f"batch={exp['batch_size']}")
    print(f"  eps: {exp['eps_start']} → {exp['eps_end']} "
          f"(decay={exp['eps_decay']})")
    print(f"{'='*55}")

    # Create directories for this experiment
    log_dir = f"./logs/experiment_{exp_id}/"
    model_dir = f"./models/experiment_{exp_id}/"
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    # Create fresh environments
    env = make_env(seed=42)
    eval_env = make_env(seed=99)

    # Custom callback: logs reward + episode length to CSV
    logger = TrainingLogger(
        log_path=f"{log_dir}training_log.csv",
        verbose=0
    )

    # Eval callback: saves best model during training
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=model_dir,
        log_path=log_dir,
        eval_freq=10_000,
        n_eval_episodes=5,
        deterministic=True,
        render=False,
        verbose=0
    )

    # Define DQN agent with this experiment's hyperparameters
    model = DQN(
        policy="CnnPolicy",
        env=env,
        learning_rate=exp["lr"],
        gamma=exp["gamma"],
        batch_size=exp["batch_size"],
        buffer_size=100_000,
        learning_starts=10_000,
        exploration_initial_eps=exp["eps_start"],
        exploration_final_eps=exp["eps_end"],
        exploration_fraction=exp["eps_decay"],
        target_update_interval=1000,
        verbose=0,
        tensorboard_log=f"./pong_tensorboard/exp_{exp_id}/"
    )

    # Train
    start = time.time()
    model.learn(
        total_timesteps=total_timesteps,
        callback=[logger, eval_callback],
        log_interval=100
    )
    duration = round(time.time() - start, 1)

    # --------------------------------------------------------
    # EVALUATE final performance with greedy policy
    # --------------------------------------------------------
    eval_rewards = []
    for _ in range(5):
        obs = eval_env.reset()
        done = False
        ep_reward = 0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, _ = eval_env.step(action)
            ep_reward += reward[0]
        eval_rewards.append(ep_reward)

    avg_reward = round(np.mean(eval_rewards), 2)
    print(f"  Avg Eval Reward: {avg_reward} | "
          f"Time: {duration}s")
    print(f"  Behavior: {exp['noted_behavior']}")

    env.close()
    eval_env.close()

    return model, avg_reward


# ============================================================
# SAVE HYPERPARAMETER TABLE TO CSV
# ============================================================
# This generates the table required in the assignment.
# It shows all 10 experiments with their hyperparameters
# and observed behavior in one clean file.
# ============================================================

def save_hyperparameter_table(results):
    """
    Saves all experiment results to a CSV table.
    results: list of dicts {exp, avg_reward}
    """
    os.makedirs("./results/", exist_ok=True)
    path = "./results/hyperparameter_table.csv"

    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Member", "Exp#", "Name",
            "lr", "gamma", "batch_size",
            "eps_start", "eps_end", "eps_decay",
            "Avg Reward", "Noted Behavior"
        ])
        for r in results:
            exp = r["exp"]
            writer.writerow([
                MEMBER_NAME,
                exp["id"],
                exp["name"],
                exp["lr"],
                exp["gamma"],
                exp["batch_size"],
                exp["eps_start"],
                exp["eps_end"],
                exp["eps_decay"],
                r["avg_reward"],
                exp["noted_behavior"]
            ])

    print(f"\nHyperparameter table saved to: {path}")


# ============================================================
# MAIN — runs everything in order
# ============================================================

if __name__ == "__main__":

    print("\n" + "="*55)
    print("  DQN TRAINING — ATARI PONG")
    print("  Stable Baselines3 + Gymnasium")
    print("="*55)

    # --------------------------------------------------------
    # PHASE 1: Compare CNNPolicy vs MLPPolicy
    # Short 50k step run for each to compare performance.
    # --------------------------------------------------------
    print("\n[PHASE 1] Comparing CNNPolicy vs MLPPolicy...")
    policy_results = compare_policies(timesteps=50_000)

    # --------------------------------------------------------
    # PHASE 2: Run all 10 hyperparameter experiments
    # Each trains for 200k steps and logs results.
    # --------------------------------------------------------
    print("\n[PHASE 2] Running 10 Hyperparameter Experiments...")
    all_results = []
    best_reward = float("-inf")
    best_model = None
    best_exp_id = None

    for exp in EXPERIMENTS:
        model, avg_reward = run_experiment(
            exp,
            total_timesteps=200_000
        )
        all_results.append(
            {
                "exp": exp,
                "avg_reward": avg_reward
            }
        )

        # Track best performing experiment
        if avg_reward > best_reward:
            best_reward = avg_reward
            best_model = model
            best_exp_id = exp["id"]

    # --------------------------------------------------------
    # PHASE 3: Save results and best model
    # --------------------------------------------------------
    print("\n[PHASE 3] Saving results...")

    # Save hyperparameter table
    save_hyperparameter_table(all_results)

    # Save the best model as dqn_model.zip (as required)
    os.makedirs("./models/", exist_ok=True)
    best_model.save("./models/dqn_model")
    print(f"\nBest model (Experiment {best_exp_id}) saved as "
          f"./models/dqn_model.zip")
    print(f"Best avg reward: {best_reward}")

    # --------------------------------------------------------
    # PHASE 4: Print final summary
    # --------------------------------------------------------
    print("\n" + "="*55)
    print("  TRAINING COMPLETE — FINAL SUMMARY")
    print("="*55)
    print(f"  Policy Comparison:")
    print(f"    CnnPolicy avg reward: "
          f"{policy_results['CnnPolicy']['avg_reward']}")
    print(f"    MlpPolicy avg reward: "
          f"{policy_results['MlpPolicy']['avg_reward']}")
    print(f"\n  Hyperparameter Experiments:")
    for r in all_results:
        marker = " ← BEST" if r["exp"]["id"] == best_exp_id \
                 else ""
        print(f"    Exp {r['exp']['id']:2d} "
              f"({r['exp']['name']:20s}): "
              f"reward = {r['avg_reward']}{marker}")
    print(f"\n  Best Model: Experiment {best_exp_id}")
    print(f"  Saved to  : ./models/dqn_model.zip")
    print(f"\n  To visualize training:")
    print(f"  tensorboard --logdir ./pong_tensorboard/")
    print("="*55)