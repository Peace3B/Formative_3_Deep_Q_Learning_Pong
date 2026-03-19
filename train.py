# ============================================================
# TRAIN.PY — DQN Agent for Atari Pong
# ============================================================
# This script:
#   1. Compares CNNPolicy vs MLPPolicy
#   2. Runs hyperparameter experiments from experiments/<member>/config.py
#   3. Logs reward trends and episode length
#   4. Saves the best model as dqn_model.zip
# ============================================================

import csv
import importlib.util
import os
import sys
import time

import ale_py  # Keep import to ensure Atari env registration
import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack


MEMBER_CHOICES = ["damour", "daniel", "peace", "musembi"]


class TrainingLogger(BaseCallback):
    """Logs episode reward and length to CSV after each completed episode."""

    def __init__(self, log_path: str, verbose: int = 0) -> None:
        super().__init__(verbose)
        self.log_path = log_path
        self.episode_count = 0

        with open(self.log_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["episode", "reward", "episode_length", "timestep"])

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for info in infos:
            if "episode" not in info:
                continue

            ep_reward = float(info["episode"]["r"])
            ep_length = int(info["episode"]["l"])
            self.episode_count += 1

            with open(self.log_path, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(
                    [self.episode_count, round(ep_reward, 2), ep_length, self.num_timesteps]
                )

            if self.verbose > 0:
                print(
                    f"  Episode {self.episode_count} | "
                    f"Reward: {ep_reward:.1f} | Length: {ep_length}"
                )
        return True


def make_env(seed: int = 42):
    """Create and return a preprocessed Atari Pong environment."""
    env = make_atari_env("ALE/Pong-v5", n_envs=1, seed=seed)
    return VecFrameStack(env, n_stack=4)


def _run_eval_episodes(model: DQN, eval_env, episodes: int = 5) -> float:
    """Evaluate model over several episodes and return average reward."""
    rewards = []

    for _ in range(episodes):
        obs = eval_env.reset()
        done = np.array([False])
        ep_reward = 0.0

        while not bool(np.any(done)):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, _ = eval_env.step(action)
            ep_reward += float(reward[0])

        rewards.append(ep_reward)

    return float(np.mean(rewards))


def compare_policies(member_name: str, timesteps: int = 50_000):
    """Train and compare CnnPolicy vs MlpPolicy on a short run."""
    results = {}

    for policy in ["CnnPolicy", "MlpPolicy"]:
        print(f"\n{'=' * 50}")
        print(f"  [{member_name.upper()}] Training with {policy} for {timesteps} steps")
        print(f"{'=' * 50}")

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
            verbose=0,
        )

        start_time = time.time()
        model.learn(total_timesteps=timesteps)
        duration = time.time() - start_time

        eval_env = make_env(seed=99)
        avg_reward = _run_eval_episodes(model=model, eval_env=eval_env, episodes=5)
        results[policy] = {
            "avg_reward": round(avg_reward, 2),
            "training_time_sec": round(duration, 1),
        }

        print(f"  {policy} -> Avg Reward: {avg_reward:.2f} | Time: {duration:.0f}s")

        eval_env.close()
        env.close()

    print(f"\n{'=' * 50}")
    print(f"  [{member_name.upper()}] POLICY COMPARISON SUMMARY")
    print(f"{'=' * 50}")
    print(f"  CnnPolicy avg reward: {results['CnnPolicy']['avg_reward']}")
    print(f"  MlpPolicy avg reward: {results['MlpPolicy']['avg_reward']}")

    return results


def load_member_config(member_name: str):
    """Load MEMBER_NAME and EXPERIMENTS from experiments/<member>/config.py."""
    member_key = member_name.lower()
    config_path = f"./experiments/{member_key}/config.py"

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config not found for '{member_key}': {config_path}")

    spec = importlib.util.spec_from_file_location(f"exp_config_{member_key}", config_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load config module for '{member_key}'")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    loaded_name = getattr(module, "MEMBER_NAME", member_key.capitalize())
    loaded_experiments = getattr(module, "EXPERIMENTS", [])

    if not isinstance(loaded_experiments, list):
        raise TypeError("EXPERIMENTS must be a list in the member config.")

    return loaded_name, loaded_experiments


def run_experiment(exp: dict, member_name: str, total_timesteps: int = 200_000):
    """Run one DQN experiment and return (model, avg_reward)."""
    exp_id = exp["id"]
    member_lower = member_name.lower()

    print(f"\n{'=' * 55}")
    print(f"  [{member_name.upper()}] EXPERIMENT {exp_id}: {exp['name']}")
    print(f"  lr={exp['lr']} | gamma={exp['gamma']} | batch={exp['batch_size']}")
    print(f"  eps: {exp['eps_start']} -> {exp['eps_end']} (decay={exp['eps_decay']})")
    print(f"{'=' * 55}")

    log_dir = f"./logs/{member_lower}/experiment_{exp_id}/"
    model_dir = f"./models/{member_lower}/experiment_{exp_id}/"
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    env = make_env(seed=42)
    eval_env = make_env(seed=99)

    logger = TrainingLogger(log_path=f"{log_dir}training_log.csv", verbose=0)
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=model_dir,
        log_path=log_dir,
        eval_freq=10_000,
        n_eval_episodes=5,
        deterministic=True,
        render=False,
        verbose=0,
    )

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
        tensorboard_log=f"./pong_tensorboard/{member_lower}/exp_{exp_id}/",
    )

    start = time.time()
    model.learn(total_timesteps=total_timesteps, callback=[logger, eval_callback], log_interval=100)
    duration = round(time.time() - start, 1)

    avg_reward = round(_run_eval_episodes(model=model, eval_env=eval_env, episodes=5), 2)
    print(f"  Avg Eval Reward: {avg_reward} | Time: {duration}s")
    print(f"  Behavior: {exp.get('noted_behavior', 'N/A')}")

    env.close()
    eval_env.close()

    return model, avg_reward


def save_hyperparameter_table(results: list, member_name: str):
    """Save experiment table under results/<member>/hyperparameter_table.csv."""
    member_lower = member_name.lower()
    os.makedirs(f"./results/{member_lower}/", exist_ok=True)
    path = f"./results/{member_lower}/hyperparameter_table.csv"

    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "Member",
                "Exp#",
                "Name",
                "lr",
                "gamma",
                "batch_size",
                "eps_start",
                "eps_end",
                "eps_decay",
                "Avg Reward",
                "Noted Behavior",
            ]
        )
        for item in results:
            exp = item["exp"]
            writer.writerow(
                [
                    member_name,
                    exp["id"],
                    exp["name"],
                    exp["lr"],
                    exp["gamma"],
                    exp["batch_size"],
                    exp["eps_start"],
                    exp["eps_end"],
                    exp["eps_decay"],
                    item["avg_reward"],
                    exp.get("noted_behavior", ""),
                ]
            )

    print(f"\nHyperparameter table saved to: {path}")


def run_training(member_name: str = "damour"):
    """Run the full training pipeline for one member."""
    loaded_name, experiments = load_member_config(member_name)
    member_lower = member_name.lower()

    if len(experiments) == 0:
        print("\n" + "=" * 55)
        print(f"  [{loaded_name.upper()}] NO EXPERIMENTS CONFIGURED")
        print("=" * 55)
        print(f"\n  {loaded_name} has not yet defined experiments.")
        print(f"  Add experiments to: ./experiments/{member_lower}/config.py")
        print("=" * 55 + "\n")
        return None

    print("\n" + "=" * 55)
    print(f"  DQN TRAINING — ATARI PONG [{loaded_name.upper()}]")
    print("  Stable Baselines3 + Gymnasium")
    print("=" * 55)

    print("\n[PHASE 1] Comparing CNNPolicy vs MLPPolicy...")
    policy_results = compare_policies(member_name=loaded_name, timesteps=50_000)

    print("\n[PHASE 2] Running Hyperparameter Experiments...")
    all_results = []
    best_reward = float("-inf")
    best_model = None
    best_exp_id = None

    for exp in experiments:
        model, avg_reward = run_experiment(exp=exp, member_name=loaded_name, total_timesteps=200_000)
        all_results.append({"exp": exp, "avg_reward": avg_reward})

        if avg_reward > best_reward:
            best_reward = avg_reward
            best_model = model
            best_exp_id = exp["id"]

    print("\n[PHASE 3] Saving results...")
    save_hyperparameter_table(results=all_results, member_name=loaded_name)

    os.makedirs(f"./models/{member_lower}/", exist_ok=True)
    if best_model is not None:
        best_model.save(f"./models/{member_lower}/dqn_model")

    print(f"\nBest model (Experiment {best_exp_id}) saved as ./models/{member_lower}/dqn_model.zip")
    print(f"Best avg reward: {best_reward}")

    print("\n" + "=" * 55)
    print(f"  TRAINING COMPLETE [{loaded_name.upper()}]")
    print("=" * 55)
    print("  Policy Comparison:")
    print(f"    CnnPolicy avg reward: {policy_results['CnnPolicy']['avg_reward']}")
    print(f"    MlpPolicy avg reward: {policy_results['MlpPolicy']['avg_reward']}")
    print("\n  Hyperparameter Experiments:")
    for item in all_results:
        marker = " <- BEST" if item["exp"]["id"] == best_exp_id else ""
        print(
            f"    Exp {item['exp']['id']:2d} ({item['exp']['name']:20s}): "
            f"reward = {item['avg_reward']}{marker}"
        )
    print(f"\n  Best Model: Experiment {best_exp_id}")
    print(f"  Saved to  : ./models/{member_lower}/dqn_model.zip")
    print("\n  To visualize training:")
    print(f"  tensorboard --logdir ./pong_tensorboard/{member_lower}/")
    print("=" * 55 + "\n")

    return {
        "member": loaded_name,
        "best_reward": best_reward,
        "best_exp_id": best_exp_id,
        "all_results": all_results,
        "policy_results": policy_results,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run DQN training for a team member")
    parser.add_argument(
        "--member",
        type=str,
        default="damour",
        choices=MEMBER_CHOICES,
        help="Member name to run training for",
    )
    args = parser.parse_args()

    try:
        result = run_training(member_name=args.member)
        if result is None:
            sys.exit(0)
    except Exception as exc:
        print(f"ERROR: Training failed: {exc}")
        sys.exit(1)
