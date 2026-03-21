import argparse
import os

import ale_py  # noqa: F401  # Ensures Atari env registration
import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack


def make_play_env(seed: int = 42):
    """Create a rendered Atari Pong environment with the same preprocessing used in training."""
    env = make_atari_env(
        "ALE/Pong-v5",
        n_envs=1,
        seed=seed,
        env_kwargs={"render_mode": "human"},
    )
    return VecFrameStack(env, n_stack=4)


def resolve_model_path(member: str | None, model_path: str | None) -> str:
    """Resolve model path from explicit path or member name."""
    if model_path:
        return model_path

    member_name = (member or "musembi").lower()
    return f"./experiments/{member_name}/models/dqn_model.zip"


def run_play(model_path: str, episodes: int = 3, seed: int = 42) -> None:
    """Load a trained DQN model and run greedy evaluation episodes."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")

    env = make_play_env(seed=seed)
    model = DQN.load(model_path)

    print("=" * 55)
    print("DQN PLAY MODE — ATARI PONG")
    print(f"Model: {model_path}")
    print(f"Episodes: {episodes}")
    print("Policy: Greedy (deterministic=True)")
    print("=" * 55)

    try:
        for episode in range(1, episodes + 1):
            obs = env.reset()
            done = np.array([False])
            ep_reward = 0.0
            steps = 0

            while not bool(np.any(done)):
                # deterministic=True makes DQN act greedily (argmax Q-values)
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, _ = env.step(action)
                ep_reward += float(reward[0])
                steps += 1

            print(f"Episode {episode}: reward={ep_reward:.2f}, steps={steps}")

    except KeyboardInterrupt:
        print("\nPlayback interrupted by user.")
    finally:
        env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Play Atari Pong using a trained DQN model")
    parser.add_argument("--member", type=str, default="musembi", help="Member name used for default model path")
    parser.add_argument("--model-path", type=str, default=None, help="Path to dqn_model.zip")
    parser.add_argument("--episodes", type=int, default=3, help="Number of episodes to play")
    parser.add_argument("--seed", type=int, default=42, help="Environment seed")
    args = parser.parse_args()

    selected_model_path = resolve_model_path(member=args.member, model_path=args.model_path)
    run_play(model_path=selected_model_path, episodes=args.episodes, seed=args.seed)
