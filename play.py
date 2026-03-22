"""
Deep Q Learning Group 3: Playing Script
Loads the best trained DQN model and runs it in the Atari Pong environment.

The agent uses a greedy policy during evaluation — it always selects the action
with the highest Q-value, with no random exploration (epsilon = 0).

Usage:
    python play.py

Optional arguments:
    --model     Path to the trained model (default: dqn_model.zip)
    --episodes  Number of episodes to play (default: 5)
    --no-render Disable game rendering (useful for headless evaluation)
"""

import os
import argparse
import numpy as np
import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.atari_wrappers import AtariWrapper
import warnings

warnings.filterwarnings("ignore")

try:
    import ale_py
except ImportError:
    ale_py = None

try:
    import pygame
except ImportError:
    pygame = None

WINDOW_TITLE = "Deep Q Learning Group 3 - Atari Pong"
WINDOW_SCALE = 3  # Scale the window by 3x for better visibility


class Colors:
    """ANSI color palette for clean terminal output."""

    RESET = "\033[0m"
    BOLD = "\033[1m"
    CYAN = "\033[96m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"


def style(text, color="", bold=False, use_color=True):
    """Apply optional ANSI styling to text."""
    if not use_color:
        return text
    prefix = ""
    if bold:
        prefix += Colors.BOLD
    if color:
        prefix += color
    return f"{prefix}{text}{Colors.RESET}"


# ---------------------------------------------------------------------------
# Environment factory — matches the training environment exactly
# ---------------------------------------------------------------------------
def create_environment(env_name="ALE/Pong-v5", render=True, custom_window=True):
    """
    Create the Atari Pong environment for evaluation.
    Uses custom pygame window if requested, else falls back to default.

    Args:
        env_name (str): Name of the Atari environment
        render (bool): Whether to render the game visually
        custom_window (bool): Whether to use custom titled window

    Returns:
        gym.Env: The wrapped environment
    """
    if ale_py is None:
        raise ImportError("ale-py not installed. Run: pip install -r requirements.txt")

    gym.register_envs(ale_py)
    
    if render and custom_window and pygame is not None:
        render_mode = "rgb_array"
    else:
        render_mode = "human" if render else "rgb_array"
        
    env = gym.make(env_name, render_mode=render_mode)
    env = AtariWrapper(env)
    return env


# ---------------------------------------------------------------------------
# Greedy policy evaluation
# ---------------------------------------------------------------------------
def evaluate_agent(model, env, num_episodes=5, use_color=True, custom_window=False, render=False, fps=15):
    """
    Run the agent for a given number of episodes using a greedy policy.
    The agent always picks the action with the highest Q-value — no exploration.

    Args:
        model: The loaded DQN model
        env: The Atari environment
        num_episodes (int): Number of episodes to run

    Returns:
        list: Total rewards per episode
    """
    # Pygame setup for custom rendering
    screen = None
    clock = None
    if custom_window and render and pygame is not None:
        pygame.init()
        pygame.display.set_caption(WINDOW_TITLE)
        clock = pygame.time.Clock()

    episode_rewards = []

    for episode in range(1, num_episodes + 1):
        obs, _ = env.reset()
        total_reward = 0
        done = False
        step = 0

        print(style(f"\nEpisode {episode} starting...", Colors.CYAN, bold=True, use_color=use_color))

        while not done:
            # Greedy action selection — deterministic=True disables exploration
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            done = terminated or truncated
            step += 1

            if custom_window and render and pygame is not None:
                # Force env to give the base rgb_array
                frame = env.unwrapped.render()
                if frame is not None:
                    # Initialize screen if not done yet based on frame size
                    if screen is None:
                        h, w, _ = frame.shape
                        screen = pygame.display.set_mode((w * WINDOW_SCALE, h * WINDOW_SCALE))
                    
                    # Process frame for pygame
                    # Pygame expects (width, height) and frame is (height, width, channels)
                    # We transpose to swap width and height axes for Pygame surface
                    frame_surface = pygame.surfarray.make_surface(np.transpose(frame, (1, 0, 2)))
                    if WINDOW_SCALE > 1:
                        frame_surface = pygame.transform.scale(frame_surface, (w * WINDOW_SCALE, h * WINDOW_SCALE))
                    
                    screen.blit(frame_surface, (0, 0))
                    pygame.display.flip()
                    
                    # Control playback framerate and check events
                    # ALE runs at 60Hz and AtariWrapper uses frame-skip=4 by default,
                    # so 15 ticks/sec matches the default ALE window speed closely.
                    clock.tick(fps)
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            done = True
                            print(style("Window closed by user.", Colors.YELLOW, use_color=use_color))
                            break

        episode_rewards.append(total_reward)
        print(
            style(
                f"Episode {episode} finished - Total Reward: {total_reward:.1f}  Steps: {step}",
                Colors.GREEN,
                use_color=use_color,
            )
        )

    if screen is not None:
        pygame.quit()

    return episode_rewards


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Deep Q Learning Group 3 - Play Atari Pong with a trained DQN agent")
    parser.add_argument(
        "--model",
        type=str,
        default="dqn_model.zip",
        help="Path to the trained model file",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=5,
        help="Number of episodes to play",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=15,
        help="Viewing speed (frames per second). Default 15 matches ALE window pacing.",
    )
    parser.add_argument(
        "--no-render",
        action="store_true",
        help="Disable game rendering",
    )
    parser.add_argument(
        "--no-color",
        action="store_true",
        help="Disable colored terminal output",
    )
    parser.add_argument(
        "--ale-window",
        action="store_true",
        help="Use ALE default render window title (disables custom scaling and title)",
    )
    args = parser.parse_args()
    use_color = not args.no_color
    use_render = not args.no_render
    use_custom_window = use_render and (not args.ale_window)

    if use_custom_window and pygame is None:
        print(style("[WARN] Pygame not found. Falling back to ALE default window.", Colors.YELLOW, use_color=use_color))
        use_custom_window = False

    # Validate model path
    if not os.path.exists(args.model):
        print(style(f"[ERROR] Model file not found: {args.model}", Colors.RED, bold=True, use_color=use_color))
        print("Make sure the model path is correct and the file exists.")
        return

    print("=" * 70)
    print(style("Deep Q Learning Group 3 - DQN Agent Evaluation (Atari Pong)", Colors.CYAN, bold=True, use_color=use_color))
    print("=" * 70)
    print(f"Model : {args.model}")
    print(f"Episodes : {args.episodes}")
    print(f"Policy : Greedy (deterministic=True, no exploration)")
    print(f"Rendering : {'Disabled' if args.no_render else 'Enabled'}")
    if use_render:
        print(f"Window title : {WINDOW_TITLE if use_custom_window else 'The Arcade Learning Environment (ALE default)'}")
        if use_custom_window:
            print(f"Speed (FPS) : {args.fps}")
    print(f"Colors : {'Disabled' if args.no_color else 'Enabled'}")
    print("=" * 70)

    # Load the trained model
    print(style("\nLoading model...", Colors.YELLOW, use_color=use_color))
    model = DQN.load(args.model)
    print(style("Model loaded successfully.", Colors.GREEN, use_color=use_color))

    # Create environment
    print(style("Setting up environment...", Colors.YELLOW, use_color=use_color))
    env = create_environment(render=use_render, custom_window=use_custom_window)
    print(style("Environment ready.\n", Colors.GREEN, use_color=use_color))

    # Run evaluation episodes
    rewards = evaluate_agent(
        model, 
        env, 
        num_episodes=args.episodes, 
        use_color=use_color,
        custom_window=use_custom_window,
        render=use_render,
        fps=args.fps
    )

    # Summary
    print("\n" + "=" * 70)
    print(style("Group 3 Evaluation Complete", Colors.CYAN, bold=True, use_color=use_color))
    print("=" * 70)
    print(f"Episodes played   : {args.episodes}")
    print(f"Mean reward       : {np.mean(rewards):.2f}")
    print(f"Best episode      : {max(rewards):.1f}")
    print(f"Worst episode     : {min(rewards):.1f}")
    print(f"All rewards       : {[round(r, 1) for r in rewards]}")
    print("=" * 70)

    env.close()


if __name__ == "__main__":
    main()