"""
Deep Q Learning Group 3: Playing Script
Loads the best trained DQN model and runs it in the Atari Pong environment.

The agent uses a greedy policy during evaluation — it always selects the action
with the highest Q-value, with no random exploration (epsilon = 0).

Usage:
    python play.py

Optional arguments:
    --model     Path to the trained model (default: dqn_model.zip)
    --episodes  Number of episodes to play (default: 3)
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


class PongEvaluator:
    """Class-based evaluator for the trained DQN agent in Atari Pong."""
    
    def __init__(self, model_path="dqn_model.zip", episodes=3, fps=15, render=True, use_color=True, ale_window=False):
        self.model_path = model_path
        self.episodes = episodes
        self.fps = fps
        self.render_enabled = render
        self.use_color = use_color
        
        # Check pygame availability for custom window
        self.custom_window = self.render_enabled and (not ale_window)
        if self.custom_window and pygame is None:
            print(style("[WARN] Pygame not found. Falling back to ALE default window.", Colors.YELLOW, use_color=self.use_color))
            self.custom_window = False
            
        self.env = None
        self.model = None
        self.screen = None
        self.clock = None

    def load_model(self):
        """Load the DQN model from the specified path."""
        print(style("\nLoading model...", Colors.YELLOW, use_color=self.use_color))
        if not os.path.exists(self.model_path):
            print(style(f"[ERROR] Model file not found: {self.model_path}", Colors.RED, bold=True, use_color=self.use_color))
            print("Make sure the model path is correct and the file exists.")
            return False
        self.model = DQN.load(self.model_path)
        print(style("Model loaded successfully.", Colors.GREEN, use_color=self.use_color))
        return True

    def setup_environment(self):
        """Create and wrap the Atari Pong environment."""
        print(style("Setting up environment...", Colors.YELLOW, use_color=self.use_color))
        if ale_py is None:
            raise ImportError("ale-py not installed. Run: pip install -r requirements.txt")

        gym.register_envs(ale_py)
        
        if self.render_enabled and self.custom_window and pygame is not None:
            render_mode = "rgb_array"
        else:
            render_mode = "human" if self.render_enabled else "rgb_array"
            
        env = gym.make("ALE/Pong-v5", render_mode=render_mode)
        self.env = AtariWrapper(env)
        print(style("Environment ready.\n", Colors.GREEN, use_color=self.use_color))

    def evaluate(self):
        """Run the evaluation episodes."""
        if self.custom_window and self.render_enabled and pygame is not None:
            pygame.init()
            pygame.display.set_caption(WINDOW_TITLE)
            self.clock = pygame.time.Clock()

        episode_rewards = []

        for episode in range(1, self.episodes + 1):
            obs, _ = self.env.reset()
            total_reward = 0
            done = False
            step = 0

            print(style(f"\nEpisode {episode} starting...", Colors.CYAN, bold=True, use_color=self.use_color))

            while not done:
                # Greedy action selection — deterministic=True disables exploration
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = self.env.step(action)
                total_reward += reward
                done = terminated or truncated
                step += 1

                if self.custom_window and self.render_enabled and pygame is not None:
                    frame = self.env.unwrapped.render()
                    if frame is not None:
                        if self.screen is None:
                            h, w, _ = frame.shape
                            self.screen = pygame.display.set_mode((w * WINDOW_SCALE, h * WINDOW_SCALE))
                        
                        frame_surface = pygame.surfarray.make_surface(np.transpose(frame, (1, 0, 2)))
                        if WINDOW_SCALE > 1:
                            frame_surface = pygame.transform.scale(frame_surface, (w * WINDOW_SCALE, h * WINDOW_SCALE))
                        
                        self.screen.blit(frame_surface, (0, 0))
                        pygame.display.flip()
                        
                        self.clock.tick(self.fps)
                        for event in pygame.event.get():
                            if event.type == pygame.QUIT:
                                done = True
                                print(style("Window closed by user.", Colors.YELLOW, use_color=self.use_color))
                                break

            episode_rewards.append(total_reward)
            print(
                style(
                    f"Episode {episode} finished - Total Reward: {total_reward:.1f}  Steps: {step}",
                    Colors.GREEN,
                    use_color=self.use_color,
                )
            )

        if self.screen is not None:
            pygame.quit()

        return episode_rewards

    def run(self):
        """Execute the full evaluation pipeline and print summary."""
        print("=" * 70)
        print(style("Deep Q Learning Group 3 - DQN Agent Evaluation (Atari Pong)", Colors.CYAN, bold=True, use_color=self.use_color))
        print("=" * 70)
        print(f"Model : {self.model_path}")
        print(f"Episodes : {self.episodes}")
        print(f"Policy : Greedy (deterministic=True, no exploration)")
        print(f"Rendering : {'Disabled' if not self.render_enabled else 'Enabled'}")
        if self.render_enabled:
            print(f"Window title : {WINDOW_TITLE if self.custom_window else 'The Arcade Learning Environment (ALE default)'}")
            if self.custom_window:
                print(f"Speed (FPS) : {self.fps}")
        print(f"Colors : {'Disabled' if not self.use_color else 'Enabled'}")
        print("=" * 70)

        if not self.load_model():
            return
            
        self.setup_environment()
        rewards = self.evaluate()

        print("\n" + "=" * 70)
        print(style("Group 3 Evaluation Complete", Colors.CYAN, bold=True, use_color=self.use_color))
        print("=" * 70)
        print(f"Episodes played   : {self.episodes}")
        if len(rewards) > 0:
            print(f"Mean reward       : {np.mean(rewards):.2f}")
            print(f"Best episode      : {max(rewards):.1f}")
            print(f"Worst episode     : {min(rewards):.1f}")
            print(f"All rewards       : {[round(r, 1) for r in rewards]}")
        print("=" * 70)

        if self.env:
            self.env.close()

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
        default=3,
        help="Number of episodes to play (default: 3)",
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
    
    evaluator = PongEvaluator(
        model_path=args.model,
        episodes=args.episodes,
        fps=args.fps,
        render=not args.no_render,
        use_color=not args.no_color,
        ale_window=args.ale_window
    )
    evaluator.run()

if __name__ == "__main__":
    main()
