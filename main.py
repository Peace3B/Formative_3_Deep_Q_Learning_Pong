# ============================================================
# MAIN.PY — Entry point for DQN Pong Training
# ============================================================
# This module serves as the main entry point for the
# Deep Q-Learning Pong training pipeline.
# ============================================================

import sys
import os


def main():
    """
    Main entry point that runs the complete DQN training pipeline.
    Imports and executes the train module which handles:
      - Policy comparison (CNN vs MLP)
      - 10 hyperparameter experiments
      - Model training and evaluation
      - Results logging and best model saving
    """
    print("\n" + "="*60)
    print("  FORMATIVE 3: DEEP Q-LEARNING FOR ATARI PONG")
    print("="*60)
    print("\n  Starting DQN training pipeline...\n")
    
    try:
        # Import and run the training module
        from train import (
            compare_policies,
            EXPERIMENTS,
            run_experiment,
            save_hyperparameter_table
        )
        
        # --------------------------------------------------------
        # PHASE 1: Compare CNNPolicy vs MLPPolicy
        # --------------------------------------------------------
        print("[PHASE 1] Comparing CNNPolicy vs MLPPolicy...")
        policy_results = compare_policies(timesteps=50_000)

        # --------------------------------------------------------
        # PHASE 2: Run all 10 hyperparameter experiments
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
            all_results.append({
                "exp": exp,
                "avg_reward": avg_reward
            })

            # Track best performing experiment
            if avg_reward > best_reward:
                best_reward = avg_reward
                best_model = model
                best_exp_id = exp["id"]

        # --------------------------------------------------------
        # PHASE 3: Save results and best model
        # --------------------------------------------------------
        print("\n[PHASE 3] Saving results...")
        save_hyperparameter_table(all_results)

        # Save the best model
        os.makedirs("./models/", exist_ok=True)
        best_model.save("./models/dqn_model")
        print(f"\nBest model (Experiment {best_exp_id}) saved as "
              f"./models/dqn_model.zip")
        print(f"Best avg reward: {best_reward}")

        # --------------------------------------------------------
        # PHASE 4: Print final summary
        # --------------------------------------------------------
        print("\n" + "="*60)
        print("  TRAINING COMPLETE — FINAL SUMMARY")
        print("="*60)
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
        print("="*60 + "\n")
        
    except ImportError as e:
        print(f"ERROR: Failed to import training module: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"ERROR: Training pipeline failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
