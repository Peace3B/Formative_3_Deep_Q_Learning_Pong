# ============================================================
# MAIN.PY — Entry point for DQN Pong Training
# ============================================================
# This module serves as the main entry point for the
# Deep Q-Learning Pong training pipeline.
# Allows training for each team member with their own configs.
# ============================================================

import sys
import os
from train import run_training


MEMBERS = ["damour", "daniel", "peace", "musembi"]


def display_menu():
    """Display member selection menu."""
    print("\n" + "="*60)
    print("  FORMATIVE 3: DEEP Q-LEARNING FOR ATARI PONG")
    print("="*60)
    print("\n  SELECT TEAM MEMBER FOR TRAINING:\n")
    
    for i, member in enumerate(MEMBERS, 1):
        print(f"    {i}. {member.capitalize()}")
    
    print(f"    {len(MEMBERS) + 1}. Run all members (sequential)")
    print(f"    0. Exit\n")


def get_member_choice():
    """Prompt user to select a member."""
    while True:
        try:
            choice = input("  Enter your choice (0-5): ").strip()
            choice = int(choice)
            
            if choice == 0:
                print("\n  Exiting...")
                sys.exit(0)
            elif 1 <= choice <= len(MEMBERS):
                return MEMBERS[choice - 1]
            elif choice == len(MEMBERS) + 1:
                return "all"
            else:
                print(f"  Invalid choice. Please enter 0-{len(MEMBERS) + 1}")
        except ValueError:
            print("  Invalid input. Please enter a number.")


def main():
    """
    Main entry point that runs the complete DQN training pipeline.
    Allows selection of which member's training to execute.
    """
    print("\n" + "="*60)
    print("  DEEP Q-LEARNING FOR ATARI PONG")
    print("="*60)
    
    display_menu()
    choice = get_member_choice()
    
    if choice == "all":
        print("\n" + "="*60)
        print("  RUNNING TRAINING FOR ALL MEMBERS (SEQUENTIAL)")
        print("="*60)
        
        all_results = {}
        for member in MEMBERS:
            try:
                print(f"\n\n{'#'*60}")
                print(f"  STARTING TRAINING FOR: {member.upper()}")
                print(f"{'#'*60}\n")
                
                result = run_training(member_name=member)
                
                if result:
                    all_results[member] = result
                    print(f"\n{'='*60}")
                    print(f"  {member.upper()} TRAINING COMPLETED")
                    print(f"  Best Reward: {result['best_reward']}")
                    print(f"{'='*60}\n")
                else:
                    all_results[member] = None
                    print(f"\n{'='*60}")
                    print(f"  {member.upper()} SKIPPED (NO EXPERIMENTS)")
                    print(f"{'='*60}\n")
                
            except Exception as e:
                print(f"\nERROR training {member}: {e}")
                all_results[member] = None
        
        # Print combined summary
        print("\n\n" + "="*60)
        print("  FINAL SUMMARY — ALL MEMBERS")
        print("="*60)
        
        for member in MEMBERS:
            if all_results[member]:
                result = all_results[member]
                print(f"\n  {member.capitalize():10s}: "
                      f"Best Reward = {result['best_reward']:8.2f} "
                      f"(Exp {result['best_exp_id']})")
            else:
                print(f"\n  {member.capitalize():10s}: NOT TRAINED")
        
        print(f"\n{'='*60}\n")
        
    else:
        # Single member training
        try:
            print(f"\n  Starting training for {choice.upper()}...\n")
            result = run_training(member_name=choice)
            
            if result:
                print(f"\n  Training completed for {choice.upper()}!")
                print(f"  Results saved to: ./results/{choice}/")
                print(f"  Model saved to:   ./models/{choice}/")
            else:
                print(f"\n  {choice.upper()} cannot train yet. Configure experiments first.")
            
        except Exception as e:
            print(f"\n  ERROR: Training failed for {choice}: {e}")
            sys.exit(1)


if __name__ == "__main__":
    # Check if member specified via command line
    if len(sys.argv) > 1:
        member = sys.argv[1].lower()
        if member in MEMBERS:
            try:
                print(f"\nTraining for {member}...")
                run_training(member_name=member)
            except Exception as e:
                print(f"ERROR: {e}")
                sys.exit(1)
        else:
            print(f"Unknown member: {member}")
            print(f"Valid members: {', '.join(MEMBERS)}")
            sys.exit(1)
    else:
        # Interactive menu if no command line argument
        main()
