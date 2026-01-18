"""
Plot training results from DQN training log.
Usage: python plot.py [--csv CSV_PATH] [--save SAVE_PATH]
"""

import pandas as pd
import matplotlib.pyplot as plt
import argparse


def plot_training_results(csv_path: str, save_path: str = None):
    """
    Plot training results from the DQN training log CSV.
    
    Args:
        csv_path: Path to the training log CSV file
        save_path: Optional path to save the figure (if None, displays interactively)
    """
    # Load data
    df = pd.read_csv(csv_path)
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('DQN Training Results - LunarLander', fontsize=14, fontweight='bold')
    
    # Plot 1: Episode Rewards
    ax1 = axes[0, 0]
    ax1.plot(df['episode'], df['reward'], alpha=0.3, color='blue', label='Episode Reward')
    ax1.plot(df['episode'], df['avg_reward_100'], color='red', linewidth=2, label='100-Episode Average')
    ax1.axhline(y=200, color='green', linestyle='--', linewidth=1.5, label='Solved Threshold (200)')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Reward')
    ax1.set_title('Training Rewards')
    ax1.legend(loc='lower right')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Moving Average (zoomed)
    ax2 = axes[0, 1]
    ax2.plot(df['episode'], df['avg_reward_100'], color='red', linewidth=2)
    ax2.axhline(y=200, color='green', linestyle='--', linewidth=1.5, label='Solved Threshold')
    ax2.fill_between(df['episode'], df['avg_reward_100'], 200, 
                     where=(df['avg_reward_100'] >= 200), alpha=0.3, color='green', label='Above Threshold')
    ax2.fill_between(df['episode'], df['avg_reward_100'], 200, 
                     where=(df['avg_reward_100'] < 200), alpha=0.3, color='red', label='Below Threshold')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Average Reward (100 episodes)')
    ax2.set_title('100-Episode Moving Average')
    ax2.legend(loc='lower right')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Epsilon Decay
    ax3 = axes[1, 0]
    ax3.plot(df['episode'], df['epsilon'], color='purple', linewidth=2)
    ax3.set_xlabel('Episode')
    ax3.set_ylabel('Epsilon')
    ax3.set_title('Exploration Rate (Epsilon) Decay')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Reward Distribution (histogram)
    ax4 = axes[1, 1]
    ax4.hist(df['reward'], bins=50, color='skyblue', edgecolor='navy', alpha=0.7)
    ax4.axvline(x=df['reward'].mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {df["reward"].mean():.1f}')
    ax4.axvline(x=df['reward'].median(), color='orange', linestyle='--', linewidth=2, label=f'Median: {df["reward"].median():.1f}')
    ax4.axvline(x=200, color='green', linestyle='--', linewidth=2, label='Solved Threshold (200)')
    ax4.set_xlabel('Reward')
    ax4.set_ylabel('Frequency')
    ax4.set_title('Reward Distribution')
    ax4.legend(loc='upper left')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Print statistics
    print("=" * 50)
    print("Training Statistics")
    print("=" * 50)
    print(f"Total Episodes: {len(df)}")
    print(f"Final 100-Episode Average: {df['avg_reward_100'].iloc[-1]:.2f}")
    print(f"Best Episode Reward: {df['reward'].max():.2f}")
    print(f"Worst Episode Reward: {df['reward'].min():.2f}")
    print(f"Mean Reward: {df['reward'].mean():.2f}")
    print(f"Std Reward: {df['reward'].std():.2f}")
    print(f"Final Epsilon: {df['epsilon'].iloc[-1]:.4f}")
    
    # Check if solved
    solved_episodes = df[df['avg_reward_100'] >= 200]
    if len(solved_episodes) > 0:
        first_solved = solved_episodes['episode'].iloc[0]
        print(f"\n✓ Environment SOLVED at episode {int(first_solved)}")
        print(f"  Episodes above threshold: {len(solved_episodes)}")
    else:
        print(f"\n✗ Environment NOT SOLVED (best avg: {df['avg_reward_100'].max():.2f})")
    print("=" * 50)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\nFigure saved to: {save_path}")
    else:
        plt.show()
    
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot DQN training results")
    parser.add_argument("--csv", type=str, default="dqn_weights_training_log_4000.csv",
                        help="Path to training log CSV file")
    parser.add_argument("--save", type=str, default=None,
                        help="Path to save the figure (optional, shows interactively if not provided)")
    args = parser.parse_args()
    
    plot_training_results(args.csv, args.save)
