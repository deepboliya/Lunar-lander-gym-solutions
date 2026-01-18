"""
Main script for running the DQN agent on LunarLander-v3.

Usage:
    python main2.py
    python main2.py --weights dqn_weights_best.pth
    python main2.py --episodes 5 --no-render
"""
import gymnasium as gym
import argparse
from method_7 import MainController

GRAVITY_MAGNITUDE = 10.0


def run_episode(controller, render=True, seed=None):
    """Run a single episode and return the total reward."""
    render_mode = "human" if render else None
    env = gym.make("LunarLander-v3", gravity=-GRAVITY_MAGNITUDE, render_mode=render_mode)
    
    observation, info = env.reset(seed=seed)
    episode_over = False
    total_reward = 0

    while not episode_over:
        action = controller.compute_action(observation)
        observation, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        episode_over = terminated or truncated

    env.close()
    return total_reward


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run DQN agent on LunarLander")
    parser.add_argument("--weights", type=str, default="dqn_weights_best.pth",
                        help="Path to the weights file")
    parser.add_argument("--episodes", type=int, default=1,
                        help="Number of episodes to run")
    parser.add_argument("--no-render", action="store_true",
                        help="Disable rendering")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed (increments for each episode)")
    args = parser.parse_args()

    controller = MainController(weights_path=args.weights)
    render = not args.no_render

    rewards = []
    for i in range(args.episodes):
        seed = args.seed + i if args.seed is not None else None
        reward = run_episode(controller, render=render, seed=seed)
        rewards.append(reward)
        print(f"Episode {i+1}: reward = {reward:.2f}")

    if args.episodes > 1:
        import numpy as np
        print(f"\nMean: {np.mean(rewards):.2f}, Min: {np.min(rewards):.2f}, Max: {np.max(rewards):.2f}")
