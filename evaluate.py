"""
Multi-seed evaluation module for LunarLander controllers.
Runs multiple episodes with different seeds and reports statistics.

Usage (from main.py):
    from evaluate import evaluate
    evaluate(controller_factory, seeds, render_first=args.render)
"""
import gymnasium as gym
import numpy as np

GRAVITY_MAGNITUDE = 10.0


def run_episode(controller_factory, seed, render=False, continuous=True):
    """Run a single episode with a given seed and return the total reward."""
    render_mode = "human" if render else None
    env = gym.make("LunarLander-v3", continuous=continuous, gravity=-GRAVITY_MAGNITUDE, render_mode=render_mode)
    observation, info = env.reset(seed=seed)

    controller = controller_factory()  # Fresh controller for each episode

    episode_over = False
    total_reward = 0

    while not episode_over:
        action = controller.compute_action(observation)
        observation, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        episode_over = terminated or truncated
        # print(f"Step reward: {reward:.2f}, \t Total reward: {total_reward:.2f}")

    env.close()
    return total_reward


def evaluate(controller_factory, seeds, render_first=False, continuous=True):
    """Run evaluation across multiple seeds and print statistics."""
    rewards = []

    for i, seed in enumerate(seeds):
        render = render_first and (i == 0)
        reward = run_episode(controller_factory, seed, render=render, continuous=continuous)
        rewards.append(reward)
        print(f"  Seed {seed:4d}: reward = {reward:8.2f}")

    rewards = np.array(rewards)
    print("\n" + "=" * 50)
    print("EVALUATION RESULTS")
    print("=" * 50)
    print(f"  Episodes:  {len(rewards)}")
    print(f"  Mean:      {np.mean(rewards):8.2f}")
    print(f"  Median:    {np.median(rewards):8.2f}")
    print(f"  Std:       {np.std(rewards):8.2f}")
    print(f"  Min:       {np.min(rewards):8.2f}")
    print(f"  Max:       {np.max(rewards):8.2f}")
    print("=" * 50)

    # Flag seeds with reward < 200
    bad_seeds = [(seed, reward) for seed, reward in zip(seeds, rewards) if reward < 200]
    if bad_seeds:
        print(f"Seeds with reward < 200 ({len(bad_seeds)}/{len(seeds)}):")
        for seed, reward in bad_seeds:
            print(f"    Seed {seed:4d}: {reward:8.2f}")
    else:
        print("All seeds achieved reward >= 200")

    return rewards
