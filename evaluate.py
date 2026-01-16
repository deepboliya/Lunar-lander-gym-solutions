"""
Multi-seed evaluation script for LunarLander controllers.
Runs multiple episodes with different seeds and reports statistics.

Usage:
    python evaluate.py --method method_2 --num_seeds 10
    python evaluate.py --method method_4 --num_seeds 20 --start_seed 100
    python evaluate.py --method method_1 --render  # render first episode
"""
import gymnasium as gym
import numpy as np
import argparse
import pandas as pd

from method_1 import SimpleSolution
from method_2 import MainController
from method_3 import SimpleSolution2
from method_4 import MainController2
from method_5 import MainController3

GRAVITY_MAGNITUDE = 10.0


def run_episode(controller_factory, seed, render=False):
    """Run a single episode with a given seed and return the total reward."""
    render_mode = "human" if render else None
    env = gym.make("LunarLander-v3", continuous=True, gravity=-GRAVITY_MAGNITUDE, render_mode=render_mode)
    observation, info = env.reset(seed=seed)

    controller = controller_factory()  # Fresh controller for each episode

    episode_over = False
    total_reward = 0

    while not episode_over:
        action = controller.compute_action(observation)
        observation, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        episode_over = terminated or truncated

    env.close()
    return total_reward


def evaluate(controller_factory, seeds, render_first=False):
    """Run evaluation across multiple seeds and print statistics."""
    rewards = []

    for i, seed in enumerate(seeds):
        render = render_first and (i == 0)
        reward = run_episode(controller_factory, seed, render=render)
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


def get_controller_factory(method):
    """Return a factory function that creates a fresh controller instance."""
    if method == "method_1":
        return lambda: SimpleSolution()

    elif method == "method_2":
        param_df = pd.read_json("cmaes_params.json")
        flattened_params = np.array(param_df.iloc[0]["Params"])
        print("Using flattened cma es params:", flattened_params)
        return lambda: MainController(gravity_magnitude=GRAVITY_MAGNITUDE, print_=False)

    elif method == "method_3":
        return lambda: SimpleSolution2()

    elif method == "method_4":
        param_df = pd.read_json("good_method_4_params.json")
        flattened_params = np.array(param_df.iloc[0]["Params"])
        print("Using flattened cma es params:", flattened_params)
        return lambda fp=flattened_params: MainController2(fp, GRAVITY_MAGNITUDE, print_=False)

    elif method == "method_5":
        param_df = pd.read_json("cmaes_params.json")
        flattened_params = np.array(param_df.iloc[0]["Params"])
        print("Using flattened cma es params:", flattened_params)
        return lambda fp=flattened_params: MainController3(fp, GRAVITY_MAGNITUDE)

    else:
        raise ValueError("Invalid method. Choose from: method_1, method_2, method_3, method_4, method_5")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate LunarLander controller over multiple seeds")
    parser.add_argument("--method", type=str, default="method_1",
                        help="Controller method: method_1, method_2, method_3, method_4, method_5")
    parser.add_argument("--num_seeds", type=int, default=10,
                        help="Number of seeds to evaluate")
    parser.add_argument("--start_seed", type=int, default=0,
                        help="Starting seed value")
    parser.add_argument("--render", action="store_true",
                        help="Render the first episode")
    args = parser.parse_args()

    seeds = list(range(args.start_seed, args.start_seed + args.num_seeds))
    print(f"Evaluating {args.method} with {len(seeds)} seeds: {seeds[0]} to {seeds[-1]}")
    print("-" * 50)

    controller_factory = get_controller_factory(args.method)
    evaluate(controller_factory, seeds, render_first=args.render)
