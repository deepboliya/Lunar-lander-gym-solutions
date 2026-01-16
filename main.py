import gymnasium as gym
import numpy as np
from method_1 import SimpleSolution
from method_3 import SimpleSolution2
from method_2 import MainController
from method_4 import MainController2
from method_5 import MainController3

import argparse
import pandas as pd
GRAVITY_MAGNITUDE = 10.0
env = gym.make("LunarLander-v3", continuous=True, gravity=-GRAVITY_MAGNITUDE, render_mode="human")
# Reset environment to start a new episode

def main(controller, seed ,render=False):
    observation, info = env.reset(seed=seed)

    print(f"Starting observation: {observation}")

    episode_over = False
    total_reward = 0

    while not episode_over:

        action = controller.compute_action(observation)
        observation, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        episode_over = terminated or truncated
    print(info)
    print(f"Episode finished! Total reward: {total_reward}")
    env.close()
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', type=str, default='method_1', help='Method to run: method_1 or method_2')
    parser.add_argument('--seed', type=int, default=0, help='Random seed for the environment')
    parser.add_argument('--render', action='store_true', help='Render the environment')
    args = parser.parse_args()

    if args.method == 'method_1':
        controller = SimpleSolution()
    elif args.method == 'method_2':
        param_df = pd.read_json("cmaes_params.json")
        flattened_params = np.array(param_df.iloc[0]["Params"])
        print("Using flattened cma es params:", flattened_params)
        # controller = MainController(flattened_params, GRAVITY_MAGNITUDE)
        controller = MainController(gravity_magnitude=GRAVITY_MAGNITUDE, print_=True)
    elif args.method == 'method_3':
        controller = SimpleSolution2()
    elif args.method == 'method_4':
        param_df = pd.read_json("good_method_4_params.json")
        flattened_params = np.array(param_df.iloc[0]["Params"])
        print("Using flattened cma es params:", flattened_params)
        controller = MainController2(flattened_params, GRAVITY_MAGNITUDE, print_=True)
    elif args.method == 'method_5':
        param_df = pd.read_json("cmaes_params.json")
        flattened_params = np.array(param_df.iloc[0]["Params"])
        print("Using flattened cma es params:", flattened_params)
        controller = MainController3(flattened_params, GRAVITY_MAGNITUDE)
    else:
        raise ValueError("Invalid method specified. Choose 'method_1', 'method_2', or 'method_3'.")

    main(controller, seed=args.seed, render=args.render)