import argparse
import numpy as np
import pandas as pd
from evaluate import evaluate

from method_1 import SimpleSolution
from method_2 import AgentPID
from method_3 import SimpleSolution2
from method_4 import MainController2
from method_5 import MainController3
from method_6 import SimpleSolution3
from method_7 import DQNAgent
from method_8 import Solution4

GRAVITY_MAGNITUDE = 10.0

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run LunarLander controller evaluation")
    parser.add_argument('--method', type=str, default='method_1', 
                        help='Method to run: method_1, method_2, method_3, method_4, method_5, method_6')
    parser.add_argument('--num-seeds', type=int, default=100, 
                        help='Number of seeds to evaluate')
    parser.add_argument('--start-seed', type=int, default=0, 
                        help='Starting seed value')
    parser.add_argument('--render', action='store_true', 
                        help='Render the first episode')
    parser.add_argument('--discrete', action='store_true',
                        help='Use discrete action space (default is continuous)')
    parser.add_argument('--print', action='store_true',
                        help='Print debug information')
    args = parser.parse_args()
    
    continuous = not args.discrete

    # Create controller factory based on method
    if args.method == 'method_1':
        controller_factory = lambda: SimpleSolution()

    elif args.method == 'method_2':
        param_df = pd.read_json("method_2.json")
        flattened_params = np.array(param_df.iloc[0]["Params"])
        print("Using flattened cma es params:", flattened_params)
        controller_factory = lambda: AgentPID(flattened_params=flattened_params, gravity_magnitude=GRAVITY_MAGNITUDE, print_=False)

    elif args.method == 'method_3':
        controller_factory = lambda: SimpleSolution2()

    elif args.method == 'method_4':
        param_df = pd.read_json("method_4.json")
        flattened_params = np.array(param_df.iloc[0]["Params"])
        print("Using flattened cma es params:", flattened_params)
        controller_factory = lambda fp=flattened_params: MainController2(fp, GRAVITY_MAGNITUDE, print_=False)

    elif args.method == 'method_5':
        param_df = pd.read_json("method_5.json")
        flattened_params = np.array(param_df.iloc[0]["Params"])
        # flattened_params = None
        print("Using flattened cma es params:", flattened_params)
        controller_factory = lambda fp=flattened_params: MainController3(fp, GRAVITY_MAGNITUDE)

    elif args.method == 'method_6':
        param_df = pd.read_json("method_6.json")
        flattened_params = np.array(param_df.iloc[0]["Params"])
        print("Using flattened cma es params:", flattened_params)
        controller_factory = lambda fp=flattened_params: SimpleSolution3(fp)

    elif args.method == 'method_7':
        controller_factory = lambda: DQNAgent.load_for_eval()

    elif args.method == 'method_8':
        controller_factory = lambda: Solution4(gravity_magnitude=GRAVITY_MAGNITUDE, print_=args.print)

    else:
        raise ValueError("Invalid method. Choose from: method_1, method_2, method_3, method_4, method_5, method_6, method_7")

    seeds = list(range(args.start_seed, args.start_seed + args.num_seeds))
    print(f"Evaluating {args.method} with {len(seeds)} seeds: {seeds[0]} to {seeds[-1]}")
    print(f"Continuous: {continuous}")
    print("-" * 50)

    evaluate(controller_factory, seeds, render_first=args.render, continuous=continuous)