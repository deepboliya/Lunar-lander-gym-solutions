from gymnasium.wrappers import RecordVideo
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import cma
from cma.optimization_tools import EvalParallel2
import argparse
import time
import warnings
import gymnasium as gym

from method_5 import MainController3
from main import GRAVITY_MAGNITUDE

warnings.filterwarnings('ignore')

env = gym.make("LunarLander-v3", continuous=True, gravity=-GRAVITY_MAGNITUDE)# , render_mode="human")


def fitness(params):

    """This is the fitness function which is optimised by CMA-ES.
    Note that the cma library minimises the fitness function by default.
    You should make suitable adjustments to make sure fitness is maximised"""

    # Write your fitness function below. You have to write the code to interact with the environment and
    # use the information provided by the environment to formulate the fitness function in terms of CMA-ES params
    # which are provided as an argument to this function. You can refer the code provided in evaluation section of
    # the main function to see how to interact with the environment. You should invoke your policy using the following:
    # policy(state, info, False, params)
    total_reward = 0
    main_controller = MainController3(params, GRAVITY_MAGNITUDE)

    (obs, info) = env.reset()  # Getting initial state information from the environment
    done = False
    while not done:  # While the episode is not done
        action = main_controller.compute_action(obs)  # Call policy to produce action
        (obs, reward, term, trunc, info) = env.step(action)  # Take action in the environment
        total_reward += reward
        done = term or trunc  # If episode has terminated or truncated, set boolean variable done to True

    print("Total reward:", total_reward)

    return -total_reward

def call_cma(num_gen=2, pop_size=2, num_policy_params = 1):
    sigma0 = 1
    x0 = np.random.normal(0, 1, (num_policy_params, 1))  # Initialisation of parameter vector
    opts = {'maxiter':num_gen, 'popsize':pop_size}
    es = cma.CMAEvolutionStrategy(x0, sigma0, opts)
    with EvalParallel2(fitness, es.popsize + 1) as eval_all:
        while not es.stop():
            X = es.ask()
            es.tell(X, eval_all(X))
            es.logger.add()  # write data to disc for plotting
            es.disp()
    es.result_pretty()
    return es.result

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action='store_true')  # For training using CMA-ES
    parser.add_argument("--numTracks", type=int, default=6, required=False)  # Number of tracks for evaluation
    parser.add_argument("--seed", type=int, default=2025, required=False)  # Seed for evaluation
    parser.add_argument("--render", action='store_true')  # For rendering the evaluations

    args = parser.parse_args()

    train_mode = args.train
    num_tracks = args.numTracks
    seed = args.seed
    rendering = args.render

    """CMA-ES code begins"""
    # You can skip this part if you don't intend to use CMA-ES

    if train_mode:
        num_gen = 100
        pop_size = 300
        num_policy_params = 17
        X = call_cma(num_gen, pop_size, num_policy_params)
        cmaes_params = X[0]  # Parameters returned by CMA-ES after training
        cmaes_params_df = pd.DataFrame({
            'Params': [cmaes_params]
        })
        cmaes_params_df.to_json("cmaes_params.json")  # Storing parameters for evaluation purpose

    """CMA-ES code ends"""

    """Evaluation code begins"""
    # Do not modify this part.

    if rendering:
        env = RecordVideo(env, video_folder="videos", name_prefix="eval", episode_trigger=lambda x: True)

    """Code to generate learning curve and logs of CMA-ES"""
    # To be used only if your policy has parameters which are optimised using CMA-ES
    if train_mode:
        datContent = [i.strip().split() for i in open("outcmaes/fit.dat").readlines()]

        generations = []
        evaluations = []
        bestever = []
        best = []
        median = []
        worst = []

        for i in range(1, len(datContent)):
            generations.append(int(datContent[i][0]))
            evaluations.append(int(datContent[i][1]))
            bestever.append(-float(datContent[i][4]))
            best.append(-float(datContent[i][5]))
            median.append(-float(datContent[i][6]))
            worst.append(-float(datContent[i][7]))

        logs_df = pd.DataFrame()
        logs_df['Generations'] = generations
        logs_df['Evaluations'] = evaluations
        logs_df['BestEver'] = bestever
        logs_df['Best'] = best
        logs_df['Median'] = median
        logs_df['Worst'] = worst

        logs_df.to_csv('logs.csv')

        plt.plot(generations, best, color='green')
        plt.plot(generations, median, color='blue')
        plt.xlabel("Number of generations")
        plt.ylabel("Fitness")
        plt.legend(["Best", "Median"])
        plt.title('Evolution of fitness across generations')
        plt.savefig('LearningCurve.jpg')
        plt.close()

