# Run `pip install "gymnasium[classic-control]"` for this example.
import gymnasium as gym
import numpy as np
import math
import time
import matplotlib.pyplot as plt
from controller import MainController
from controller import GRAVITY_MAGNITUDE
import pandas as pd

env = gym.make("LunarLander-v3", continuous=True, gravity=-GRAVITY_MAGNITUDE, render_mode="human")

observation, info = env.reset()

print(f"Starting observation: {observation}")
all_params={"x_position": (1.0, 0.0, 0.0),
            "y_position": (0.2, 0.0, 0.0),
            "x_velocity": (1.0, 0.0, 0.1),
            "y_velocity": (1.0, 0.0, 0.1),
            "angle":      (15.0, 0.0, 2.0),
            "angle_rate": (10.0, 0.0, 1.0)}
# flattened_params = np.array([val for sublist in all_params.values() for val in sublist])

# flattened_params = [0.0201632737,2.7968016213,-1.7746332775,2.2254618276,0.8571352612,2.0366640502,-1.1598781474,2.2414483106,2.1165370135,-0.648357219,-0.8171384849,-0.9515809643,0.8150724454,-0.0275258006,1.1750177064,-1.4873066879,-1.144660962,-1.2778310145]

param_df = pd.read_json("cmaes_params.json")
flattened_params = np.array(param_df.iloc[0]["Params"])
print("Using flattened params:", flattened_params)
hierchal_controller = MainController(flattened_params)

step_count = 0
episode_over = False
total_reward = 0
while not episode_over:

    action = hierchal_controller.compute_action(observation)
    
    observation, reward, terminated, truncated, info = env.step(action)

    step_count += 1

    total_reward += reward
    episode_over = terminated or truncated
    # time.sleep(0.1)

print(f"Episode finished! Total reward: {total_reward}")
env.close()

#Plotting
history = hierchal_controller.get_history() 
t = history["t"]

fig, axes = plt.subplots(4, 2, figsize=(12, 11), sharex=True)

# Row 0: Position (x, y) with target = 0
axes[0, 0].plot(t, history["x"], label="x")
axes[0, 0].axhline(0, color="tab:red", linestyle="--", linewidth=1, label="target x=0")
axes[0, 0].set_ylabel("x (position)")
axes[0, 0].legend()
axes[0, 0].grid(True)

axes[0, 1].plot(t, history["y"], label="y", color="tab:orange")
axes[0, 1].axhline(0, color="tab:red", linestyle="--", linewidth=1, label="target y=0")
axes[0, 1].set_ylabel("y (position)")
axes[0, 1].legend()
axes[0, 1].grid(True)

# Row 1: Velocity (vx, vy) with target_vx, target_vy
axes[1, 0].plot(t, history["vx"], label="vx")
axes[1, 0].plot(t, history["target_vx"], label="target_vx", linestyle="--", color="tab:red")
axes[1, 0].axhline(0, color="gray", linestyle=":", linewidth=0.5)
axes[1, 0].set_ylabel("vx (velocity)")
axes[1, 0].legend()
axes[1, 0].grid(True)

axes[1, 1].plot(t, history["vy"], label="vy", color="tab:orange")
axes[1, 1].plot(t, history["target_vy"], label="target_vy", linestyle="--", color="tab:red")
axes[1, 1].axhline(0, color="gray", linestyle=":", linewidth=0.5)
axes[1, 1].set_ylabel("vy (velocity)")
axes[1, 1].legend()
axes[1, 1].grid(True)

# Row 2: Commanded acceleration (target_ax, target_ay)
axes[2, 0].plot(t, history["target_ax"], label="target_ax")
axes[2, 0].axhline(0, color="gray", linestyle=":", linewidth=0.5)
axes[2, 0].set_ylabel("ax (cmd)")
axes[2, 0].legend()
axes[2, 0].grid(True)

axes[2, 1].plot(t, history["target_ay"], label="target_ay", color="tab:orange")
axes[2, 1].axhline(0, color="gray", linestyle=":", linewidth=0.5)
axes[2, 1].set_ylabel("ay (cmd)")
axes[2, 1].legend()
axes[2, 1].grid(True)

# Row 3: Target thrust and torque
axes[3, 0].plot(t, history["target_thrust"], label="target_thrust", color="tab:green")
axes[3, 0].axhline(0, color="gray", linestyle=":", linewidth=0.5)
axes[3, 0].set_ylabel("Thrust (cmd)")
axes[3, 0].set_xlabel("Time (s)")
axes[3, 0].legend()
axes[3, 0].grid(True)

axes[3, 1].plot(t, history["target_torque"], label="target_torque", color="tab:purple")
axes[3, 1].axhline(0, color="gray", linestyle=":", linewidth=0.5)
axes[3, 1].set_ylabel("Torque (cmd)")
axes[3, 1].set_xlabel("Time (s)")
axes[3, 1].legend()
axes[3, 1].grid(True)

fig.suptitle(f"LunarLander PID Run — Total Reward: {total_reward:.1f}")
plt.tight_layout()
plt.show()