# Run `pip install "gymnasium[classic-control]"` for this example.
import gymnasium as gym
import numpy as np
import math
import time
import matplotlib.pyplot as plt
from method_4 import MainController2
from main import GRAVITY_MAGNITUDE
# Create our training environment - a cart with a pole that needs balancing
env = gym.make("LunarLander-v3", continuous=True, gravity=-GRAVITY_MAGNITUDE, render_mode="human")
# Reset environment to start a new episode
observation, info = env.reset()

print(f"Starting observation: {observation}")

hierchal_controller = MainController2(gravity_magnitude=GRAVITY_MAGNITUDE, print_=True)

step_count = 0
episode_over = False
total_reward = 0
while not episode_over:

    # action = 1 if action > 0 else 0
    action = hierchal_controller.compute_action(observation)
    observation, reward, terminated, truncated, info = env.step(action)

    step_count += 1

    # print(f"Terminated: {terminated}, Truncated: {truncated}, info: {info}")
    total_reward += reward
    episode_over = terminated or truncated
    # time.sleep(0.1)

print(f"Episode finished! Total reward: {total_reward}")
env.close()

# ─────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────
history = hierchal_controller.get_history() 
t = history["t"]

fig, axes = plt.subplots(3, 2, figsize=(12, 9), sharex=True)

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
axes[2, 0].set_xlabel("Time (s)")
axes[2, 0].legend()
axes[2, 0].grid(True)

axes[2, 1].plot(t, history["target_ay"], label="target_ay", color="tab:orange")
axes[2, 1].axhline(0, color="gray", linestyle=":", linewidth=0.5)
axes[2, 1].set_ylabel("ay (cmd)")
axes[2, 1].set_xlabel("Time (s)")
axes[2, 1].legend()
axes[2, 1].grid(True)

fig.suptitle(f"LunarLander PID Run — Total Reward: {total_reward:.1f}")
plt.tight_layout()
plt.show()