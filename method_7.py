
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import os
from collections import deque
import gymnasium as gym
import pandas as pd

# Default weights path (relative to this file)
# p = "best_weights_3relu_3hidden.pth"
# p = "dqn_weights.pth"
p = "dqn_weights_ep1200.pth"
DEFAULT_WEIGHTS_PATH = os.path.join(os.path.dirname(__file__), p)


class QNetwork(nn.Module):
    """Deep q network to apprx q values for the continous state space"""
    def __init__(self, state_dim, action_dim, hidden_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )

    def forward(self, x):
        return self.fc(x)


class DQNAgent:
    def __init__(self, state_dim, action_dim, lr, gamma,
                epsilon_start, epsilon_min, epsilon_decay,
                 memory_size, hidden_dim, weights_path=None):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.memory = deque(maxlen=memory_size)

        # Networks
        self.policy_net = QNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        self.target_net = QNetwork(state_dim, action_dim, hidden_dim).to(self.device)

        # Optionally load pre-trained weights
        if weights_path is not None and os.path.exists(weights_path):
            checkpoint = torch.load(weights_path, map_location=self.device)
            self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
            print(f"Loaded weights from {weights_path}")

        self.target_net.load_state_dict(self.policy_net.state_dict())

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.loss_fn = nn.SmoothL1Loss()

    @classmethod
    def load_for_eval(cls, weights_path=None):
        """
        Load a pre-trained agent for evaluation only.
        
        Args:
            weights_path: Path to weights file (uses default if None)
            
        Returns:
            DQNAgent instance ready for evaluation
        """
        if weights_path is None:
            weights_path = DEFAULT_WEIGHTS_PATH

        if not os.path.exists(weights_path):
            raise FileNotFoundError(
                f"weights not found:{weights_path}\n"
            )

        # Load checkpoint to get hyperparameters
        checkpoint = torch.load(weights_path, map_location='cpu')
        state_dim = checkpoint.get('state_dim')
        action_dim = checkpoint.get('action_dim')
        hidden_dim = checkpoint.get('hidden_dim')

        print(f"Loading for eval - State dim: {state_dim}, Action dim: {action_dim}, Hidden dim: {hidden_dim}")

        # Create agent with dummy training params (not used in eval)
        agent = cls(
            state_dim=state_dim,
            action_dim=action_dim,
            lr=0.0,  # Not used in eval
            gamma=0.0,  # Not used in eval
            epsilon_start=0.0,  # No exploration in eval
            epsilon_min=0.0,
            epsilon_decay=1.0,
            memory_size=1,  # Minimal memory
            hidden_dim=hidden_dim,
            weights_path=weights_path
        )
        agent.policy_net.eval()
        return agent

    def select_action(self, state, training=False):
        """Epsilon-greedy action selection."""
        if training and random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)
        else:
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            with torch.no_grad():
                return torch.argmax(self.policy_net(state_tensor)).item()

    def compute_action(self, observation):
        """
        this function is just here so that i dont have to change main.py
        """
        return self.select_action(observation, training=False)

    def store_transition(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train_step(self, batch_size):
        if len(self.memory) < batch_size:
            return None

        batch = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)

        # Current Q values
        current_q = self.policy_net(states).gather(1, actions)

        # Bellman equation
        with torch.no_grad():
            next_q = self.target_net(next_states).max(1)[0].unsqueeze(1)
            target_q = rewards + (self.gamma * next_q * (1 - dones))

        # Optimize
        loss = self.loss_fn(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()

        return loss.item()

    def update_target_network(self):
        """Copy policy network weights to target network."""
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def decay_epsilon(self):
        """Decay exploration rate."""
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)

    def save(self, path):
        """Save model weights and hyperparameters."""
        torch.save({
            'state_dim': self.state_dim,
            'action_dim': self.action_dim,
            'policy_net_state_dict': self.policy_net.state_dict(),
            'hidden_dim': self.policy_net.fc[0].out_features,
        }, path)
        print(f"Model saved to {path}")


def train(episodes, save_path, render_interval, lr, gamma, batch_size, 
          target_update_freq, save_interval, epsilon_start, epsilon_min, 
          epsilon_decay, memory_size, hidden_dim, load_weights=None):
    """Train the DQN agent."""
    
    
    env = gym.make("LunarLander-v3")
    agent = DQNAgent(
        state_dim=8, 
        action_dim=4, 
        lr=lr, 
        gamma=gamma,
        epsilon_start=epsilon_start,
        epsilon_min=epsilon_min,
        epsilon_decay=epsilon_decay,
        memory_size=memory_size,
        hidden_dim=hidden_dim,
        weights_path=load_weights
    )

    best_reward = float('-inf')
    reward_history = []
    
    # DataFrame to store training results

    results_df = pd.DataFrame(columns=['episode', 'reward', 'avg_reward_100', 'epsilon'])

    for episode in range(episodes):
        # Optionally render
        if render_interval and episode % render_interval == 0:
            env.close()
            env = gym.make("LunarLander-v3", render_mode="human")
        elif render_interval and episode % render_interval == 1:
            env.close()
            env = gym.make("LunarLander-v3")

        state, _ = env.reset()
        total_reward = 0
        done = False

        while not done:
            action = agent.select_action(state, training=True)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            agent.store_transition(state, action, reward, next_state, done)
            agent.train_step(batch_size)

            state = next_state
            total_reward += reward

        agent.decay_epsilon()
        reward_history.append(total_reward)
        
        # Store in DataFrame
        avg_reward = np.mean(reward_history[-100:]) if reward_history else 0
        results_df.loc[episode] = [episode, total_reward, avg_reward, agent.epsilon]

        # Update target network
        if episode % target_update_freq == 0:
            agent.update_target_network()

        # Logging
        if episode % 10 == 0:
            print(f"Episode {episode:4d} | Reward: {total_reward:7.2f} | "
                  f"Avg(100): {avg_reward:7.2f} | Epsilon: {agent.epsilon:.3f}")

        # Keep saving the best model
        if total_reward > best_reward:
            best_reward = total_reward
            agent.save(save_path.replace('.pth', '_best.pth'))

        if save_interval and episode % save_interval == 0 and episode > 0:
            agent.save(save_path.replace('.pth', f'_ep{episode}.pth'))

    # Final save
    agent.save(save_path)
    env.close()
    
    # Save training results to CSV
    csv_path = save_path.replace('.pth', '_training_log.csv')
    results_df.to_csv(csv_path, index=False)
    print(f"Training log saved to {csv_path}")

    print("\n" + "=" * 50)
    print("TRAINING COMPLETE")
    print("=" * 50)
    print(f"  Episodes:     {episodes}")
    print(f"  Best Reward:  {best_reward:.2f}")
    print(f"  Final Avg:    {np.mean(reward_history[-100:]):.2f}")
    print(f"  Model saved:  {save_path}")
    print("=" * 50)

    return agent, reward_history, results_df


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="DQN Agent for LunarLander - Train or Evaluate")
    parser.add_argument("--train", action="store_true", help="Train the agent (otherwise just load for eval)")
    parser.add_argument("--episodes", type=int, default=2000, help="Number of training episodes")
    parser.add_argument("--save_path", type=str, default="dqn_weights.pth", help="Path to save weights")
    parser.add_argument("--load_weights", type=str, default=None, help="Path to load pre-trained weights (optional)")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training")
    parser.add_argument("--target_update_freq", type=int, default=5, help="Frequency to update target network")
    parser.add_argument("--save_interval", type=int, default=100, help="Frequency to save checkpoints")
    parser.add_argument("--epsilon_start", type=float, default=1.0, help="Initial exploration rate")
    parser.add_argument("--epsilon_min", type=float, default=0.02, help="Minimum exploration rate")
    parser.add_argument("--epsilon_decay", type=float, default=0.995, help="Exploration decay rate")
    parser.add_argument("--memory_size", type=int, default=20000, help="Replay buffer size")
    parser.add_argument("--hidden_dim", type=int, default=128, help="Hidden layer dimension")
    parser.add_argument("--render", action="store_true", help="Render every 200 episodes during training")
    args = parser.parse_args()

    if args.train:
        render_interval = 200 if args.render else None
        train(
            episodes=args.episodes,
            save_path=args.save_path,
            render_interval=render_interval,
            lr=args.lr,
            gamma=args.gamma,
            batch_size=args.batch_size,
            target_update_freq=args.target_update_freq,
            save_interval=args.save_interval,
            epsilon_start=args.epsilon_start,
            epsilon_min=args.epsilon_min,
            epsilon_decay=args.epsilon_decay,
            memory_size=args.memory_size,
            hidden_dim=args.hidden_dim,
            load_weights=args.load_weights,
        )
    else:
        print("bruh why are you running method_7.py directly bro. use --train")