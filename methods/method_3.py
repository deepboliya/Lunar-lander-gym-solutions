import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import os
from collections import deque
from pathlib import Path
from methods.base_controller import BaseController

WEIGHTS_DIR = Path(__file__).parent / "weights"

MODULE_CONFIG = {
    'class_name': 'DQNAgent',
    'params_file': None,
    'weights_file': 'best_method_7.pth',
    'discrete_action': True,
}


class QNetwork(nn.Module):
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


class DQNAgent(BaseController):
    # Intuition: Just a typical Q network.
    def __init__(self, gravity_magnitude, params, weights):
        super().__init__(params=params, weights=weights, gravity_magnitude=gravity_magnitude)
        
        self.state_dim = 8
        self.action_dim = 4
        self.lr = 0.001
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.memory_size = 10000
        
        if self.weights is not None:
            self.hidden_dim = self.weights.get('hidden_dim', 64)
        else:
            self.hidden_dim = 128
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.memory = deque(maxlen=self.memory_size)

        self.policy_net = QNetwork(self.state_dim, self.action_dim, self.hidden_dim).to(self.device)
        self.target_net = QNetwork(self.state_dim, self.action_dim, self.hidden_dim).to(self.device)

        if self.weights is not None:
            self.policy_net.load_state_dict(self.weights['policy_net_state_dict'])

        self.target_net.load_state_dict(self.policy_net.state_dict())

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.lr)
        self.loss_fn = nn.SmoothL1Loss()

    @classmethod
    def load_for_eval(cls, weights_path=None, gravity_magnitude=10.0):
        if weights_path is None:
            weights_path = WEIGHTS_DIR / "method_7_best.pth"
        
        weights_path = Path(weights_path)
        checkpoint = torch.load(weights_path, map_location='cpu')
        
        agent = cls(weights=checkpoint, gravity_magnitude=gravity_magnitude)
        agent.policy_net.eval()
        return agent

    def select_action(self, state, training=False):
        if training and random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)
        else:
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            with torch.no_grad():
                return torch.argmax(self.policy_net(state_tensor)).item()

    def compute_action(self, observation):
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

        current_q = self.policy_net(states).gather(1, actions)

        with torch.no_grad():
            next_q = self.target_net(next_states).max(1)[0].unsqueeze(1)
            target_q = rewards + (self.gamma * next_q * (1 - dones))

        loss = self.loss_fn(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()

        return loss.item()

    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)

    def save(self, path):
        torch.save({
            'state_dim': self.state_dim,
            'action_dim': self.action_dim,
            'policy_net_state_dict': self.policy_net.state_dict(),
            'hidden_dim': self.policy_net.fc[0].out_features,
        }, path)
