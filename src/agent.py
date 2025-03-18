# agent.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DQN(nn.Module):
    """
    Simple MLP network for DQN.
    """
    def __init__(self, input_dim, output_dim, hidden_dims=[128, 128]):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dims[0])
        self.fc2 = nn.Linear(hidden_dims[0], hidden_dims[1])
        self.out = nn.Linear(hidden_dims[1], output_dim)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.out(x)

class DQNAgent:
    """
    DQN agent with epsilon-greedy exploration and cost-sensitive loss.
    """
    def __init__(self, input_dim, num_actions, lr=1e-3, gamma=0.99, epsilon=1.0,
                 epsilon_min=0.1, epsilon_decay=0.995, cost_weights=None, device='cpu'):
        self.input_dim = input_dim
        self.num_actions = num_actions
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.device = device
        
        self.policy_net = DQN(input_dim, num_actions).to(self.device)
        self.target_net = DQN(input_dim, num_actions).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.lr)
        
        if cost_weights is None:
            self.cost_weights = torch.ones(num_actions).to(self.device)
        else:
            self.cost_weights = torch.tensor(cost_weights, dtype=torch.float32, device=self.device)
            
    def select_action(self, state):
        """
        Select an action using epsilon-greedy.
        """
        if random.random() < self.epsilon:
            return random.randrange(self.num_actions)
        else:
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            with torch.no_grad():
                q_values = self.policy_net(state_tensor)
            return int(q_values.argmax().item())
    
    def update_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
    
    def compute_loss(self, batch, weights):
        """
        Compute weighted MSE loss with cost-sensitive weighting.
        """
        states, actions, rewards, next_states, dones = batch
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)
        weights = torch.FloatTensor(weights).unsqueeze(1).to(self.device)

        current_q = self.policy_net(states).gather(1, actions)
        with torch.no_grad():
            next_actions = self.policy_net(next_states).argmax(dim=1, keepdim=True)
            next_q = self.target_net(next_states).gather(1, next_actions)
            target_q = rewards + self.gamma * next_q * (1 - dones)

        action_cost = self.cost_weights[actions].unsqueeze(1)
        loss = (current_q - target_q).pow(2) * weights * action_cost
        loss = loss.mean()
        return loss

    def optimize_model(self, replay_buffer, batch_size, beta):
        states, actions, rewards, next_states, dones, indices, weights = replay_buffer.sample(batch_size, beta)
        loss = self.compute_loss((states, actions, rewards, next_states, dones), weights)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()
    
    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())
