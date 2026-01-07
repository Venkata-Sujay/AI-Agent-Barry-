import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque

# --- THE BRAIN (Convolutional Neural Network) ---
class DuelingCNNetwork(nn.Module):
    def __init__(self, input_shape, num_actions):
        super(DuelingCNNetwork, self).__init__()
        
        # 1. The "Eyes" (Convolutional Layers)
        # We process the 84x84 image to find patterns (edges, shapes)
        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        # Calculate the size of the output from conv layers
        # For 84x84 input, this flattens to 3136 features
        conv_out_size = self._get_conv_out(input_shape)

        # 2. The "Decision Maker" (Fully Connected Layers)
        self.fc_adv = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, num_actions)
        )
        
        self.fc_val = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )

    def _get_conv_out(self, shape):
        o = torch.zeros(1, *shape)
        o = self.conv(o)
        return int(np.prod(o.size()))

    def forward(self, x):
        conv_out = self.conv(x).view(x.size()[0], -1)
        val = self.fc_val(conv_out)
        adv = self.fc_adv(conv_out)
        # Dueling Architecture magic: Value + (Advantage - Mean Advantage)
        return val + (adv - adv.mean(dim=1, keepdim=True))

# --- THE AGENT ---
class DoubleDQNAgent:
    def __init__(self, input_shape, num_actions, lr=1e-4, gamma=0.99, buffer_size=10000):
        self.num_actions = num_actions
        self.gamma = gamma
        self.epsilon = 1.0  # Exploration rate (starts high, decreases over time)
        self.epsilon_min = 0.02
        self.epsilon_decay = 1e-5
        self.batch_size = 32
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"🧠 Brain running on: {self.device}")

        # Double DQN needs TWO networks
        self.online_net = DuelingCNNetwork(input_shape, num_actions).to(self.device)
        self.target_net = DuelingCNNetwork(input_shape, num_actions).to(self.device)
        self.target_net.load_state_dict(self.online_net.state_dict())

        self.optimizer = optim.Adam(self.online_net.parameters(), lr=lr)
        self.memory = deque(maxlen=buffer_size)

    def choose_action(self, state):
        # Epsilon-Greedy Strategy
        if np.random.random() < self.epsilon:
            return np.random.randint(self.num_actions)  # Random move (Explore)
        
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.online_net(state_t)
            return torch.argmax(q_values).item()  # Best move (Exploit)

    def store_transition(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def learn(self):
        if len(self.memory) < 1000: return  # Wait until we have enough data
        
        # Sample a batch of memories
        batch = random.sample(self.memory, self.batch_size)
        state, action, reward, next_state, done = zip(*batch)

        state = torch.FloatTensor(np.array(state)).to(self.device)
        action = torch.LongTensor(action).unsqueeze(1).to(self.device)
        reward = torch.FloatTensor(reward).unsqueeze(1).to(self.device)
        next_state = torch.FloatTensor(np.array(next_state)).to(self.device)
        done = torch.FloatTensor(done).unsqueeze(1).to(self.device)

        # --- DOUBLE DQN LOGIC ---
        # 1. Select best action using ONLINE net
        next_actions = self.online_net(next_state).argmax(dim=1, keepdim=True)
        # 2. Evaluate that action using TARGET net
        next_q_values = self.target_net(next_state).gather(1, next_actions)
        
        target_q_values = reward + (1 - done) * self.gamma * next_q_values
        
        # Current Q values
        current_q_values = self.online_net(state).gather(1, action)

        # Loss Calculation
        loss = nn.MSELoss()(current_q_values, target_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Decay exploration
        self.epsilon = max(self.epsilon_min, self.epsilon - self.epsilon_decay)

    def update_target_net(self):
        self.target_net.load_state_dict(self.online_net.state_dict())