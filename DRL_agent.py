


import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
from collections import deque
import os
import h5py

# Set random seeds for reproducibility
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()

        # Define the network layers with increased complexity
        self.layer1 = nn.Linear(state_dim, 512)
        self.layer2 = nn.Linear(512, 384)
        self.layer3 = nn.Linear(384, 256)
        self.layer4 = nn.Linear(256, 128)
        self.layer5 = nn.Linear(128, 64)
        self.layer6 = nn.Linear(64, action_dim)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Apply Xavier initialization to the layers
        torch.nn.init.xavier_uniform_(self.layer1.weight)
        torch.nn.init.xavier_uniform_(self.layer2.weight)
        torch.nn.init.xavier_uniform_(self.layer3.weight)
        torch.nn.init.xavier_uniform_(self.layer4.weight)
        torch.nn.init.xavier_uniform_(self.layer5.weight)
        torch.nn.init.xavier_uniform_(self.layer6.weight)
        
        # Define the scaling factors for each action
        self.momentum_wheel_scale = 160 * 2 * np.pi / 60  # Convert 160 RPM to radians/sec
        self.back_wheel_scale = 60 * 2 * np.pi / 60  # Convert 60 RPM to radians/sec
        self.steering_scale = np.radians(20)  # Convert 30 degrees to radians
        
    def forward(self, state):
        # Pass the state through the layers with ReLU activations
        x = F.relu(self.layer1(state))
        x = F.relu(self.layer2(x))
        x = F.relu(self.layer3(x))
        x = F.relu(self.layer4(x))
        x = F.relu(self.layer5(x))
        x = torch.tanh(self.layer6(x))  # Output in range [-1, 1]
        
        # Scale the output to the desired range for each action
        momentum_wheel_action = x[:, 0] * self.momentum_wheel_scale
        back_wheel_action = x[:, 1] * self.back_wheel_scale
        steering_action = x[:, 2] * self.steering_scale
        
        # Concatenate the scaled actions into a single tensor
        actions = torch.stack([momentum_wheel_action, back_wheel_action, steering_action], dim=1)
        return actions



class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()

        # Q1 architecture
        self.layer1 = nn.Linear(state_dim + action_dim, 512)
        self.layer2 = nn.Linear(512, 384)
        self.layer3 = nn.Linear(384, 256)
        self.layer4 = nn.Linear(256, 128)
        self.layer5 = nn.Linear(128, 64)
        self.layer6 = nn.Linear(64, 1)

        # Q2 architecture
        self.layer7 = nn.Linear(state_dim + action_dim, 512)
        self.layer8 = nn.Linear(512, 384)
        self.layer9 = nn.Linear(384, 256)
        self.layer10 = nn.Linear(256, 128)
        self.layer11 = nn.Linear(128, 64)
        self.layer12 = nn.Linear(64, 1)

        # Apply Xavier initialization to the layers
        torch.nn.init.xavier_uniform_(self.layer1.weight)
        torch.nn.init.xavier_uniform_(self.layer2.weight)
        torch.nn.init.xavier_uniform_(self.layer3.weight)
        torch.nn.init.xavier_uniform_(self.layer4.weight)
        torch.nn.init.xavier_uniform_(self.layer5.weight)
        torch.nn.init.xavier_uniform_(self.layer6.weight)
        torch.nn.init.xavier_uniform_(self.layer7.weight)
        torch.nn.init.xavier_uniform_(self.layer8.weight)
        torch.nn.init.xavier_uniform_(self.layer9.weight)
        torch.nn.init.xavier_uniform_(self.layer10.weight)
        torch.nn.init.xavier_uniform_(self.layer11.weight)
        torch.nn.init.xavier_uniform_(self.layer12.weight)

    def forward(self, state, action):
        sa = torch.cat([state, action], 1)

        # Q1 computation
        q1 = F.relu(self.layer1(sa))
        q1 = F.relu(self.layer2(q1))
        q1 = F.relu(self.layer3(q1))
        q1 = F.relu(self.layer4(q1))
        q1 = F.relu(self.layer5(q1))
        q1 = self.layer6(q1)

        # Q2 computation
        q2 = F.relu(self.layer7(sa))
        q2 = F.relu(self.layer8(q2))
        q2 = F.relu(self.layer9(q2))
        q2 = F.relu(self.layer10(q2))
        q2 = F.relu(self.layer11(q2))
        q2 = self.layer12(q2)

        return q1, q2

    def Q1(self, state, action):
        sa = torch.cat([state, action], 1)

        q1 = F.relu(self.layer1(sa))
        q1 = F.relu(self.layer2(q1))
        q1 = F.relu(self.layer3(q1))
        q1 = F.relu(self.layer4(q1))
        q1 = F.relu(self.layer5(q1))
        q1 = self.layer6(q1)

        return q1



class ReplayBuffer:
    def __init__(self, max_size=1e6):
        self.storage = deque(maxlen=int(max_size))

    def add(self, data):
        self.storage.append(data)

    def sample(self, batch_size):
        batch = random.sample(self.storage, batch_size)
        state, action, next_state, reward, done = map(np.stack, zip(*batch))
        return (
            torch.FloatTensor(state),
            torch.FloatTensor(action),
            torch.FloatTensor(next_state),
            torch.FloatTensor(reward).unsqueeze(1),
            torch.FloatTensor(done).unsqueeze(1)
        )


class DRLAgent:
    def __init__(self, state_dim, action_dim, epsilon=1, epsilon_decay=0.995, epsilon_min=0.01, checkpoint_dir="checkpoints"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.actor = Actor(state_dim, action_dim).to(self.device)
        self.actor_target = Actor(state_dim, action_dim).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=1e-4)  # Reduced learning rate

        self.critic = Critic(state_dim, action_dim).to(self.device)
        self.critic_target = Critic(state_dim, action_dim).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=1e-4)  # Reduced learning rate

        self.replay_buffer = ReplayBuffer()
        self.discount = 0.99
        self.tau = 0.005
        self.policy_freq = 20  # Update actor policy every policy_freq time
        self.total_it = 0

        # Epsilon-greedy parameters
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        
        # Checkpoint directory
        self.checkpoint_dir = checkpoint_dir
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)
        
    def select_action(self, state):
        if np.random.rand() < self.epsilon:
            # Reduced random action range for safer exploration
            action = np.random.uniform(-1, 1, size=(3,))
            action[0] *= self.actor.momentum_wheel_scale  # Momentum wheel
            action[1] *= self.actor.back_wheel_scale   # Back wheel
            action[2] *= self.actor.steering_scale     # Steering
        else:
            state_tensor = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
            action = self.actor(state_tensor).cpu().data.numpy().flatten()
        
        return action
    
    def exploit(self, state):
        state_tensor = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        action = self.actor(state_tensor).cpu().data.numpy().flatten()
        
        return action
    
    def load_actor(self, filepath):
        with h5py.File(filepath, 'r') as h5_file:
            self._load_layer_from_h5(self.actor.layer1, h5_file, 'layer1')
            self._load_layer_from_h5(self.actor.layer2, h5_file, 'layer2')
            self._load_layer_from_h5(self.actor.layer3, h5_file, 'layer3')
        print(f"Actor model weights loaded from {filepath}")

    def load_critic(self, filepath):
        with h5py.File(filepath, 'r') as h5_file:
            self._load_layer_from_h5(self.critic.layer1, h5_file, 'layer1')
            self._load_layer_from_h5(self.critic.layer2, h5_file, 'layer2')
            self._load_layer_from_h5(self.critic.layer3, h5_file, 'layer3')
            self._load_layer_from_h5(self.critic.layer4, h5_file, 'layer4')
            self._load_layer_from_h5(self.critic.layer5, h5_file, 'layer5')
            self._load_layer_from_h5(self.critic.layer6, h5_file, 'layer6')
        print(f"Critic model weights loaded from {filepath}")

    def _load_layer_from_h5(self, layer, h5_file, layer_name):
        weight = torch.tensor(h5_file[f"{layer_name}/weight"][:])
        bias = torch.tensor(h5_file[f"{layer_name}/bias"][:])
        layer.weight.data.copy_(weight)
        layer.bias.data.copy_(bias)

    def update_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def train(self, batch_size=64, episode=0):  # Increased batch size
        if len(self.replay_buffer.storage) < batch_size:
            return  # Wait until the replay buffer has enough samples

        self.total_it += 1

        # Sample replay buffer
        state, action, next_state, reward, done = self.replay_buffer.sample(batch_size)
        state = state.to(self.device)
        action = action.to(self.device)
        next_state = next_state.to(self.device)
        reward = reward.to(self.device)
        done = done.to(self.device)

        with torch.no_grad():
            # Select action according to policy
            next_action = self.actor_target(next_state)

            # Compute the target Q value
            target_q1, target_q2 = self.critic_target(next_state, next_action)
            target_q = torch.min(target_q1, target_q2)
            target_q = reward + ((1 - done) * self.discount * target_q).detach()

        # Get current Q estimates
        current_q1, current_q2 = self.critic(state, action)

        # Compute critic loss
        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Delayed policy updates
        if self.total_it % self.policy_freq == 0:
            # Compute actor loss
            actor_loss = -self.critic.Q1(state, self.actor(state)).mean()

            # Optimize the actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Update the frozen target models
            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        # Save checkpoint every 50 episodes
        if episode % 50 == 0:
            self.save_checkpoint(episode)
            
    def save_checkpoint(self, episode):
        checkpoint_path = os.path.join(self.checkpoint_dir, f"checkpoint_{episode}.pth")
        checkpoint = {
            'actor_state_dict': self.actor.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
            'actor_target_state_dict': self.actor_target.state_dict(),
            'critic_target_state_dict': self.critic_target.state_dict(),
            'epsilon': self.epsilon,
            'episode': episode
        }
        torch.save(checkpoint, checkpoint_path)
        print(f"Checkpoint saved at {checkpoint_path}")

    def load_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
        self.actor_target.load_state_dict(checkpoint['actor_target_state_dict'])
        self.critic_target.load_state_dict(checkpoint['critic_target_state_dict'])
        self.epsilon = checkpoint['epsilon']
        print(f"Checkpoint loaded from {checkpoint_path}")
    
    def save_to_h5(self):
        def save_layer_to_h5(layer, h5_file, layer_name):
            weight, bias = layer.weight.data.cpu().numpy(), layer.bias.data.cpu().numpy()
            h5_file.create_dataset(f"{layer_name}/weight", data=weight)
            h5_file.create_dataset(f"{layer_name}/bias", data=bias)

        # Save Actor network
        with h5py.File('actor.h5', 'w') as h5_file:
            for i, layer in enumerate([self.actor.layer1, self.actor.layer2, self.actor.layer3], start=1):
                save_layer_to_h5(layer, h5_file, f'layer{i}')

        # Save Critic network
        with h5py.File('critic.h5', 'w') as h5_file:
            for i, layer in enumerate([self.critic.layer1, self.critic.layer2, self.critic.layer3,
                                    self.critic.layer4, self.critic.layer5, self.critic.layer6], start=1):
                save_layer_to_h5(layer, h5_file, f'layer{i}')
            
        print("Saved actor and critic networks to .h5 files")