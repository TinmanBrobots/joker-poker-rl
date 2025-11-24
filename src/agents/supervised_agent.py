import numpy as np
import random
from .base_agent import BaseAgent
from typing import Any, Dict, Optional
import pickle
import os
import torch
import torch.nn as nn
from collections import deque

class PokerNet(nn.Module):
	def __init__(self, input_dim: int = 52, hidden_dims: list = [128, 64], output_dim: int = 32, dropout_rate: float = 0.1):
		super().__init__()

		layers = [nn.Flatten()]

		# Input layer
		layers.extend([
			nn.Linear(input_dim, hidden_dims[0]),
			nn.ReLU(),
			nn.Dropout(dropout_rate)
		])

		# Hidden layers
		for i in range(len(hidden_dims) - 1):
			layers.extend([
				nn.Linear(hidden_dims[i], hidden_dims[i+1]),
				nn.ReLU(),
				nn.Dropout(dropout_rate)
			])

		# Output layer
		layers.append(nn.Linear(hidden_dims[-1], output_dim))

		self.network = nn.Sequential(*layers)

	def forward(self, x):
		return self.network(x)


class LinearPokerNet(nn.Module):
	"""Simple linear model - sometimes works better than deep nets for tabular data"""
	def __init__(self, input_dim: int = 52, output_dim: int = 32):
		super().__init__()
		self.network = nn.Sequential(
			nn.Flatten(),
			nn.Linear(input_dim, output_dim)
		)

	def forward(self, x):
		return self.network(x)

	def forward(self, x):
		return self.network(x)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		return self.network(x)

class SupervisedAgent(BaseAgent):
	"""
	Supervised learning agent for Jacks-or-Better poker.

	Uses supervised learning to learn the optimal policy.
	State is represented as a 13x4 binary array.
	"""

	def __init__(
		self,
		observation_space,
		action_space,
		network: PokerNet,
		learning_rate: float = 0.01,  # Increased from 0.001
		batch_size: int = 128,        # Increased from 32
		buffer_size: int = 100000,    # Increased from 50000
		epsilon_start: float = 1.0,
		epsilon_end: float = 0.05,    # Increased from 0.01 for more exploration
		epsilon_decay: float = 0.9999, # Slower decay
		weight_decay: float = 1e-4,   # L2 regularization
		seed: Optional[int] = None,
	):
		self.observation_space = observation_space
		self.action_space = action_space
		self.network = network

		# Use AdamW with weight decay for better regularization
		self.optimizer = torch.optim.AdamW(
			self.network.parameters(),
			lr=learning_rate,
			weight_decay=weight_decay
		)

		# Use Huber loss for robustness to outliers
		self.loss_fn = nn.SmoothL1Loss()  # Changed from MSE

		self.batch_size = batch_size
		self.buffer_size = buffer_size
		self.replay_buffer = deque(maxlen=buffer_size)
		self.rng = np.random.RandomState(seed)

		self.epsilon = epsilon_start
		self.epsilon_end = epsilon_end
		self.epsilon_decay = epsilon_decay

		# Training stats
		self.training_steps = 0
	

	def act(self, observation: np.ndarray) -> int:
		"""
		Choose action using the network.

		Args:
			observation: Current state observation

		Returns:
			action: Integer action to take
		"""
		if self.rng.random() < self.epsilon:
			return self.rng.randint(0, self.action_space.n)
		else:
			state_tensor = torch.from_numpy(observation).float().unsqueeze(0)
			with torch.no_grad():
				predictions = self.network(state_tensor)
				action = torch.argmax(predictions).item()
			return action


	def learn(self, experience: Dict[str, Any]) -> Optional[Dict[str, float]]:
		"""
		Update network using supervised learning with experience replay.

		Args:
			experience: Dict with keys 'state', 'action', 'reward'

		Returns:
			metrics: Dict with training metrics, or None if no training occurred
		"""
		# Add experience to replay buffer
		state, action, reward = experience['state'], experience['action'], experience['reward']
		self.replay_buffer.append((state, action, reward))

		# Only train if we have enough experiences
		if len(self.replay_buffer) < self.batch_size:
			return None

		# Sample random batch from replay buffer
		batch = random.sample(self.replay_buffer, self.batch_size)
		states, actions, rewards = zip(*batch)

		# Convert to tensors
		states = torch.from_numpy(np.array(states)).float()
		actions = torch.from_numpy(np.array(actions)).long()
		rewards = torch.from_numpy(np.array(rewards)).float()

		# Forward pass
		q_values = self.network(states)

		# Get predicted Q-values for the actions taken
		predicted_q = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

		# Loss: MSE between predicted and actual rewards
		loss = self.loss_fn(predicted_q, rewards)

		# Update network with gradient clipping
		self.optimizer.zero_grad()
		loss.backward()

		# Clip gradients to prevent exploding gradients
		torch.nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=1.0)

		self.optimizer.step()

		# Decay epsilon
		self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

		self.training_steps += 1

		return {
			'loss': loss.item(),
			'buffer_size': len(self.replay_buffer),
			'epsilon': self.epsilon,
			'training_steps': self.training_steps
		}


	def save(self, filepath: str) -> None:
		"""Save agent state to file."""
		data = {
			'network': self.network.state_dict(),
			'optimizer': self.optimizer.state_dict(),
			'batch_size': self.batch_size,
			'buffer_size': self.buffer_size,
			'epsilon': self.epsilon,
			'epsilon_end': self.epsilon_end,
			'epsilon_decay': self.epsilon_decay,
			'replay_buffer': list(self.replay_buffer),  # Save experiences
		}

		os.makedirs(os.path.dirname(filepath), exist_ok=True)
		with open(filepath, 'wb') as f:
			pickle.dump(data, f)


	def load(self, filepath: str) -> None:
		"""Load agent state from file."""
		with open(filepath, 'rb') as f:
			data = pickle.load(f)

		self.network.load_state_dict(data['network'])
		self.optimizer.load_state_dict(data['optimizer'])
		self.epsilon = data['epsilon']
		self.epsilon_end = data['epsilon_end']
		self.epsilon_decay = data['epsilon_decay']
		self.batch_size = data['batch_size']
		self.buffer_size = data.get('buffer_size', 50000)

		# Restore replay buffer
		if 'replay_buffer' in data:
			self.replay_buffer.extend(data['replay_buffer'])


	def reset(self) -> None:
		"""Reset agent state."""
		pass
