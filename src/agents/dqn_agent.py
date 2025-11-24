import numpy as np
from .base_agent import BaseAgent
from typing import Any, Dict, Optional
import pickle
import os
import torch
import torch.nn as nn


class DQNAgent(BaseAgent):
	"""
	Deep Q-Network agent for Jacks-or-Better poker.

	Uses Deep Q-Network with epsilon-greedy exploration.
	State is represented as a flattened 52-element binary vector.
	"""

	def __init__(
		self,
		observation_space,
		action_space,
		learning_rate: float = 0.1,
		discount_factor: float = 0.95,
		epsilon_start: float = 1.0,
		epsilon_end: float = 0.01,
		epsilon_decay: float = 0.995,
		buffer_size: int = 10_000,
		batch_size: int = 32,
		seed: Optional[int] = None,
	):
		super().__init__(observation_space, action_space)

		# DQN parameters
		self.learning_rate = learning_rate
		self.discount_factor = discount_factor
		self.epsilon_start = epsilon_start
		self.epsilon_end = epsilon_end
		self.epsilon_decay = epsilon_decay
		self.buffer_size = buffer_size
		self.batch_size = batch_size

		input_dim = observation_space.shape[0] * observation_space.shape[1]
		hidden_dim = 128
		output_dim = action_space.n

		# Q-Network
		self.network = nn.Sequential(
			nn.Flatten(),
			nn.Linear(input_dim, hidden_dim),
			nn.ReLU(),
			nn.Linear(hidden_dim, hidden_dim),
			nn.ReLU(),
			nn.Linear(hidden_dim, output_dim),
		)

		# Target Q-Network
		self.target_network = nn.Sequential(
			nn.Flatten(),
			nn.Linear(input_dim, hidden_dim),
			nn.ReLU(),
			nn.Linear(hidden_dim, hidden_dim),
			nn.ReLU(),
			nn.Linear(hidden_dim, output_dim),
		)
		self.target_network.load_state_dict(self.network.state_dict())

		# Optimizer and loss function
		self.optimizer = torch.optim.Adam(self.network.parameters(), lr=self.learning_rate)
		self.loss_fn = nn.MSELoss()

		# Replay buffer
		self.replay_buffer = deque(maxlen=self.buffer_size)

		# Exploration
		self.epsilon = epsilon_start
		self.rng = np.random.RandomState(seed)

		# Training stats
		self.episode_count = 0

	
	def act(self, observation: np.ndarray) -> int:
		"""
		Choose action using epsilon-greedy policy.

		Args:
			observation: Current state observation

		Returns:
			action: Integer action to take
		"""
		# Epsilon-greedy action selection
		if self.rng.random() < self.epsilon:
			# Explore: random action
			return self.rng.randint(0, self.action_space.n)
		else:
			# Exploit: DQN action selection
			state_tensor = torch.from_numpy(observation).float().unsqueeze(0)
			with torch.no_grad():
				q_values = self.network(state_tensor)
				action = torch.argmax(q_values).item()
			return action

	def add_experience(self, state, action, reward, next_state, done):
		"""Store experience in replay buffer."""
		self.replay_buffer.append((state, action, reward, next_state, done))

	def sample_batch(self):
		"""Sample a batch from replay buffer."""
		batch = random.sample(self.replay_buffer, min(len(self.replay_buffer), self.batch_size))
		return tuple(map(np.array, zip(*batch)))

	def learn(self, experience: Dict[str, Any]) -> Optional[Dict[str, float]]:
		"""
		Update Q-network using experience replay.

		Args:
			batch: Tuple of tensors (states, actions, rewards, next_states, dones)

		Returns:
			loss: Scalar loss value
		"""
		self.add_experience(
			experience['state'],
			experience['action'],
			experience['reward'],
			experience['next_state'],
			experience['done']
		)

		if len(self.replay_buffer) < self.batch_size:
			return None

		# Sample batch
		states, actions, rewards, next_states, dones = self.sample_batch()

		# Convert to tensors
		states = torch.from_numpy(states).float()
		actions = torch.from_numpy(actions).long()
		rewards = torch.from_numpy(rewards).float()
		next_states = torch.from_numpy(next_states).float()
		dones = torch.from_numpy(dones).float()

		# Get Q-values
		current_qs = self.network(states).gather(1, actions.unsqueeze(1)).squeeze(1)
		next_qs = self.target_network(next_states).max(dim=1)[0]

		# 
		with torch.no_grad():
			next_qs = self.target_network(next_states).max(dim=1)[0]

		# Calculate target Q-values
		if done:
			target_q = reward
		else:
			next_q = self.network(torch.from_numpy(next_state).float())
			target_q = reward + self.discount_factor * np.max(self.q_table[next_state_idx])

		# Update Q-value
		current_q[action] += self.learning_rate * (target_q - current_q[action])

		# Decay epsilon
		self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

		# Update episode count
		if done:
			self.episode_count += 1

		return {
			'epsilon': self.epsilon,
			'episode': self.episode_count,
			'q_value': current_q,
			'states_learned': len(self.q_table)
		}


	def save(self, filepath: str) -> None:
		"""Save agent state to file."""
		# Convert Q-table dict to a more efficient format for saving
		q_table_list = [(state_idx, q_values.tolist()) for state_idx, q_values in self.q_table.items()]

		data = {
			'q_table': q_table_list,
			'epsilon': self.epsilon,
			'episode_count': self.episode_count,
			'learning_rate': self.learning_rate,
			'discount_factor': self.discount_factor,
			'epsilon_start': self.epsilon_start,
			'epsilon_end': self.epsilon_end,
			'epsilon_decay': self.epsilon_decay,
			'default_q_value': self.default_q_value
		}

		os.makedirs(os.path.dirname(filepath), exist_ok=True)
		with open(filepath, 'wb') as f:
			pickle.dump(data, f)


	def load(self, filepath: str) -> None:
		"""Load agent state from file."""
		with open(filepath, 'rb') as f:
			data = pickle.load(f)

		# Convert Q-table list back to dict
		self.q_table = {state_idx: np.array(q_values) for state_idx, q_values in data['q_table_list']}
		self.epsilon = data['epsilon']
		self.episode_count = data['episode_count']

		# Load additional parameters if available (for backward compatibility)
		if 'default_q_value' in data:
			self.default_q_value = data['default_q_value']


	def reset(self) -> None:
		"""Reset agent state."""
		pass


	def get_q_values(self, observation: np.ndarray, copy: bool = False) -> np.ndarray:
		"""
		Get Q-values for all actions in current state.
		Useful for analysis and debugging.

		Args:
			observation: Current state observation

		Returns:
			q_values: Array of Q-values for each action
		"""
		state_idx = self._state_to_index(observation)
		q_values = self.q_table.get(state_idx, np.full(self.action_space.n, self.default_q_value))
		return q_values.copy() if copy else q_values