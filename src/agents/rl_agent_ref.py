import numpy as np
from .base_agent import BaseAgent
from typing import Any, Dict, Optional
import pickle
import os


class QAgent(BaseAgent):
    """
    Q-Learning agent for Jacks-or-Better poker.

    Uses tabular Q-learning with epsilon-greedy exploration.
    State is represented as a flattened 52-element binary vector.
    """

    def __init__(self,
                 observation_space,
                 action_space,
                 learning_rate: float = 0.1,
                 discount_factor: float = 0.95,
                 epsilon_start: float = 1.0,
                 epsilon_end: float = 0.01,
                 epsilon_decay: float = 0.995,
                 seed: Optional[int] = None):
        super().__init__(observation_space, action_space)

        # Q-learning parameters
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay

        # Initialize Q-table as dictionary for memory efficiency
        # Only stores states we've actually encountered
        # Key: state_index (int), Value: q_values array
        self.q_table = {}

        # Default Q-value for unvisited states
        self.default_q_value = 0.0

        # Exploration
        self.epsilon = epsilon_start
        self.rng = np.random.RandomState(seed)

        # Training stats
        self.episode_count = 0

    def _state_to_index(self, observation: np.ndarray) -> int:
        """
        Convert observation bitmap to state index.

        Args:
            observation: 13x4 binary array

        Returns:
            state_index: Integer index for Q-table
        """
        # Flatten and convert to binary string, then to int
        flat_obs = observation.flatten().astype(int)
        binary_string = ''.join(map(str, flat_obs))
        return int(binary_string, 2)

    def act(self, observation: np.ndarray) -> int:
        """
        Choose action using epsilon-greedy policy.

        Args:
            observation: Current state observation

        Returns:
            action: Integer action to take
        """
        state_idx = self._state_to_index(observation)

        # Epsilon-greedy action selection
        if self.rng.random() < self.epsilon:
            # Explore: random action
            action = self.rng.randint(0, self.action_space.n)
        else:
            # Exploit: best action
            q_values = self.q_table.get(state_idx, np.full(self.action_space.n, self.default_q_value))
            action = np.argmax(q_values)

        return action

    def learn(self, experience: Dict[str, Any]) -> Optional[Dict[str, float]]:
        """
        Update Q-table using Q-learning update rule.

        Args:
            experience: Dict with keys 'state', 'action', 'reward', 'next_state', 'done'

        Returns:
            metrics: Dict with training metrics
        """
        state = experience['state']
        action = experience['action']
        reward = experience['reward']
        next_state = experience['next_state']
        done = experience['done']

        state_idx = self._state_to_index(state)
        next_state_idx = self._state_to_index(next_state)

        # Initialize Q-values for state if not seen before
        if state_idx not in self.q_table:
            self.q_table[state_idx] = np.full(self.action_space.n, self.default_q_value)

        # Get current Q-value
        current_q = self.q_table[state_idx][action]

        # Calculate target Q-value
        if done:
            target_q = reward
        else:
            # Initialize Q-values for next state if not seen before
            if next_state_idx not in self.q_table:
                self.q_table[next_state_idx] = np.full(self.action_space.n, self.default_q_value)
            target_q = reward + self.discount_factor * np.max(self.q_table[next_state_idx])

        # Update Q-value
        self.q_table[state_idx][action] += self.learning_rate * (target_q - current_q)

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
        # Convert Q-table dict to more efficient format for saving
        q_table_list = [(state_idx, q_values.tolist()) for state_idx, q_values in self.q_table.items()]

        data = {
            'q_table': q_table_list,
            'epsilon': self.epsilon,
            'episode_count': self.episode_count,
            'learning_rate': self.learning_rate,
            'discount_factor': self.discount_factor,
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
        self.q_table = {state_idx: np.array(q_values) for state_idx, q_values in data['q_table']}
        self.epsilon = data['epsilon']
        self.episode_count = data['episode_count']

        # Load additional parameters if available (for backward compatibility)
        if 'default_q_value' in data:
            self.default_q_value = data['default_q_value']

    def reset(self) -> None:
        """Reset agent state."""
        pass

    def get_q_values(self, observation: np.ndarray) -> np.ndarray:
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
        return q_values.copy()
