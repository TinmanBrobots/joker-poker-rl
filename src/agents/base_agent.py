from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
import numpy as np


class BaseAgent(ABC):
    """Abstract base class for all RL agents."""

    def __init__(self, observation_space, action_space):
        self.observation_space = observation_space
        self.action_space = action_space

    @abstractmethod
    def act(self, observation: np.ndarray) -> int:
        """
        Choose an action based on the current observation.

        Args:
            observation: Current state observation

        Returns:
            action: Integer action to take
        """
        pass

    @abstractmethod
    def learn(self, experience: Dict[str, Any]) -> Optional[Dict[str, float]]:
        """
        Update agent based on experience.

        Args:
            experience: Dictionary containing experience data
                      (state, action, reward, next_state, done)

        Returns:
            metrics: Optional dictionary of training metrics
        """
        pass

    def save(self, filepath: str) -> None:
        """Save agent state to file."""
        pass

    def load(self, filepath: str) -> None:
        """Load agent state from file."""
        pass

    def reset(self) -> None:
        """Reset agent state (e.g., for new episode)."""
        pass
