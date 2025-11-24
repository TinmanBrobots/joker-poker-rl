import numpy as np
from .base_agent import BaseAgent
from typing import Any, Dict, Optional


class RandomAgent(BaseAgent):
    """
    Random agent that selects actions uniformly at random.
    Useful for testing environments and establishing baseline performance.
    """

    def __init__(self, observation_space, action_space, seed: Optional[int] = None):
        super().__init__(observation_space, action_space)
        self.rng = np.random.RandomState(seed)

    def act(self, observation: np.ndarray) -> int:
        """
        Choose a random action uniformly from the action space.

        Args:
            observation: Current state observation (ignored by random agent)

        Returns:
            action: Random integer action
        """
        return self.rng.randint(0, self.action_space.n)

    def learn(self, experience: Dict[str, Any]) -> Optional[Dict[str, float]]:
        """
        Random agent doesn't learn from experience.

        Args:
            experience: Experience data (ignored)

        Returns:
            None (no training metrics)
        """
        return None

    def reset(self) -> None:
        """Reset agent state - no state to reset for random agent."""
        pass
