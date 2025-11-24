import numpy as np
from game.joker_poker import JokerPoker

import gymnasium as gym
from gymnasium import spaces


class PokerEnv(gym.Env):
	def __init__(self):
		super().__init__()
		self.game = JokerPoker()

		# Define action space: 32 discrete actions (2^5 hold/discard combinations)
		self.action_space = spaces.Discrete(32)

		# Define observation space: 13x4 bitmap (ranks 2-Ace x 4 suits)
		self.observation_space = spaces.Box(
			low=0, high=1,
			shape=(13, 4),
			dtype=np.float32
		)


	def reset(self):
		self.game.reset()
		return self.get_observation(), {}


	def get_observation(self):
		bitmap = np.zeros((13, 4), dtype=np.float32)
		for card in self.game.get_hand():
			bitmap[card.rank - 2, card.suit] = 1
		return bitmap


	def get_reward(self):
		return self.game.score()


	def step(self, action: int):
		self.game.redraw(action)
		return self.get_observation(), self.get_reward(), True, False, {}


	def close(self):
		pass


	def render(self, mode='human'):
		pass
