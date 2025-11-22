from game.deck import Deck
from game.poker_hand import PokerHand

score_map = {
	'high_card': 0,
	'low_pair': 0, # below pair of Jacks
	'high_pair': 1, # pair of Jacks or better
	'two_pair': 2,
	'three_of_a_kind': 3,
	'straight': 4,
	'flush': 6,
	'full_house': 9,
	'four_of_a_kind': 25,
	'straight_flush': 800,
}

class JokerPoker:
	def __init__(self):
		self.deck = Deck()
		self.hand = []

	def deal(self):
		for _ in range(5):
			self.hand.append(self.deck.draw())

	def redraw(self, action: int):
		for i in range(5):
			if action & (1 << i):
				self.hand[i] = self.deck.draw()
			action >>= 1

	def score(self):
		return score_map[PokerHand(self.hand).evaluate()] - 1

	def reset(self):
		self.hand = []
		self.deck.reset()