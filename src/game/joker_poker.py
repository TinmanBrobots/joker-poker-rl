from .deck import Deck
from .poker_hand import PokerHand

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
		self.deal()

	def get_hand(self):
		return self.hand

	def deal(self):
		for _ in range(5):
			self.hand.append(self.deck.draw())
		self.hand.sort()

	def redraw(self, action: int):
		assert 0 <= action < 32, "Invalid action"
		for i in range(5):
			if action & (1 << i):
				self.hand[i] = self.deck.draw()
			action >>= 1
		self.hand.sort()

	def score(self):
		return score_map[PokerHand(self.hand).evaluate()]

	def reset(self):
		self.deck.reset()
		self.hand.clear()
		self.deal()