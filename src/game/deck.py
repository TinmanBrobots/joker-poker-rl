from collections import namedtuple
import random

Card = namedtuple('Card', ['rank', 'suit'])

ranks = {
	2: '2',
	3: '3',
	4: '4',
	5: '5',
	6: '6',
	7: '7',
	8: '8',
	9: '9',
	10: '10',
	11: 'J',
	12: 'Q',
	13: 'K',
	14: 'A',
}

suits = {
	0: 'clubs',
	1: 'diamonds',
	2: 'hearts',
	3: 'spades',
}

class Deck:
	def __init__(self):
		self.cards = []
		self.reset()

	def shuffle(self):
		random.shuffle(self.cards)

	def draw(self):
		if len(self.cards) == 0:
			return None
		return self.cards.pop()

	def reset(self):
		self.cards = [Card(rank, suit) for rank in ranks.keys() for suit in suits.keys()]
		self.shuffle()