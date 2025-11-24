from .deck import Card
from collections import Counter
from typing import Literal

HandType = Literal['high_card', 'low_pair', 'high_pair', 'two_pair', 'three_of_a_kind', 'straight', 'flush', 'full_house', 'four_of_a_kind', 'straight_flush']

class PokerHand:
	def __init__(self, hand: list[Card]):
		self.hand = hand

	def evaluate(self) -> HandType:
		hand_values = [card.rank for card in self.hand]
		hand_suits = [card.suit for card in self.hand]
		hand_values.sort()
		hand_suits.sort()

		is_straight = all(hand_values[i] + 1 == hand_values[i + 1] for i in range(len(hand_values) - 1))
		is_straight = is_straight or hand_values == [2, 3, 4, 5, 14]
		is_flush = all(hand_suits[i] == hand_suits[i + 1] for i in range(len(hand_suits) - 1))

		# get non-zero rank counts
		rank_counts = Counter(hand_values)
		rank_counts = { k: v for k, v in rank_counts.items() if v > 0 }
		sorted_rank_counts = sorted(rank_counts.values(), reverse=True)

		# check for straight flush
		if is_straight and is_flush:
			return 'straight_flush'

		# check for four of a kind
		if sorted_rank_counts == [4, 1]:
			return 'four_of_a_kind'

		# check for full house
		if sorted_rank_counts == [3, 2]:
			return 'full_house'

		# check for flush
		if is_flush:
			return 'flush'

		# check for straight
		if is_straight:
			return 'straight'

		# check for three of a kind
		if sorted_rank_counts == [3, 1, 1]:
			return 'three_of_a_kind'

		# check for two pair
		if sorted_rank_counts == [2, 2, 1]:
			return 'two_pair'

		# check for high/low pair
		if sorted_rank_counts == [2, 1, 1, 1]:
			pair_rank = next(rank for rank, count in rank_counts.items() if count == 2)
			return 'high_pair' if pair_rank >= 11 else 'low_pair'

		return 'high_card'