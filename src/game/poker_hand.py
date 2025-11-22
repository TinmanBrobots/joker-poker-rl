from collections import Counter


class PokerHand:
	def __init__(self, hand: list[Card]):
		self.hand = hand

	def evaluate(self):
		hand_values = [card.rank for card in self.hand]
		hand_suits = [card.suit for card in self.hand]
		hand_values.sort()
		hand_suits.sort()

		is_straight = all(hand_values[i] + 1 == hand_values[i + 1] for i in range(len(hand_values) - 1))
		is_staight = is_straight or hand_values == [2, 3, 4, 5, 14]
		is_flush = all(hand_suits[i] == hand_suits[i + 1] for i in range(len(hand_suits) - 1))

		# get non-zero rank counts
		rank_counts = Counter(hand_values)
		rank_counts = { k: v for k, v in rank_counts.items() if v > 0 }
		sorted_rank_counts = sorted(rank_counts.values(), reverse=True)

		# check for straight flush
		if is_straight and is_flush:
			return hand_ranks['straight_flush']

		# check for four of a kind
		if sorted_rank_counts == [4, 1]:
			return hand_ranks['four_of_a_kind']

		# check for full house
		if sorted_rank_counts == [3, 2]:
			return hand_ranks['full_house']

		# check for flush
		if is_flush:
			return hand_ranks['flush']

		# check for straight
		if is_straight:
			return hand_ranks['straight']

		# check for three of a kind
		if sorted_rank_counts == [3, 1, 1]:
			return hand_ranks['three_of_a_kind']

		# check for two pair
		if sorted_rank_counts == [2, 2, 1]:
			return hand_ranks['two_pair']

		# check for high pair
		if sorted_rank_counts == [2, 1, 1, 1] and any(rank_counts[k] == 2 for k in [11, 12, 13, 14]):
			return hand_ranks['high_pair']

		# check for low pair
		if sorted_rank_counts == [2, 1, 1, 1]:
			return hand_ranks['low_pair']

		return hand_ranks['high_card']