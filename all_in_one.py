# Generated separately with Claude

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import Counter
from itertools import combinations
from tqdm import tqdm

# Card representation: 0-51 (13 ranks × 4 suits)
RANKS = ['2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A']
SUITS = ['♠', '♥', '♦', '♣']

def card_to_idx(rank, suit):
    return RANKS.index(rank) * 4 + SUITS.index(suit)

def idx_to_card(idx):
    return RANKS[idx // 4], SUITS[idx % 4]

# Jacks or Better payout table
PAYOUTS = {
    'royal_flush': 250,
    'straight_flush': 50,
    'four_of_a_kind': 25,
    'full_house': 9,
    'flush': 6,
    'straight': 4,
    'three_of_a_kind': 3,
    'two_pair': 2,
    'jacks_or_better': 1,
    'nothing': 0
}

def evaluate_hand(cards):
    """Evaluate 5 cards and return hand rank and payout"""
    ranks = [RANKS.index(c[0]) for c in cards]
    suits = [c[1] for c in cards]
    rank_counts = Counter(ranks)
    
    is_flush = len(set(suits)) == 1
    sorted_ranks = sorted(ranks)
    is_straight = (sorted_ranks == list(range(sorted_ranks[0], sorted_ranks[0] + 5)) or
                   sorted_ranks == [0, 1, 2, 3, 12])  # A-2-3-4-5
    
    counts = sorted(rank_counts.values(), reverse=True)
    
    # Check hands from best to worst
    if is_straight and is_flush:
        if sorted_ranks == [8, 9, 10, 11, 12]:  # T-J-Q-K-A
            return 'royal_flush', PAYOUTS['royal_flush']
        return 'straight_flush', PAYOUTS['straight_flush']
    
    if counts == [4, 1]:
        return 'four_of_a_kind', PAYOUTS['four_of_a_kind']
    
    if counts == [3, 2]:
        return 'full_house', PAYOUTS['full_house']
    
    if is_flush:
        return 'flush', PAYOUTS['flush']
    
    if is_straight:
        return 'straight', PAYOUTS['straight']
    
    if counts == [3, 1, 1]:
        return 'three_of_a_kind', PAYOUTS['three_of_a_kind']
    
    if counts == [2, 2, 1]:
        return 'two_pair', PAYOUTS['two_pair']
    
    if counts == [2, 1, 1, 1]:
        # Check if pair is Jacks or better
        pair_rank = [r for r, c in rank_counts.items() if c == 2][0]
        if pair_rank >= 9:  # J, Q, K, A
            return 'jacks_or_better', PAYOUTS['jacks_or_better']
    
    return 'nothing', PAYOUTS['nothing']


def compute_action_values(hand_indices, num_samples=100):
    """
    For a given 5-card hand, compute expected value of each of 32 possible actions
    by sampling possible draws from the remaining deck.
    
    Returns: array of 32 expected values
    """
    deck = list(range(52))
    remaining_deck = [c for c in deck if c not in hand_indices]
    
    action_values = np.zeros(32)
    
    for action in range(32):
        # Determine which cards to hold
        hold_mask = [(action >> i) & 1 for i in range(5)]
        held_cards = [hand_indices[i] for i in range(5) if hold_mask[i]]
        num_to_draw = 5 - len(held_cards)
        
        if num_to_draw == 0:
            # Holding all cards - deterministic outcome
            cards = [idx_to_card(idx) for idx in hand_indices]
            _, payout = evaluate_hand(cards)
            action_values[action] = payout
        else:
            # Sample possible draws
            total_payout = 0
            for _ in range(num_samples):
                drawn_cards = random.sample(remaining_deck, num_to_draw)
                final_hand = held_cards + drawn_cards
                cards = [idx_to_card(idx) for idx in final_hand]
                _, payout = evaluate_hand(cards)
                total_payout += payout
            action_values[action] = total_payout / num_samples
    
    return action_values


def generate_training_data(num_hands=10_000, num_samples=100):
    """
    Generate training data by dealing random hands and computing
    expected value for each action via sampling.
    
    Returns: (states, action_values) where action_values is (num_hands, 32)
    """
    print(f"Generating {num_hands} training hands...")
    
    states = []
    all_action_values = []
    
    deck = list(range(52))
    
    for i in tqdm(range(num_hands)):
        # if (i + 1) % 1000 == 0:
        #     print(f"  Generated {i + 1}/{num_hands} hands...")
        
        # Deal random hand
        random.shuffle(deck)
        hand = deck[:5]
        
        # Create state representation
        state = np.zeros(52, dtype=np.float32)
        for card in hand:
            state[card] = 1
        
        # Compute expected values for all actions
        action_values = compute_action_values(hand, num_samples)
        
        states.append(state)
        all_action_values.append(action_values)
    
    return np.array(states), np.array(all_action_values)


class VideoPokerNet(nn.Module):
    """Neural network to predict expected value of each action"""
    def __init__(self, state_size=52, action_size=32, hidden_size=256):
        super(VideoPokerNet, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_size)
        )
    
    def forward(self, x):
        return self.network(x)


def train_network(states, action_values, num_epochs=100, batch_size=64, learning_rate=1e-3):
    """Train neural network via supervised learning"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nTraining network on {device}...")
    
    model = VideoPokerNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    
    # Convert to tensors
    states_tensor = torch.FloatTensor(states).to(device)
    values_tensor = torch.FloatTensor(action_values).to(device)
    
    num_samples = len(states)
    
    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1:3d}/{num_epochs}")
        model.train()
        total_loss = 0
        num_batches = 0
        
        # Shuffle data
        indices = torch.randperm(num_samples)
        
        for i in tqdm(range(0, num_samples, batch_size)):
            batch_indices = indices[i:i + batch_size]
            batch_states = states_tensor[batch_indices]
            batch_values = values_tensor[batch_indices]
            
            # Forward pass
            predicted_values = model(batch_states)
            loss = criterion(predicted_values, batch_values)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        avg_loss = total_loss / num_batches
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1:3d}/{num_epochs} | Loss: {avg_loss:.6f}")
    
    return model


class VideoPokerAgent:
    """Agent that uses trained network to play Video Poker"""
    def __init__(self, model):
        self.model = model
        self.device = next(model.parameters()).device
        self.model.eval()
    
    def select_action(self, state):
        """Select action with highest expected value"""
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            action_values = self.model(state_tensor)
            return action_values.argmax().item()
    
    def get_action_values(self, state):
        """Get all action values for a state"""
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            return self.model(state_tensor).cpu().numpy()[0]


def evaluate_agent(agent, num_hands=10000):
    """Evaluate trained agent by playing actual hands"""
    print(f"\nEvaluating agent on {num_hands} hands...")
    
    deck = list(range(52))
    results = Counter()
    total_payout = 0
    
    for _ in tqdm(range(num_hands)):
        # Deal hand
        random.shuffle(deck)
        hand = deck[:5]
        
        # Create state
        state = np.zeros(52, dtype=np.float32)
        for card in hand:
            state[card] = 1
        
        # Agent selects action
        action = agent.select_action(state)
        
        # Execute action
        hold_mask = [(action >> i) & 1 for i in range(5)]
        held_cards = [hand[i] for i in range(5) if hold_mask[i]]
        num_to_draw = 5 - len(held_cards)
        
        if num_to_draw == 0:
            final_hand = hand
        else:
            remaining = [c for c in deck[5:] if c not in held_cards]
            drawn = remaining[:num_to_draw]
            final_hand = held_cards + drawn
        
        # Evaluate final hand
        cards = [idx_to_card(idx) for idx in final_hand]
        hand_name, payout = evaluate_hand(cards)
        
        results[hand_name] += 1
        total_payout += payout
    
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"Hands played: {num_hands}")
    print(f"Total payout: {total_payout}")
    print(f"Average payout: {total_payout / num_hands:.4f}")
    print(f"Return: {100 * total_payout / num_hands:.2f}%")
    print("\nHand distribution:")
    for hand in sorted(results.keys(), key=lambda x: PAYOUTS.get(x, 0), reverse=True):
        count = results[hand]
        pct = 100 * count / num_hands
        print(f"  {hand:20s}: {count:6d} ({pct:5.2f}%)")
    
    return total_payout / num_hands


if __name__ == "__main__":
    print("=" * 60)
    print("VIDEO POKER SUPERVISED LEARNING")
    print("=" * 60)
    
    # Generate training data
    states, action_values = generate_training_data(num_hands=10_000, num_samples=100)
    
    # Train network
    model = train_network(states, action_values, num_epochs=1_000, batch_size=64)
    
    # Create agent
    agent = VideoPokerAgent(model)
    
    # Evaluate
    avg_return = evaluate_agent(agent, num_hands=10000)
    
    # Save model
    torch.save(model.state_dict(), 'video_poker_supervised.pth')
    print("\nModel saved to video_poker_supervised.pth")