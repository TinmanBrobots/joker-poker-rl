#!/usr/bin/env python3
"""
Interactive script to play Jacks-or-Better poker with trained agents.

Allows you to watch agents play or compare different agents.
"""

import sys
import os
import numpy as np

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from environment.poker_env import PokerEnv
from agents.random_agent import RandomAgent
from agents.q_agent import QAgent


def decode_action(action: int) -> str:
    """
    Decode action integer into human-readable hold/discard pattern.

    Args:
        action: Integer action (0-31)

    Returns:
        description: String describing which cards to hold/discard
    """
    holds = []
    for i in range(5):
        if action & (1 << i):
            holds.append(f"Card {i+1}")
        else:
            holds.append(f"~Card {i+1}~")

    return " | ".join(holds)


def print_hand(cards, title="Hand"):
    """Print poker hand in readable format."""
    print(f"\n{title}:")
    for i, card in enumerate(cards):
        rank_names = {11: 'J', 12: 'Q', 13: 'K', 14: 'A'}
        suit_names = {0: '♣', 1: '♦', 2: '♥', 3: '♠'}

        rank = rank_names.get(card.rank, str(card.rank))
        suit = suit_names[card.suit]
        print(f"  Card {i+1}: {rank}{suit}")


def play_episode(agent, env, verbose=True):
    """
    Play one episode with an agent.

    Args:
        agent: Agent to play with
        env: Poker environment
        verbose: Whether to print game details

    Returns:
        reward: Final reward for the episode
    """
    # Reset environment
    obs, _ = env.reset()

    if verbose:
        print("\n" + "="*50)
        print("NEW HAND")
        print("="*50)
        print_hand(env.game.hand, "Initial Hand")

    # Agent makes decision
    action = agent.act(obs)

    if verbose:
        print(f"\nAgent Action: {action}")
        print(f"Decision: {decode_action(action)}")

    # Execute action
    next_obs, reward, terminated, truncated, info = env.step(action)

    if verbose:
        print_hand(env.game.hand, "Final Hand")
        print(f"\nHand Type: {env.game.hand_type}")
        print(f"Reward: {reward}")

    return reward


def compare_agents(num_games=10):
    """Compare random vs RL agent performance."""
    env = PokerEnv()

    # Create agents
    random_agent = RandomAgent(env.observation_space, env.action_space, seed=42)

    # Try to load trained RL agent
    q_agent = QAgent(env.observation_space, env.action_space, seed=42)
    try:
        q_agent.load('models/q_agent.pkl')
        print("Loaded trained RL agent from models/q_agent.pkl")
    except FileNotFoundError:
        print("No trained RL agent found. Using untrained agent.")

    # Compare performance
    random_rewards = []
    rl_rewards = []

    print("Comparing agents over", num_games, "games...")
    print("-" * 50)

    for game in range(num_games):
        # Random agent
        random_reward = play_episode(random_agent, env, verbose=False)
        random_rewards.append(random_reward)

        # RL agent
        rl_reward = play_episode(q_agent, env, verbose=False)
        rl_rewards.append(rl_reward)

        print("2d"
              "4.1f"
              "4.1f")

    # Summary
    print("\n" + "="*50)
    print("SUMMARY")
    print("="*50)
    print(".2f")
    print(".2f")
    print(f"RL Win Rate: {np.mean([r > 0 for r in rl_rewards]):.1%}")
    print(f"Random Win Rate: {np.mean([r > 0 for r in random_rewards]):.1%}")


def interactive_play():
    """Let user watch agents play interactively."""
    env = PokerEnv()

    # Create agents
    random_agent = RandomAgent(env.observation_space, env.action_space, seed=42)

    q_agent = QAgent(env.observation_space, env.action_space, seed=42)
    try:
        q_agent.load('models/q_agent.pkl')
        print("Loaded trained RL agent!")
    except FileNotFoundError:
        print("No trained RL agent found. Using untrained agent.")

    print("\nChoose agent to watch:")
    print("1. Random Agent")
    print("2. RL Agent")

    choice = input("Enter choice (1 or 2): ").strip()

    if choice == "1":
        agent = random_agent
        agent_name = "Random Agent"
    elif choice == "2":
        agent = q_agent
        agent_name = "RL Agent"
    else:
        print("Invalid choice. Using Random Agent.")
        agent = random_agent
        agent_name = "Random Agent"

    print(f"\nPlaying with {agent_name}")
    print("Press Enter to play next hand, or 'q' to quit.")

    while True:
        user_input = input("\nPress Enter to continue (q to quit): ").strip().lower()
        if user_input == 'q':
            break

        reward = play_episode(agent, env, verbose=True)


def main():
    """Main function."""
    print("Jacks-or-Better Poker Agent Demo")
    print("=" * 40)

    print("\nChoose mode:")
    print("1. Interactive play (watch agent decisions)")
    print("2. Compare agents (Random vs RL)")

    choice = input("Enter choice (1 or 2): ").strip()

    if choice == "1":
        interactive_play()
    elif choice == "2":
        compare_agents()
    else:
        print("Invalid choice. Running interactive play.")
        interactive_play()


if __name__ == "__main__":
    main()
