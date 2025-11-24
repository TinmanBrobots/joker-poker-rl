#!/usr/bin/env python3
"""
Quick test script to try different model configurations and hyperparameters.
"""

import sys
import os
import numpy as np

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from environment.poker_env import PokerEnv
from agents.supervised_agent import SupervisedAgent, PokerNet, LinearPokerNet
from agents.random_agent import RandomAgent


def evaluate_agent_quick(agent, env, num_games=1000):
    """Quick evaluation for testing."""
    wins = 0
    total_reward = 0

    for _ in range(num_games):
        obs, _ = env.reset()
        action = agent.act(obs)
        _, reward, _, _, _ = env.step(action)
        total_reward += reward
        if reward > 0:
            wins += 1

    return total_reward / num_games, wins / num_games


def test_configuration(name, network, agent_kwargs, train_episodes=5000):
    """Test a specific configuration."""
    print(f"\n{'='*50}")
    print(f"Testing: {name}")
    print(f"{'='*50}")

    env = PokerEnv()

    agent = SupervisedAgent(
        observation_space=env.observation_space,
        action_space=env.action_space,
        network=network,
        **agent_kwargs
    )

    # Pre-fill buffer
    print("Pre-filling buffer...")
    while len(agent.replay_buffer) < agent.batch_size:
        obs, _ = env.reset()
        action = agent.rng.randint(0, env.action_space.n)
        _, reward, _, _, _ = env.step(action)
        agent.replay_buffer.append({'state': obs, 'action': action, 'reward': reward})

    # Quick training
    print(f"Training for {train_episodes} episodes...")
    for episode in range(train_episodes):
        obs, _ = env.reset()
        action = agent.act(obs)
        _, reward, _, _, _ = env.step(action)

        experience = {'state': obs, 'action': action, 'reward': reward}
        agent.learn(experience)

        if (episode + 1) % 1000 == 0:
            print(f"Episode {episode+1}: epsilon={agent.epsilon:.3f}")

    # Evaluate
    random_agent = RandomAgent(env.observation_space, env.action_space, seed=42)
    random_reward, random_win_rate = evaluate_agent_quick(random_agent, env, 5000)

    agent_reward, agent_win_rate = evaluate_agent_quick(agent, env, 5000)

    print("Results:")
    print(".3f")
    print(".1%")
    print(".3f")
    print(".1%")
    print(".1%")

    return agent_reward, agent_win_rate


def main():
    """Test different configurations."""
    np.random.seed(42)

    # Test different architectures
    configs = [
        ("Linear", LinearPokerNet(), {
            'learning_rate': 0.01,
            'batch_size': 128,
            'buffer_size': 50000,
            'epsilon_decay': 0.999
        }),

        ("Simple NN", PokerNet(hidden_dims=[64]), {
            'learning_rate': 0.01,
            'batch_size': 128,
            'buffer_size': 50000,
            'epsilon_decay': 0.999
        }),

        ("Medium NN", PokerNet(hidden_dims=[128, 64]), {
            'learning_rate': 0.005,
            'batch_size': 256,
            'buffer_size': 100000,
            'epsilon_decay': 0.9995
        }),

        ("Large Batch", PokerNet(hidden_dims=[128, 64]), {
            'learning_rate': 0.01,
            'batch_size': 512,
            'buffer_size': 200000,
            'epsilon_decay': 0.9999
        }),
    ]

    results = []
    for name, network, kwargs in configs:
        reward, win_rate = test_configuration(name, network, kwargs, train_episodes=10000)
        results.append((name, reward, win_rate))

    print("\n" + "="*60)
    print("SUMMARY - Best Configurations:")
    print("="*60)
    results.sort(key=lambda x: x[2], reverse=True)  # Sort by win rate
    for name, reward, win_rate in results:
        print("25s")


if __name__ == "__main__":
    main()
