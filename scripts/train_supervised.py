#!/usr/bin/env python3
"""
Training script for Jacks-or-Better RL agent.

This script demonstrates how to train an RL agent to play Jacks-or-Better poker
using the environment and agents we've built.
"""

import sys
import os
import math
import numpy as np
from tqdm import tqdm

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from environment.poker_env import PokerEnv
from agents.random_agent import RandomAgent
from agents.supervised_agent import SupervisedAgent, PokerNet


def evaluate_agent(agent, env, num_episodes=1000):
	"""
	Evaluate agent performance over multiple episodes.

	Args:
		agent: Agent to evaluate
		env: Environment to test in
		num_episodes: Number of episodes to evaluate

	Returns:
		avg_reward: Average reward per episode
		win_rate: Percentage of episodes with positive reward
	"""
	total_reward = 0
	wins = 0

	for _ in range(num_episodes):
		obs, _ = env.reset()
		action = agent.act(obs)
		_,reward, _, _, _ = env.step(action)

		total_reward += reward
		if reward > 0:
			wins += 1

	avg_reward = total_reward / num_episodes
	win_rate = wins / num_episodes

	return avg_reward, win_rate


def train_supervised_agent(num_episodes=200_000, eval_every=10_000):
	"""
	Train a supervised learning agent on Jacks-or-Better.

	Args:
		num_episodes: Total episodes to train for
		eval_every: Evaluate agent every N episodes
	"""
	# Create environment and agent
	env = PokerEnv()

	# Try different architectures
	# network = LinearPokerNet()                  # Simplest: 52->32 linear
	# network = PokerNet(hidden_dims=[64])        # Simple: 52->64->32
	network = PokerNet(hidden_dims=[128, 64])    # Medium: 52->128->64->32
	# network = PokerNet(hidden_dims=[256, 128]) # Complex: 52->256->128->32

	agent = SupervisedAgent(
		observation_space=env.observation_space,
		action_space=env.action_space,
		network=network,
		learning_rate=0.01,     # Increased for faster learning
		batch_size=256,         # Larger batches for stability
		buffer_size=200000,     # Much larger buffer
		epsilon_start=1.0,
		epsilon_end=0.05,       # Keep some exploration
		epsilon_decay=0.99995,  # Very slow decay
		weight_decay=1e-4,      # L2 regularization
	)

	# Create random agent for baseline comparison
	random_agent = RandomAgent(env.observation_space, env.action_space, seed=42)

	print("Starting training...")
	print(f"Training for {num_episodes} episodes")
	print("-" * 50)

	# Evaluate baseline random agent
	random_reward, random_win_rate = evaluate_agent(random_agent, env, 50_000)
	print(f"Random Agent:")
	print(f"  - Avg Reward: {random_reward:.3f}")
	print(f"  - Win Rate: {random_win_rate:.1%}")

	# Training loop
	rewards_history = []
	win_rates_history = []

	# Pre-fill buffer with random experiences for faster initial training
	print("Pre-filling replay buffer with random experiences...")
	while len(agent.replay_buffer) < agent.batch_size:
		obs, _ = env.reset()
		action = agent.rng.randint(0, env.action_space.n)  # Pure random for pre-fill
		_, reward, _, _, _ = env.step(action)
		# Store as tuple (state, action, reward) like the learn method does
		agent.replay_buffer.append((obs, action, reward))
	print(f"Buffer pre-filled with {len(agent.replay_buffer)} experiences")

	# Training loop - optimized for speed
	loss_history = []

	for episode in tqdm(range(num_episodes), desc="Training"):
		# Reset environment
		obs, _ = env.reset()

		# Agent takes action (with epsilon-greedy exploration)
		action = agent.act(obs)

		# Environment responds
		_, reward, _, _, _ = env.step(action)

		# Agent learns from experience
		experience = {'state': obs, 'action': action, 'reward': reward}
		metrics = agent.learn(experience)

		# Track metrics
		rewards_history.append(reward)
		if metrics and 'loss' in metrics:
			loss_history.append(metrics['loss'])

		# Less frequent evaluation for speed
		if (episode + 1) % eval_every == 0:
			avg_reward, win_rate = evaluate_agent(agent, env, 5000)  # Reduced eval size
			win_rates_history.append(win_rate)

			avg_loss = np.mean(loss_history[-1000:]) if loss_history else 0  # Recent loss

			tqdm.write(f"Episode {episode+1:6d} | "
					  f"Avg Reward: {avg_reward:.3f} | "
					  f"Win Rate: {win_rate:.1%} | "
					  f"Recent Loss: {avg_loss:.4f} | "
					  f"Buffer: {len(agent.replay_buffer)} | "
					  f"Epsilon: {agent.epsilon:.3f}")

	print("\nTraining completed!")
	print("-" * 50)

	# Final evaluation
	final_reward, final_win_rate = evaluate_agent(agent, env, 25_000)
	print(f"Final Evaluation")
	print(f"  - Avg Reward: {final_reward:.3f}")
	print(f"  - Win Rate: {final_win_rate:.1%}")
	print(f"Improvement over random: {final_win_rate - random_win_rate:.1%}")

	# Save trained agent
	agent.save('models/supervised_agent.pkl')
	print("Model saved to models/supervised_agent.pkl")

	return agent, rewards_history, win_rates_history


def main():
	"""Main training function."""
	# Set random seeds for reproducibility
	np.random.seed(42)

	# Create models directory
	os.makedirs('models', exist_ok=True)

	# Train agent
	agent, rewards, win_rates = train_supervised_agent(num_episodes=200_000, eval_every=10_000)

	print("\n" + "="*60)
	print("TRAINING SUMMARY")
	print("="*60)
	print(f"Total episodes trained: {len(rewards)}")
	print(f"Final epsilon: {agent.epsilon:.3f}")
	print(f"Average reward over training: {np.mean(rewards):.3f}")

if __name__ == "__main__":
	main()
