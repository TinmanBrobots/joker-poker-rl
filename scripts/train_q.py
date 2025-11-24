#!/user/bin/env python3
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
from agents.q_agent import QAgent


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


def train_q_agent(num_episodes=10000, eval_every=1000):
	"""
	Train an RL agent on Jacks-or-Better.

	Args:
		num_episodes: Total episodes to train for
		eval_every: Evaluate agent every N episodes
	"""
	# Create environment and agent
	env = PokerEnv()
	agent = QAgent(
		observation_space=env.observation_space,
		action_space=env.action_space,
		learning_rate=0.1,
		discount_factor=0.95,
		epsilon_start=1.0,
		epsilon_end=0.01,
		epsilon_decay=0.9995,
		seed=42
	)

	# Create random agent for baseline comparison
	random_agent = RandomAgent(env.observation_space, env.action_space, seed=42)

	print("Starting RL training...")
	print(f"Training for {num_episodes} episodes")
	print("-" * 50)

	# Evaluate baseline random agent
	random_reward, random_win_rate = evaluate_agent(random_agent, env, 10000)
	print(f"Random Agent")
	print(f"  - Avg Reward: {random_reward:.3f}")
	print(f"  - Win Rate: {random_win_rate:.1%}")

	# Training loop
	rewards_history = []
	win_rates_history = []

	for episode in tqdm(range(num_episodes), desc="Training"):
		# Reset environment
		obs, _ = env.reset()

		# Agent takes action
		action = agent.act(obs)

		# Environment responds
		next_obs, reward, terminated, _, _ = env.step(action)

		# Agent learns from experience
		experience = {
			'state': obs,
			'action': action,
			'reward': reward,
			'next_state': next_obs,
			'done': terminated
		}

		metrics = agent.learn(experience)

		# Track rewards
		rewards_history.append(reward)

		# Periodic evaluation
		if (episode + 1) % eval_every == 0:
			avg_reward, win_rate = evaluate_agent(agent, env, 1000)
			win_rates_history.append(win_rate)

			tqdm.write(f"Episode {episode+1:6d} | "
					  f"Avg Reward: {avg_reward:.3f} | "
					  f"Win Rate: {win_rate:.1%} | "
					  f"Epsilon: {metrics['epsilon']:.3f} | "
					  f"States Learned: {metrics['states_learned']}")

	print("\nTraining completed!")
	print("-" * 50)

	# Final evaluation
	final_reward, final_win_rate = evaluate_agent(agent, env, 5000)
	print(f"Final Evaluation")
	print(f"  - Avg Reward: {final_reward:.3f}")
	print(f"  - Win Rate: {final_win_rate:.1%}")
	print(f"Improvement over random: {final_win_rate - random_win_rate:.1%}")

	# Save trained agent
	agent.save('models/q_agent.pkl')
	print("Model saved to models/q_agent.pkl")

	return agent, rewards_history, win_rates_history


def main():
	"""Main training function."""
	# Set random seeds for reproducibility
	np.random.seed(42)

	# Create models directory
	os.makedirs('models', exist_ok=True)

	# Train agent
	agent, rewards, win_rates = train_q_agent(num_episodes=10_000_000, eval_every=1_000_000)

	print("\n" + "="*60)
	print("TRAINING SUMMARY")
	print("="*60)
	print(f"Total episodes trained: {len(rewards)}")
	print(f"States learned: {len(agent.q_table)} / {math.comb(52, 5)}")
	print(f"Final epsilon: {agent.epsilon:.3f}")
	print(f"Average reward over training: {np.mean(rewards):.3f}")

if __name__ == "__main__":
	main()
