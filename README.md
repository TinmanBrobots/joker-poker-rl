# joker-poker-rl
A simple game environment for training an RL agent to play the game Joker Poker (Jacks-or-Better)



# Phase 0: Project Setup


## GitHub Repository Creation

Create a new repository on GitHub with:
- [x] Repository name: joker-poker-rl or similar
- [x] Initialize with a README.md
- [x] Add a Python .gitignore
- [x] Choose MIT license
- [x] Clone and Initial File Structure
- [x] After cloning your repo, create this organized structure:

## Clone and Initial File Structure
After cloning your repo, create this organized structure:

joker-poker-rl/
├── README.md
├── requirements.txt
├── setup.py
├── .gitignore
├── src/
│   ├── __init__.py
│   ├── game/
│   │   ├── __init__.py
│   │   ├── deck.py
│   │   ├── poker_hand.py
│   │   └── joker_poker.py
│   ├── environment/
│   │   ├── __init__.py
│   │   ├── poker_env.py
│   │   └── reward_system.py
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── base_agent.py
│   │   ├── random_agent.py
│   │   └── rl_agent.py
│   └── training/
│       ├── __init__.py
│       ├── trainer.py
│       └── evaluation.py
├── tests/
│   ├── __init__.py
│   ├── test_game.py
│   ├── test_environment.py
│   └── test_agent.py
├── notebooks/
│   ├── exploration.ipynb
│   └── analysis.ipynb
├── scripts/
│   ├── train.py
│   ├── evaluate.py
│   └── play_game.py
└── config/
    └── default.yaml



# Phase 1: Game Environment Creation


## Core Game Components

### Deck Management (`deck.py`):
- Use a `collections.deque` for efficient card drawing
- Consider using named tuples or dataclasses for card representation: `Card = namedtuple('Card', ['rank', 'suit'])`
- Implement shuffle using `random.shuffle()`

### Poker Hand Evaluation (`poker_hand.py`):
- Create an enum for hand rankings (High Card, Pair, Two Pair, etc.)
- Implement hand evaluation logic following standard poker rules
- Use bit manipulation for efficient hand comparison if performance becomes an issue

### Game Logic (`joker_poker.py`):
- Implement game state as a class with methods for dealing, drawing, and scoring
- Use a set or list to track which cards are held/discarded
- Track game phase (initial deal, draw phase, final hand)
- Include validation for legal moves


## RL Environment Design

### State Representation:
- Option 1: Flat vector representation (52 cards + hand state)
- Option 2: Dictionary with structured information
- Option 3: Image-like representation (5x13 grid for cards)

### Action Space:
- Discrete actions: For each of the 5 card positions, decide hold/discard (2^5 = 32 possible actions)
- Multi-discrete: Separate action for each card position
- Consider simplified action space initially (fewer actions = faster learning)

### Reward Structure:
- Sparse rewards: Only at end of hand based on final poker hand value
- Dense rewards: Intermediate rewards for improving hand strength during draw phase
- Shaped rewards: Bonus for achieving certain hand types, penalties for poor decisions


## Recommended Libraries

### Core Dependencies:
```
	gymnasium>=0.29.0        # Modern successor to OpenAI Gym
	numpy>=1.24.0           # Numerical computations
	pandas>=2.0.0           # Data manipulation and analysis
	pytest>=7.4.0           # Testing framework
	pyyaml>=6.0             # Configuration files
```

### Optional but Recommended:
```
	pydantic>=2.0.0         # Data validation and settings
	loguru>=0.7.0          # Better logging than standard library
	tqdm>=4.65.0           # Progress bars
	matplotlib>=3.7.0      # Plotting and visualization
	seaborn>=0.12.0        # Statistical visualizations
```



# Phase 2: Agent and Training Environment


## Agent Architecture

### Base Agent Interface (base_agent.py):
Abstract base class with act(state) and learn(experience) methods
Support for both discrete and continuous action spaces
Include methods for saving/loading agent state
Random Agent (random_agent.py):
Simple baseline agent that makes random decisions
Useful for testing environment and establishing baseline performance

### RL Agent (rl_agent.py):
Choose algorithm based on action space size and complexity
Consider these algorithms for different scenarios:
Q-Learning: Good for small discrete action spaces (32 actions)
Deep Q-Network (DQN): Handles larger state spaces well
Policy Gradient (REINFORCE): Good for learning stochastic policies
Actor-Critic (A2C/PPO): More stable training, better for complex environments


## Training Infrastructure

### Experience Replay:
Use collections.deque with max length for efficient memory management
Store transitions as named tuples: Transition = namedtuple('Transition', ['state', 'action', 'reward', 'next_state', 'done'])

### Training Loop (trainer.py):
Implement epsilon-greedy exploration strategy
Include learning rate scheduling
Add early stopping based on performance metrics
Support for multiple random seeds for reproducible results


## Recommended RL Libraries

### Primary RL Framework:
```
	stable-baselines3>=2.0.0    # High-level RL
	algorithmstorch>=2.0.0      # PyTorch for neural networks
```

### Alternative Options:
```
	ray[rllib]>=2.7.0           # Distributed RL
	trainingtianshou>=0.5.0     # Research-oriented RL framework
```



# Phase 3: Training the Agent


## Training Configuration

### Hyperparameter Management:
- Use YAML configuration files for experiment management
- Include parameters for:
	- Learning rate, batch size, replay buffer size
	- Exploration parameters (epsilon start/end/decay)
	- Network architecture (hidden layers, activation functions)
	- Training duration (episodes, timesteps)

### Experiment Tracking:
- Log training metrics (episode reward, loss, exploration rate)
- Track evaluation performance against random agent
- Save model checkpoints periodically


## Evaluation and Analysis

### Performance Metrics:
- Average hand score achieved
- Win rate against random baseline
- Hand type distribution (percentage of each poker hand type)
- Learning curves showing improvement over time

### Visualization:
- Plot training progress (reward vs episodes)
- Analyze action distributions during different game phases
- Create heatmaps showing state-action value functions

## Training Best Practices
1. **Start Simple**: Begin with basic Q-learning before moving to deep RL
2. **Baseline Testing**: Always compare against random agent performance
3. **Hyperparameter Search**: Use grid search or random search for optimal parameters
4. **Reproducibility**: Set random seeds and log all hyperparameters
5. **Regular Evaluation**: Test agent performance every N episodes during training
6. **Model Saving**: Save best models and periodic checkpoints


## Advanced Considerations

### Curriculum Learning:
- Start with simpler hand types, gradually introduce more complex ones
- Begin with unlimited draws, then restrict to standard rules

### Multi-Agent Training:
- Train against itself (self-play) for potentially stronger agents
- Compare different algorithm variants

### State Space Optimization:
- Consider state abstraction techniques if learning is slow
- Use convolutional networks if using image-like state representation
