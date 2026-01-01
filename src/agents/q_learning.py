"""Q-Learning agent for prescription safety.

Implements off-policy temporal difference learning with epsilon-greedy
exploration for discrete action spaces.
"""

import random
import pickle
from typing import Dict, Optional, Tuple
from collections import defaultdict
from src.agents.state_encoding import encode_state
from src.environment.observation_model import Observation


class QLearningAgent:
    """Q-Learning agent with epsilon-greedy exploration.
    
    Implements off-policy TD(0) learning:
    Q(s,a) ← Q(s,a) + α[r + γ max_a' Q(s',a') - Q(s,a)]
    
    Args:
        n_actions: Number of possible actions (default: 4)
        alpha: Learning rate (default: 0.1)
        gamma: Discount factor (default: 0.95)
        epsilon: Exploration rate (default: 0.2)
        seed: Random seed for reproducibility
    """
    
    def __init__(
        self,
        n_actions: int = 4,
        alpha: float = 0.1,
        gamma: float = 0.95,
        epsilon: float = 0.2,
        seed: Optional[int] = None,
    ):
        """Initialize Q-Learning agent with hyperparameters.
        
        Args:
            n_actions: Action space size
            alpha: Learning rate (0-1)
            gamma: Discount factor (0-1)
            epsilon: Exploration probability (0-1)
            seed: Optional random seed
        """
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        
        # Q-table: dict of (state, action) -> Q-value
        # Default to 0.0 (optimistic initialization)
        self.q_table: Dict[Tuple, float] = defaultdict(float)
        
        # Random number generator
        self.rng = random.Random(seed)
        
        # Statistics
        self.total_updates = 0
    
    def choose_action(
        self, observation: Observation, training: bool = True
    ) -> int:
        """Choose action using epsilon-greedy policy.
        
        Args:
            observation: Current observation (partial patient info)
            training: If True, use epsilon-greedy. If False, use greedy.
            
        Returns:
            Action index (0-3)
        """
        state = encode_state(observation)
        
        # Epsilon-greedy exploration (only during training)
        if training and self.rng.random() < self.epsilon:
            # Explore: random action
            return self.rng.randint(0, self.n_actions - 1)
        else:
            # Exploit: greedy action
            return self._greedy_action(state)
    
    def _greedy_action(self, state: Tuple) -> int:
        """Select action with highest Q-value (greedy policy).
        
        Args:
            state: Encoded state
            
        Returns:
            Action with max Q-value (ties broken randomly)
        """
        # Get Q-values for all actions in this state
        q_values = [self.q_table[(state, a)] for a in range(self.n_actions)]
        
        # Find max Q-value
        max_q = max(q_values)
        
        # Get all actions with max Q-value (for tie-breaking)
        best_actions = [a for a, q in enumerate(q_values) if q == max_q]
        
        # Random tie-breaking
        return self.rng.choice(best_actions)
    
    def update(
        self,
        observation: Observation,
        action: int,
        reward: float,
        next_observation: Optional[Observation] = None,
        done: bool = True,
    ) -> float:
        """Update Q-value using Q-Learning rule (off-policy).
        
        Q(s,a) ← Q(s,a) + α[r + γ max_a' Q(s',a') - Q(s,a)]
        
        Args:
            observation: Current observation
            action: Action taken
            reward: Reward received
            next_observation: Next observation (None if terminal)
            done: Whether episode ended
            
        Returns:
            TD error (for logging)
        """
        state = encode_state(observation)
        
        # Current Q-value
        current_q = self.q_table[(state, action)]
        
        # TD target
        if done or next_observation is None:
            # Terminal state: no future value
            td_target = reward
        else:
            # Non-terminal: add discounted max future Q-value
            next_state = encode_state(next_observation)
            max_next_q = max(
                [self.q_table[(next_state, a)] for a in range(self.n_actions)]
            )
            td_target = reward + self.gamma * max_next_q
        
        # TD error
        td_error = td_target - current_q
        
        # Update Q-value
        self.q_table[(state, action)] = current_q + self.alpha * td_error
        
        self.total_updates += 1
        
        return td_error
    
    def save(self, filepath: str) -> None:
        """Save Q-table to file.
        
        Args:
            filepath: Path to save Q-table (pickle format)
        """
        with open(filepath, "wb") as f:
            pickle.dump(
                {
                    "q_table": dict(self.q_table),
                    "hyperparameters": {
                        "n_actions": self.n_actions,
                        "alpha": self.alpha,
                        "gamma": self.gamma,
                        "epsilon": self.epsilon,
                    },
                    "total_updates": self.total_updates,
                },
                f,
            )
    
    def load(self, filepath: str) -> None:
        """Load Q-table from file.
        
        Args:
            filepath: Path to Q-table file
        """
        with open(filepath, "rb") as f:
            data = pickle.load(f)
            self.q_table = defaultdict(float, data["q_table"])
            self.total_updates = data.get("total_updates", 0)
            
            # Optionally restore hyperparameters
            hyper = data.get("hyperparameters", {})
            self.n_actions = hyper.get("n_actions", self.n_actions)
            self.alpha = hyper.get("alpha", self.alpha)
            self.gamma = hyper.get("gamma", self.gamma)
            self.epsilon = hyper.get("epsilon", self.epsilon)
    
    def get_q_table_size(self) -> int:
        """Get number of (state, action) pairs in Q-table.
        
        Returns:
            Q-table size
        """
        return len(self.q_table)
    
    def __repr__(self) -> str:
        """String representation of agent."""
        return (
            f"QLearningAgent(α={self.alpha}, γ={self.gamma}, "
            f"ε={self.epsilon}, Q-table size={len(self.q_table)})"
        )
