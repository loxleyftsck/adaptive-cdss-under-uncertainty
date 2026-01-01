"""SARSA agent for prescription safety.

Implements on-policy temporal difference learning with epsilon-greedy
exploration for discrete action spaces.
"""

import random
import pickle
from typing import Dict, Optional, Tuple
from collections import defaultdict
from src.agents.state_encoding import encode_state
from src.environment.observation_model import Observation


class SARSAAgent:
    """SARSA agent with epsilon-greedy exploration (on-policy).
    
    Implements on-policy TD(0) learning:
    Q(s,a) ← Q(s,a) + α[r + γ Q(s',a') - Q(s,a)]
    
    More conservative than Q-Learning, suitable for safety-critical domains.
    
    Args:
        n_actions: Number of possible actions (default: 4)
        alpha: Learning rate (default: 0.1)
        gamma: Discount factor (default: 0.95)
        epsilon: Exploration rate (default: 0.25, slightly higher than Q-Learning)
        seed: Random seed for reproducibility
    """
    
    def __init__(
        self,
        n_actions: int = 4,
        alpha: float = 0.1,
        gamma: float = 0.95,
        epsilon: float = 0.25,
        seed: Optional[int] = None,
    ):
        """Initialize SARSA agent with hyperparameters.
        
        Args:
            n_actions: Action space size
            alpha: Learning rate (0-1)
            gamma: Discount factor (0-1)
            epsilon: Exploration probability (0-1), higher for on-policy safety
            seed: Optional random seed
        """
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        
        # Q-table: dict of (state, action) -> Q-value
        self.q_table: Dict[Tuple, float] = defaultdict(float)
        
        # Random number generator
        self.rng = random.Random(seed)
        
        # Statistics
        self.total_updates = 0
    
    def choose_action(
        self, observation: Observation, training: bool = True
    ) -> int:
        """Choose action using epsilon-greedy policy (on-policy).
        
        Args:
            observation: Current observation
            training: If True, use epsilon-greedy. If False, use greedy.
            
        Returns:
            Action index (0-3)
        """
        state = encode_state(observation)
        
        # Epsilon-greedy (same as Q-Learning, but update differs)
        if training and self.rng.random() < self.epsilon:
            return self.rng.randint(0, self.n_actions - 1)
        else:
            return self._greedy_action(state)
    
    def _greedy_action(self, state: Tuple) -> int:
        """Select greedy action (max Q-value).
        
        Args:
            state: Encoded state
            
        Returns:
            Best action
        """
        q_values = [self.q_table[(state, a)] for a in range(self.n_actions)]
        max_q = max(q_values)
        best_actions = [a for a, q in enumerate(q_values) if q == max_q]
        return self.rng.choice(best_actions)
    
    def update(
        self,
        observation: Observation,
        action: int,
        reward: float,
        next_observation: Optional[Observation] = None,
        next_action: Optional[int] = None,
        done: bool = True,
    ) -> float:
        """Update Q-value using SARSA rule (on-policy).
        
        Q(s,a) ← Q(s,a) + α[r + γ Q(s',a') - Q(s,a)]
        
        Note: Requires next_action (the actual action taken in next state).
        
        Args:
            observation: Current observation
            action: Action taken
            reward: Reward received
            next_observation: Next observation (None if terminal)
            next_action: Next action (must be provided if not terminal)
            done: Whether episode ended
            
        Returns:
            TD error
        """
        state = encode_state(observation)
        current_q = self.q_table[(state, action)]
        
        # TD target (on-policy: uses actual next action, not max)
        if done or next_observation is None:
            td_target = reward
        else:
            if next_action is None:
                raise ValueError("next_action required for SARSA update when not done")
            next_state = encode_state(next_observation)
            next_q = self.q_table[(next_state, next_action)]
            td_target = reward + self.gamma * next_q
        
        # TD error
        td_error = td_target - current_q
        
        # Update Q-value
        self.q_table[(state, action)] = current_q + self.alpha * td_error
        
        self.total_updates += 1
        
        return td_error
    
    def save(self, filepath: str) -> None:
        """Save Q-table to file."""
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
        """Load Q-table from file."""
        with open(filepath, "rb") as f:
            data = pickle.load(f)
            self.q_table = defaultdict(float, data["q_table"])
            self.total_updates = data.get("total_updates", 0)
            
            hyper = data.get("hyperparameters", {})
            self.n_actions = hyper.get("n_actions", self.n_actions)
            self.alpha = hyper.get("alpha", self.alpha)
            self.gamma = hyper.get("gamma", self.gamma)
            self.epsilon = hyper.get("epsilon", self.epsilon)
    
    def get_q_table_size(self) -> int:
        """Get Q-table size."""
        return len(self.q_table)
    
    def __repr__(self) -> str:
        return (
            f"SARSAAgent(α={self.alpha}, γ={self.gamma}, "
            f"ε={self.epsilon}, Q-table size={len(self.q_table)})"
        )
