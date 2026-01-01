"""Random baseline policy for comparison."""

import random
from typing import Optional
from src.environment.observation_model import Observation


class RandomPolicy:
    """Random action selection baseline.
    
    Provides lower bound on performance - any learning agent should
    outperform random action selection.
    
    Args:
        n_actions: Number of possible actions (default: 4)
        seed: Random seed for reproducibility
    """
    
    def __init__(self, n_actions: int = 4, seed: Optional[int] = None):
        """Initialize random policy.
        
        Args:
            n_actions: Action space size
            seed: Optional random seed
        """
        self.n_actions = n_actions
        self.rng = random.Random(seed)
    
    def choose_action(self, observation: Observation, training: bool = True) -> int:
        """Choose random action (ignores observation).
        
        Args:
            observation: Current observation (ignored)
            training: Training flag (ignored)
            
        Returns:
            Random action index (0 to n_actions-1)
        """
        return self.rng.randint(0, self.n_actions - 1)
    
    def update(self, *args, **kwargs) -> float:
        """No-op update (random policy doesn't learn).
        
        Returns:
            0.0 (no learning)
        """
        return 0.0
    
    def __repr__(self) -> str:
        return f"RandomPolicy(n_actions={self.n_actions})"
