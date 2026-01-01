"""Enhanced Trainer with MLflow logging and early stopping."""

import random
import os
from typing import Dict, List, Optional, Any
from tqdm import tqdm

try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    print("⚠️  MLflow not installed - logging disabled")

from src.environment.cdss_env import CDSSEnvironment


class Trainer:
    """Trainer for RL agents with MLflow integration and early stopping.
    
    Features:
    - Episode sampling and agent updates
    - MLflow experiment logging (metrics, params, artifacts)
    - Model checkpointing every N episodes
    - Early stopping if no improvement
    - Learning curve tracking
    
    Args:
        env: CDSS environment
        agent: RL agent (Q-Learning or SARSA)
        experiment_name: MLflow experiment name
        run_name: MLflow run name
        use_mlflow: Whether to use MLflow logging
    """
    
    def __init__(
        self,
        env: CDSSEnvironment,
        agent: Any,
        experiment_name: str = "CDSS-RL-Training",
        run_name: Optional[str] = None,
        use_mlflow: bool = True,
    ):
        """Initialize trainer."""
        self.env = env
        self.agent = agent
        self.experiment_name = experiment_name
        self.run_name = run_name or agent.__class__.__name__
        self.use_mlflow = use_mlflow and MLFLOW_AVAILABLE
        
        # Training statistics
        self.episode_rewards: List[float] = []
        self.cumulative_reward = 0.0
        
        # Safety metrics
        self.false_negatives = 0
        self.false_positives = 0
        self.true_positives = 0
        self.true_negatives = 0
        
        # Early stopping
        self.best_avg_reward = float('-inf')
        self.episodes_without_improvement = 0
    
    def train(
        self,
        n_episodes: int = 5000,
        log_interval: int = 100,
        save_interval: int = 500,
        checkpoint_dir: str = "models/checkpoints",
        early_stopping_patience: Optional[int] = 500,
    ) -> Dict[str, Any]:
        """Train agent for N episodes with MLflow logging.
        
        Args:
            n_episodes: Number of training episodes
            log_interval: Log metrics every N episodes
            save_interval: Save checkpoint every N episodes
            checkpoint_dir: Directory to save checkpoints
            early_stopping_patience: Stop if no improvement for N episodes (None = disabled)
            
        Returns:
            Dictionary of training metrics
        """
        print(f"🚀 Starting training: {self.run_name}")
        print(f"   Episodes: {n_episodes}")
        print(f"   Agent: {self.agent}")
        print(f"   Environment: {self.env}")
        print(f"   MLflow: {'✅ Enabled' if self.use_mlflow else '❌ Disabled'}")
        print(f"   Early Stopping: {'✅ Enabled' if early_stopping_patience else '❌ Disabled'}")
        print("=" * 70)
        
        # MLflow setup
        if self.use_mlflow:
            self._setup_mlflow()
        
        # Training loop
        for episode in tqdm(range(n_episodes), desc="Training"):
            # Run episode
            reward, info = self._run_episode()
            
            # Track metrics
            self.episode_rewards.append(reward)
            self.cumulative_reward += reward
            
            # Safety metrics
            self._update_safety_metrics(info)
            
            # Logging to MLflow and console
            if (episode + 1) % log_interval == 0:
                self._log_metrics(episode + 1)
            
            # Checkpointing
            if (episode + 1) % save_interval == 0:
                self._save_checkpoint(episode + 1, checkpoint_dir)
            
            # Early stopping check
            if early_stopping_patience and (episode + 1) % 100 == 0:
                if self._check_early_stopping(early_stopping_patience):
                    print(f"\n⚠️  Early stopping triggered at episode {episode + 1}")
                    break
        
        # Final save
        final_episode = len(self.episode_rewards)
        self._save_checkpoint(final_episode, checkpoint_dir, final=True)
        
        # End MLflow run
        if self.use_mlflow:
            mlflow.end_run()
        
        print("\n✅ Training complete!")
        print(f"   Total episodes: {final_episode}")
        print(f"   Final avg reward (last 100): {self._moving_average(100):.2f}")
        print(f"   Q-table size: {self.agent.get_q_table_size()}")
        
        return self._get_metrics()
    
    def _setup_mlflow(self) -> None:
        """Setup MLflow experiment and run."""
        mlflow.set_experiment(self.experiment_name)
        mlflow.start_run(run_name=self.run_name)
        
        # Log hyperparameters
        mlflow.log_param("agent_type", self.agent.__class__.__name__)
        mlflow.log_param("alpha", getattr(self.agent, 'alpha', 'N/A'))
        mlflow.log_param("gamma", getattr(self.agent, 'gamma', 'N/A'))
        mlflow.log_param("epsilon", getattr(self.agent, 'epsilon', 'N/A'))
        mlflow.log_param("missing_rate", self.env.obs_model.missing_rate)
        
        print("✅ MLflow run started")
    
    def _run_episode(self) -> tuple:
        """Run single episode."""
        obs = self.env.reset()
        action = self.agent.choose_action(obs, training=True)
        next_obs, reward, done, info = self.env.step(action)
        
        # Update agent
        if hasattr(self.agent, 'update'):
            from src.agents.q_learning import QLearningAgent
            from src.agents.sarsa import SARSAAgent
            
            if isinstance(self.agent, QLearningAgent):
                self.agent.update(obs, action, reward, next_obs, done)
            elif isinstance(self.agent, SARSAAgent):
                next_action = self.agent.choose_action(next_obs, training=True) if not done else None
                self.agent.update(obs, action, reward, next_obs, next_action, done)
        
        return reward, info
    
    def _update_safety_metrics(self, info: Dict[str, Any]) -> None:
        """Update safety metrics."""
        action = info["action"]
        true_risk = info["true_risk"]
        
        if action == 0:  # APPROVE
            if true_risk >= 8:
                self.false_negatives += 1  # Dangerous!
            elif true_risk == 0:
                self.true_negatives += 1
        elif action in [1, 2]:  # WARN or SUGGEST_ALT
            if true_risk >= 3:
                self.true_positives += 1
            else:
                self.false_positives += 1
    
    def _log_metrics(self, episode: int) -> None:
        """Log metrics to console and MLflow."""
        avg_reward_100 = self._moving_average(100)
        
        if episode >= 100:
            print(f"\nEpisode {episode}:")
            print(f"  Avg Reward (last 100): {avg_reward_100:.2f}")
            print(f"  Q-table size: {self.agent.get_q_table_size()}")
            print(f"  False Negatives: {self.false_negatives}")
            print(f"  False Positives: {self.false_positives}")
            
            # MLflow logging
            if self.use_mlflow:
                mlflow.log_metric("avg_reward_100", avg_reward_100, step=episode)
                mlflow.log_metric("q_table_size", self.agent.get_q_table_size(), step=episode)
                mlflow.log_metric("false_negatives", self.false_negatives, step=episode)
                mlflow.log_metric("false_positives", self.false_positives, step=episode)
                mlflow.log_metric("cumulative_reward", self.cumulative_reward, step=episode)
                
                # Log current epsilon (if available)
                if hasattr(self.agent, 'epsilon'):
                    mlflow.log_metric("epsilon", self.agent.epsilon, step=episode)
    
    def _save_checkpoint(
        self, episode: int, checkpoint_dir: str, final: bool = False
    ) -> None:
        """Save agent checkpoint."""
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        suffix = "final" if final else f"ep{episode}"
        filepath = os.path.join(
            checkpoint_dir,
            f"{self.agent.__class__.__name__}_{suffix}.pkl"
        )
        
        self.agent.save(filepath)
        
        # Verify checkpoint was created
        if os.path.exists(filepath):
            size_kb = os.path.getsize(filepath) / 1024
            if final:
                print(f"\n💾 Final checkpoint saved: {filepath} ({size_kb:.1f} KB)")
            else:
                print(f"\n💾 Checkpoint saved: {filepath} ({size_kb:.1f} KB)")
            
            # Log to MLflow
            if self.use_mlflow:
                mlflow.log_artifact(filepath)
        else:
            print(f"\n❌ ERROR: Checkpoint NOT saved: {filepath}")
    
    def _check_early_stopping(self, patience: int) -> bool:
        """Check if early stopping criteria met.
        
        Args:
            patience: Episodes without improvement to trigger stop
            
        Returns:
            True if should stop early
        """
        current_avg = self._moving_average(100)
        
        if current_avg > self.best_avg_reward:
            self.best_avg_reward = current_avg
            self.episodes_without_improvement = 0
            return False
        else:
            self.episodes_without_improvement += 100  # Check every 100 episodes
            return self.episodes_without_improvement >= patience
    
    def _moving_average(self, window: int) -> float:
        """Calculate moving average of rewards."""
        if len(self.episode_rewards) < window:
            return sum(self.episode_rewards) / len(self.episode_rewards) if self.episode_rewards else 0.0
        return sum(self.episode_rewards[-window:]) / window
    
    def _get_metrics(self) -> Dict[str, Any]:
        """Get all training metrics."""
        return {
            "episode_rewards": self.episode_rewards,
            "cumulative_reward": self.cumulative_reward,
            "false_negatives": self.false_negatives,
            "false_positives": self.false_positives,
            "true_positives": self.true_positives,
            "true_negatives": self.true_negatives,
            "q_table_size": self.agent.get_q_table_size(),
            "best_avg_reward": self.best_avg_reward,
        }
    
    def get_safety_metrics(self) -> Dict[str, float]:
        """Calculate safety metrics."""
        total_decisions = (
            self.false_negatives + self.false_positives +
            self.true_positives + self.true_negatives
        )
        
        if total_decisions == 0:
            return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "fn_rate": 0.0}
        
        precision = (
            self.true_positives / (self.true_positives + self.false_positives)
            if (self.true_positives + self.false_positives) > 0 else 0.0
        )
        recall = (
            self.true_positives / (self.true_positives + self.false_negatives)
            if (self.true_positives + self.false_negatives) > 0 else 0.0
        )
        f1 = (
            2 * (precision * recall) / (precision + recall)
            if (precision + recall) > 0 else 0.0
        )
        
        return {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "fn_rate": self.false_negatives / total_decisions,
        }
