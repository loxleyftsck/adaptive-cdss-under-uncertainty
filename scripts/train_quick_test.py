"""Quick training script for Q-Learning agent."""

import sys
sys.path.insert(0, "C:\\Users\\LENOVO\\Documents\\adaptive-cdss-under-uncertainty")

from src.environment.cdss_env import CDSSEnvironment
from src.agents.q_learning import QLearningAgent
from src.training.trainer import Trainer

print("=" * 60)
print("Q-LEARNING TRAINING - QUICK TEST (100 episodes)")
print("=" * 60)

# Setup
env = CDSSEnvironment(knowledge_path="knowledge/", missing_rate=0.4, seed=42)
agent = QLearningAgent(
    n_actions=4,
    alpha=0.1,
    gamma=0.95,
    epsilon=0.2,
    seed=42
)

trainer = Trainer(
    env=env,
    agent=agent,
    experiment_name="CDSS-QuickTest",
    run_name="Q-Learning-100ep"
)

# Train
metrics = trainer.train(
    n_episodes=100,  #  Quick test
    log_interval=25,
    save_interval=100
)

# Results
print("\n" + "=" * 60)
print("TRAINING RESULTS")
print("=" * 60)
print(f"Total Episodes: 100")
print(f"Cumulative Reward: {metrics['cumulative_reward']:.2f}")
print(f"Avg Reward: {metrics['cumulative_reward']/100:.2f}")
print(f"Q-table Size: {metrics['q_table_size']}")

print("\nSafety Metrics:")
print(f"  False Negatives: {metrics['false_negatives']}")
print(f"  False Positives: {metrics['false_positives']}")
print(f"  True Positives: {metrics['true_positives']}")
print(f"  True Negatives: {metrics['true_negatives']}")

safety = trainer.get_safety_metrics()
print(f"\nSafety Scores:")
print(f"  Precision: {safety['precision']:.2%}")
print(f"  Recall: {safety['recall']:.2%}")
print(f"  F1 Score: {safety['f1']:.2%}")
print(f"  FN Rate: {safety['false_negative_rate']:.2%}")

print("\n✅ Training pipeline working!")
