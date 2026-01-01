"""🚀 AUTOMATED TRAINING - Q-Learning (5000 Episodes)

Fully automated, no user input required.
Starts immediately and runs to completion.
"""

import sys
import os
sys.path.insert(0, "C:\\Users\\LENOVO\\Documents\\adaptive-cdss-under-uncertainty")

from src.environment.cdss_env import CDSSEnvironment
from src.agents.q_learning import QLearningAgent
from src.training.trainer import Trainer

print("=" * 70)
print("🚀 AUTOMATED TRAINING - Q-LEARNING (5000 EPISODES)")
print("=" * 70)
print("\nConfiguration:")
print("  Episodes: 5000")
print("  Agent: Q-Learning")
print("  Alpha: 0.1, Gamma: 0.95, Epsilon: 0.2")
print("  Missing Rate: 40%")
print("  MLflow: Enabled (live logging)")
print("  Checkpoints: Every 500 episodes")
print("\n" + "=" * 70)

# Setup
print("\n[1/4] Initializing environment and agent...")
env = CDSSEnvironment(knowledge_path="knowledge/", missing_rate=0.4, seed=42)
agent = QLearningAgent(n_actions=4, alpha=0.1, gamma=0.95, epsilon=0.2, seed=42)
print("  ✅ Ready")

# Create trainer
print("\n[2/4] Creating trainer with MLflow...")
trainer = Trainer(
    env=env,
    agent=agent,
    experiment_name="CDSS-Production-Training",
    run_name="Q-Learning-5000ep-Auto",
    use_mlflow=True
)
print("  ✅ Trainer initialized")

# MLflow instructions
print("\n" + "=" * 70)
print("📊 WATCH LIVE IN MLFLOW UI:")
print("   http://127.0.0.1:5000")
print("   Experiment: CDSS-Production-Training")
print("   Run: Q-Learning-5000ep-Auto")
print("=" * 70)

# Start training immediately (NO INPUT REQUIRED)
print("\n[3/4] Starting training NOW...")
print("=" * 70)

try:
    metrics = trainer.train(
        n_episodes=5000,
        log_interval=100,
        save_interval=500,
        checkpoint_dir="models/checkpoints",
        early_stopping_patience=None
    )
    
    print("\n" + "=" * 70)
    print("✅ TRAINING COMPLETE!")
    print("=" * 70)
    
except KeyboardInterrupt:
    print("\n⚠️  Training interrupted")
except Exception as e:
    print(f"\n❌ Training failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Save to production
print("\n[4/4] Saving production model...")
os.makedirs("models/production", exist_ok=True)
production_path = "models/production/q_learning_5000ep.pkl"
agent.save(production_path)

if os.path.exists(production_path):
    size_kb = os.path.getsize(production_path) / 1024
    print(f"  ✅ Saved: {production_path} ({size_kb:.2f} KB)")

# Final stats
print("\n" + "=" * 70)
print("📊 FINAL STATISTICS")
print("=" * 70)
print(f"\nPerformance:")
print(f"  Episodes: {len(metrics['episode_rewards'])}")
print(f"  Cumulative Reward: {metrics['cumulative_reward']:.2f}")
print(f"  Average Reward: {metrics['cumulative_reward'] / len(metrics['episode_rewards']):.2f}")
print(f"  Best Avg (100): {metrics['best_avg_reward']:.2f}")
print(f"  Q-table Size: {metrics['q_table_size']}")

print(f"\nSafety:")
print(f"  False Negatives: {metrics['false_negatives']}")
print(f"  False Positives: {metrics['false_positives']}")
print(f"  True Positives: {metrics['true_positives']}")
print(f"  True Negatives: {metrics['true_negatives']}")

safety = trainer.get_safety_metrics()
print(f"\nScores:")
print(f"  Precision: {safety['precision']:.2%}")
print(f"  Recall: {safety['recall']:.2%}")
print(f"  F1 Score: {safety['f1']:.2%}")
print(f"  FN Rate: {safety['fn_rate']:.2%}")

print("\n" + "=" * 70)
print("🎉 Q-LEARNING TRAINING COMPLETED!")
print("=" * 70)
print(f"\nModel: {production_path}")
print(f"Checkpoints: models/checkpoints/")
print(f"MLflow: http://127.0.0.1:5000")
print("=" * 70)
