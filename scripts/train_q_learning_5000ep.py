"""🚀 FULL TRAINING - Q-Learning Agent (5000 Episodes)

This script runs production training with:
- 5000 episodes
- MLflow experiment tracking
- Checkpoints every 500 episodes
- Safety metrics logging
- Final model save to production/
"""

import sys
import os
sys.path.insert(0, "C:\\Users\\LENOVO\\Documents\\adaptive-cdss-under-uncertainty")

from src.environment.cdss_env import CDSSEnvironment
from src.agents.q_learning import QLearningAgent
from src.training.trainer import Trainer

print("=" * 70)
print("🚀 PRODUCTION TRAINING - Q-LEARNING AGENT")
print("=" * 70)
print("\nConfiguration:")
print("  Episodes: 5000")
print("  Agent: Q-Learning (off-policy)")
print("  Alpha: 0.1")
print("  Gamma: 0.95")
print("  Epsilon: 0.2")
print("  Missing Rate: 40%")
print("  MLflow: Enabled")
print("  Checkpoints: Every 500 episodes")
print("  Early Stopping: Disabled (full 5000 run)")
print("\n" + "=" * 70)

# Setup environment and agent
print("\n[1/3] Initializing environment and agent...")
env = CDSSEnvironment(
    knowledge_path="knowledge/",
    missing_rate=0.4,
    seed=42
)

agent = QLearningAgent(
    n_actions=4,
    alpha=0.1,
    gamma=0.95,
    epsilon=0.2,
    seed=42
)
print("  ✅ Environment and agent created")

# Create trainer with MLflow
print("\n[2/3] Creating trainer with MLflow integration...")
trainer = Trainer(
    env=env,
    agent=agent,
    experiment_name="CDSS-Production-Training",
    run_name="Q-Learning-5000ep-MissingRate40",
    use_mlflow=True  # Enable MLflow logging
)
print("  ✅ Trainer initialized")

print("\n" + "=" * 70)
print("📊 MLFLOW UI INSTRUCTIONS:")
print("=" * 70)
print("\nTo watch training metrics in real-time:")
print("  1. Open a NEW terminal/PowerShell window")
print("  2. Navigate to project directory:")
print("     cd C:\\Users\\LENOVO\\Documents\\adaptive-cdss-under-uncertainty")
print("  3. Run MLflow UI:")
print("     mlflow ui")
print("  4. Open your browser to:")
print("     http://127.0.0.1:5000")
print("\n  You'll see live updates of:")
print("    - Avg Reward (last 100 episodes)")
print("    - Q-table size")
print("    - False Negatives/Positives")
print("    - Cumulative Reward")
print("    - Epsilon value")
print("\n" + "=" * 70)

input("\nPress ENTER to start training (or Ctrl+C to cancel)...")

# Start training
print("\n[3/3] Starting full 5000-episode training...")
print("=" * 70)

try:
    metrics = trainer.train(
        n_episodes=5000,
        log_interval=100,  # Log every 100 episodes
        save_interval=500,  # Checkpoint every 500 episodes
        checkpoint_dir="models/checkpoints",
        early_stopping_patience=None  # Disabled - full run
    )
    
    print("\n" + "=" * 70)
    print("✅ TRAINING COMPLETE!")
    print("=" * 70)
    
except KeyboardInterrupt:
    print("\n⚠️  Training interrupted by user")
    print("   Saving current progress...")
    
except Exception as e:
    print(f"\n❌ Training failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Copy final model to production directory
print("\n[4/4] Saving final model to production...")
os.makedirs("models/production", exist_ok=True)
production_path = "models/production/q_learning_agent_5000ep.pkl"
agent.save(production_path)

if os.path.exists(production_path):
    size_kb = os.path.getsize(production_path) / 1024
    print(f"  ✅ Production model saved: {production_path}")
    print(f"  ✅ Size: {size_kb:.2f} KB")
else:
    print(f"  ❌ ERROR: Production model not saved")

# Print final statistics
print("\n" + "=" * 70)
print("📊 FINAL STATISTICS")
print("=" * 70)
print(f"\nTraining Results:")
print(f"  Total Episodes: {len(metrics['episode_rewards'])}")
print(f"  Cumulative Reward: {metrics['cumulative_reward']:.2f}")
print(f"  Average Reward: {metrics['cumulative_reward'] / len(metrics['episode_rewards']):.2f}")
print(f"  Best Avg (last 100): {metrics['best_avg_reward']:.2f}")
print(f"  Final Q-table Size: {metrics['q_table_size']}")

print(f"\nSafety Metrics:")
print(f"  False Negatives: {metrics['false_negatives']}")
print(f"  False Positives: {metrics['false_positives']}")
print(f"  True Positives: {metrics['true_positives']}")
print(f"  True Negatives: {metrics['true_negatives']}")

safety = trainer.get_safety_metrics()
print(f"\nSafety Scores:")
print(f"  Precision: {safety['precision']:.2%}")
print(f"  Recall: {safety['recall']:.2%}")
print(f"  F1 Score: {safety['f1']:.2%}")
print(f"  False Negative Rate: {safety['fn_rate']:.2%}")

print("\n" + "=" * 70)
print("🎉 Q-LEARNING TRAINING COMPLETED SUCCESSFULLY!")
print("=" * 70)
print(f"\nModel Location: {production_path}")
print(f"Checkpoints: models/checkpoints/")
print(f"MLflow Logs: mlruns/")
print("\nNext: Train SARSA agent for comparison")
print("=" * 70)
