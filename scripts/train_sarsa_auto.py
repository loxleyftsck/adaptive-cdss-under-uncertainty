"""🚀 AUTOMATED TRAINING - SARSA (5000 Episodes)

On-policy conservative agent for medical safety.
Same hyperparameters as Q-Learning for fair comparison.
"""

import sys
import os
sys.path.insert(0, "C:\\Users\\LENOVO\\Documents\\adaptive-cdss-under-uncertainty")

from src.environment.cdss_env import CDSSEnvironment
from src.agents.sarsa import SARSAAgent
from src.training.trainer import Trainer

print("=" * 70)
print("🚀 AUTOMATED TRAINING - SARSA (5000 EPISODES)")
print("=" * 70)
print("\nConfiguration:")
print("  Episodes: 5000")
print("  Agent: SARSA (On-Policy)")
print("  Alpha: 0.1, Gamma: 0.95, Epsilon: 0.2")
print("  Missing Rate: 40%")
print("  MLflow: Enabled (SAME experiment as Q-Learning)")
print("  Goal: FN Rate < 10% (vs Q-Learning 12.18%)")
print("\n" + "=" * 70)

# Setup with EXACT SAME parameters as Q-Learning
print("\n[1/4] Initializing environment and agent...")
env = CDSSEnvironment(knowledge_path="knowledge/", missing_rate=0.4, seed=42)
agent = SARSAAgent(
    n_actions=4,
    alpha=0.1,      # SAME as Q-Learning
    gamma=0.95,     # SAME as Q-Learning
    epsilon=0.2,    # SAME as Q-Learning (conservative for safety)
    seed=42
)
print("  ✅ Ready")

# Create trainer - SAME MLflow experiment for comparison
print("\n[2/4] Creating trainer with MLflow...")
trainer = Trainer(
    env=env,
    agent=agent,
    experiment_name="CDSS-Production-Training",  # SAME experiment!
    run_name="SARSA-5000ep-Auto",
    use_mlflow=True
)
print("  ✅ Trainer initialized")

# MLflow instructions
print("\n" + "=" * 70)
print("📊 WATCH LIVE IN MLFLOW UI:")
print("   http://127.0.0.1:5000")
print("   Experiment: CDSS-Production-Training")
print("   Compare: Q-Learning vs SARSA (overlay graphs!)")
print("=" * 70)

# Start training immediately
print("\n[3/4] Starting SARSA training NOW...")
print("   Hypothesis: On-policy should have LOWER FN rate")
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
    print("✅ SARSA TRAINING COMPLETE!")
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
production_path = "models/production/sarsa_5000ep.pkl"
agent.save(production_path)

if os.path.exists(production_path):
    size_kb = os.path.getsize(production_path) / 1024
    print(f"  ✅ Saved: {production_path} ({size_kb:.2f} KB)")

# Final stats
print("\n" + "=" * 70)
print("📊 SARSA FINAL STATISTICS")
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
print(f"  FN Rate: {safety['fn_rate']:.2%} {'✅ BETTER!' if safety['fn_rate'] < 0.1218 else '⚠️ SIMILAR'}")

# Comparison
print("\n" + "=" * 70)
print("📊 Q-LEARNING vs SARSA COMPARISON")
print("=" * 70)
print("\nQ-Learning Results:")
print("  Avg Reward (100): +0.49")
print("  F1 Score: 70.44%")
print("  FN Rate: 12.18%")
print("\nSARSA Results:")
print(f"  Avg Reward (100): {metrics['best_avg_reward']:.2f}")
print(f"  F1 Score: {safety['f1']:.2%}")
print(f"  FN Rate: {safety['fn_rate']:.2%}")

print("\n" + "=" * 70)
print("🎉 SARSA TRAINING COMPLETED!")
print("=" * 70)
print(f"\nModel: {production_path}")
print(f"Checkpoints: models/checkpoints/")
print(f"MLflow: http://127.0.0.1:5000 (Compare both runs!)")
print("=" * 70)
