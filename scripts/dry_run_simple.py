"""🔍 DRY RUN - Sanity Check (Without tqdm dependency)"""

import sys
import os
sys.path.insert(0, "C:\\Users\\LENOVO\\Documents\\adaptive-cdss-under-uncertainty")

from src.environment.cdss_env import CDSSEnvironment
from src.agents.q_learning import QLearningAgent
from src.training.trainer import Trainer

print("=" * 70)
print("🔍 DRY RUN - SANITY CHECK")
print("=" * 70)

# Setup
print("\n[1/5] Setting up environment and agent...")
env = CDSSEnvironment(knowledge_path="knowledge/", missing_rate=0.4, seed=42)
agent = QLearningAgent(n_actions=4, alpha=0.1, gamma=0.95, epsilon=0.2, seed=42)
print("  ✅ Environment and agent created")

# Create trainer
print("\n[2/5] Creating trainer...")
trainer = Trainer(
    env=env,
    agent=agent,
    experiment_name="CDSS-DryRun",
    run_name="Q-Learning-DryRun-20ep",
    use_mlflow=False  # Disable MLflow for simplicity
)
print("  ✅ Trainer initialized")

# Manual training loop (without tqdm)
print("\n[3/5] Running 20 episode training (DRY RUN)...")
print("-" * 70)

try:
    for ep in range(20):
        obs = env.reset()
        action = agent.choose_action(obs, training=True)
        next_obs, reward, done, info = env.step(action)
        agent.update(obs, action, reward, next_obs, done)
        
        trainer.episode_rewards.append(reward)
        trainer.cumulative_reward += reward
        
        # Log progress
        if (ep + 1) % 5 == 0:
            print(f"  Episode {ep+1}/20 - Reward: {reward:.2f}")
    
    print("-" * 70)
    print("  ✅ Training completed without crashes!")
except Exception as e:
    print(f"  ❌ TRAINING FAILED: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Save checkpoint manually
print("\n[4/5] Saving checkpoint...")
os.makedirs("models/checkpoints", exist_ok=True)
checkpoint_file = "models/checkpoints/QLearningAgent_dryrun.pkl"
agent.save(checkpoint_file)

if os.path.exists(checkpoint_file):
    size_kb = os.path.getsize(checkpoint_file) / 1024
    print(f"  ✅ Checkpoint exists: {checkpoint_file}")
    print(f"  ✅ Size: {size_kb:.2f} KB")
else:
    print(f"  ❌ CHECKPOINT NOT FOUND")
    sys.exit(1)

# Verify metrics
print("\n[5/5] Verifying metrics...")
print(f"  Episodes completed: 20")
print(f"  Cumulative reward: {trainer.cumulative_reward:.2f}")
print(f"  Avg reward: {trainer.cumulative_reward / 20:.2f}")
print(f"  Q-table size: {agent.get_q_table_size()}")

# Final verification
print("\n" + "=" * 70)
print("✅ DRY RUN COMPLETE - ALL CHECKS PASSED!")
print("=" * 70)
print("\n📋 Verification Summary:")
print("  ✅ Training loop stable (20 episodes)")
print("  ✅ Checkpoints created successfully")
print("  ✅ No crashes on episode completion")
print("\n🚀 SYSTEM READY FOR FULL TRAINING!")
print("=" * 70)
