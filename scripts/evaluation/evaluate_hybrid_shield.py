"""🛡️ Phase 7: Hybrid Safety Shield Evaluation

Tests the hybrid agent on 1000 episodes to prove FN rate reduction.
Goal: <5% FN rate (vs 12.18% baseline)
"""

import sys
sys.path.insert(0, "C:\\Users\\LENOVO\\Documents\\adaptive-cdss-under-uncertainty")

from src.environment.cdss_env import CDSSEnvironment
from src.agents.q_learning import QLearningAgent
from src.agents.hybrid_safe_agent import HybridSafeAgent

print("=" * 70)
print("🛡️ PHASE 7: HYBRID SAFETY SHIELD EVALUATION")
print("=" * 70)
print("\nGoal: Reduce FN Rate from 12.18% to < 5%")
print("Method: Rule-Based Safety Net + RL Policy")
print("Episodes: 1000 (evaluation)")
print("\n" + "=" * 70)

# Setup environment
print("\n[1/4] Setting up environment...")
env = CDSSEnvironment(knowledge_path="knowledge/", missing_rate=0.4, seed=42)
print("  ✅ Environment ready")

# Load trained Q-Learning agent
print("\n[2/4] Loading trained Q-Learning agent...")
rl_agent = QLearningAgent(n_actions=4, alpha=0.1, gamma=0.95, epsilon=0.2, seed=42)
try:
    rl_agent.load("models/production/q_learning_5000ep.pkl")
    print(f"  ✅ Loaded model (Q-table: {rl_agent.get_q_table_size()} states)")
except:
    print("  ⚠️  Model not found, using untrained agent (for demo)")

# Create Hybrid Safety Shield
print("\n[3/4] Creating Hybrid Safety Shield...")
hybrid_agent = HybridSafeAgent(
    rl_agent=rl_agent,
    knowledge_base=env.kb,
    safety_threshold=8  # Severe risk threshold
)
print("  ✅ Hybrid agent ready")
print("     Safety Rules:")
print("       - Visible risk ≥ 8 → SUGGEST_ALT (mandatory)")
print("       - Visible risk 3-7 + RL wants APPROVE → WARN (upgrade)")
print("       - Visible risk < 3 → Use RL policy")

# Evaluation
print("\n[4/4] Running 1000-episode evaluation...")
print("=" * 70)

# Metrics
false_negatives = 0
false_positives = 0
true_positives = 0
true_negatives = 0
total_reward = 0.0

for ep in range(1000):
    obs = env.reset()
    
    # Hybrid agent decision
    action = hybrid_agent.choose_action(obs, training=False)
    
    # Execute in environment
    next_obs, reward, done, info = env.step(action)
    total_reward += reward
    
    # Safety metrics
    true_risk = info["true_risk"]
    
    if action == 0:  # APPROVE
        if true_risk >= 8:
            false_negatives += 1  # Missed severe risk
        elif true_risk == 0:
            true_negatives += 1  # Correct approval
    elif action in [1, 2]:  # WARN or SUGGEST_ALT
        if true_risk >= 3:
            true_positives += 1  # Correct warning
        else:
            false_positives += 1  # False alarm
    
    # Progress
    if (ep + 1) % 200 == 0:
        print(f"  Episode {ep+1}/1000 - FN so far: {false_negatives}")

print("=" * 70)
print("\n✅ EVALUATION COMPLETE!")
print("=" * 70)

# Calculate metrics
total_decisions = false_negatives + false_positives + true_positives + true_negatives
precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
fn_rate = false_negatives / total_decisions if total_decisions > 0 else 0.0

# Results
print("\n📊 HYBRID SAFETY SHIELD RESULTS")
print("=" * 70)
print(f"\nPerformance:")
print(f"  Episodes: 1000")
print(f"  Total Reward: {total_reward:.2f}")
print(f"  Avg Reward: {total_reward / 1000:.2f}")

print(f"\nSafety Metrics:")
print(f"  False Negatives: {false_negatives}")
print(f"  False Positives: {false_positives}")
print(f"  True Positives: {true_positives}")
print(f"  True Negatives: {true_negatives}")

print(f"\nSafety Scores:")
print(f"  Precision: {precision:.2%}")
print(f"  Recall: {recall:.2%}")
print(f"  F1 Score: {f1:.2%}")
print(f"  False Negative Rate: {fn_rate:.2%} {'✅ SUCCESS!' if fn_rate < 0.05 else '⚠️ NEEDS TUNING' if fn_rate < 0.10 else '❌ FAILED'}")

# Hybrid statistics
hybrid_stats = hybrid_agent.get_statistics()
print(f"\nHybrid Agent Statistics:")
print(f"  Total Decisions: {hybrid_stats['total_decisions']}")
print(f"  Safety Layer Catches: {hybrid_stats['safety_layer_catches']}")
print(f"  Safety Vetoes: {hybrid_stats['safety_vetoes']}")
print(f"  RL Decisions Used: {hybrid_stats['rl_decisions']}")
print(f"  Safety Intervention Rate: {hybrid_stats['safety_intervention_rate']:.2%}")

# Comparison
print("\n" + "=" * 70)
print("📊 BASELINE vs HYBRID COMPARISON")
print("=" * 70)
print("\nQ-Learning (Baseline):")
print("  FN Rate: 12.18%")
print("  F1 Score: 70.44%")
print("  Safety: ❌ Unacceptable for production")

print(f"\nHybrid Safety Shield:")
print(f"  FN Rate: {fn_rate:.2%}")
print(f"  F1 Score: {f1:.2%}")
print(f"  Safety: {'✅ Production-Ready!' if fn_rate < 0.05 else '⚠️ Acceptable with oversight' if fn_rate < 0.10 else '❌ Needs improvement'}")

print(f"\nImprovement:")
print(f"  FN Rate Reduction: {12.18 - fn_rate*100:.2f} percentage points")
print(f"  Improvement: {((12.18 - fn_rate*100) / 12.18 * 100):.1f}% better")

print("\n" + "=" * 70)
print("🎉 HYBRID SAFETY SHIELD EVALUATION COMPLETE!")
print("=" * 70)
print(f"\nFinal Verdict: {'🏆 PRODUCTION-READY' if fn_rate < 0.05 else '⚠️ NEEDS TUNING' if fn_rate < 0.10 else '❌ NOT SAFE'}")
print("=" * 70)
