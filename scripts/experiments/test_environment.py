"""Quick smoke test for environment components."""

import sys
sys.path.insert(0, "C:\\Users\\LENOVO\\Documents\\adaptive-cdss-under-uncertainty")

from src.environment.cdss_env import CDSSEnvironment

print("=" * 60)
print("CDSS ENVIRONMENT SMOKE TEST")
print("=" * 60)

# Initialize environment
print("\n1. Initializing environment...")
env = CDSSEnvironment(knowledge_path="knowledge/", missing_rate=0.4, seed=42)
print(f"✅ Environment created: {env}")

# Reset and get first observation
print("\n2. Resetting environment...")
obs = env.reset()
print(f"✅ Observation received: {obs}")

# Render current state
print("\n3. Rendering environment state...")
env.render()

# Test all 4 actions
print("\n4. Testing all actions...")
actions = {
    0: "APPROVE",
    1: "WARN",
    2: "SUGGEST_ALT",
    3: "REQUEST_DATA"
}

for action_id, action_name in actions.items():
    # Reset for each action test
    env.reset()
    
    # Take action
    next_obs, reward, done, info = env.step(action_id)
    
    print(f"\n   Action: {action_name}")
    print(f"   Reward: {reward:.2f}")
    print(f"   True Risk: {info['true_risk']}")
    print(f"   Done: {done}")

print("\n" + "=" * 60)
print("✅ ALL TESTS PASSED!")
print("=" * 60)
print("\nEnvironment is functional and ready for agent training!")
