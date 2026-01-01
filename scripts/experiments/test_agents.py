"""Quick smoke test for RL agents."""

import sys
sys.path.insert(0, "C:\\Users\\LENOVO\\Documents\\adaptive-cdss-under-uncertainty")

from src.environment.cdss_env import CDSSEnvironment
from src.agents.q_learning import QLearningAgent
from src.agents.sarsa import SARSAAgent
from src.agents.baselines.random_policy import RandomPolicy
from src.agents.baselines.rule_based import RuleBasedCDSS
from src.agents.baselines.oracle import PerfectOracle

print("=" * 60)
print("RL AGENTS SMOKE TEST")
print("=" * 60)

# Initialize environment
env = CDSSEnvironment(knowledge_path="knowledge/", missing_rate=0.4, seed=42)
print("\n✅ Environment initialized")

# Test all agents
agents = {
    "Q-Learning": QLearningAgent(seed=42),
    "SARSA": SARSAAgent(seed=42),
    "Random": RandomPolicy(seed=42),
    "Rule-Based": RuleBasedCDSS(env.kb),
    "Oracle": PerfectOracle(env.kb),
}

print(f"\n✅ {len(agents)} agents created")

# Run quick episode with each agent
print("\n" + "=" * 60)
print("TESTING EACH AGENT (5 episodes each)")
print("=" * 60)

for agent_name, agent in agents.items():
    print(f"\n{agent_name}:")
    total_reward = 0
    
    for ep in range(5):
        obs = env.reset()
        
        # Oracle needs true patient state
        if agent_name == "Oracle":
            agent.set_true_patient(env.current_patient)
        
        action = agent.choose_action(obs, training=True)
        next_obs, reward, done, info = env.step(action)
        
        # Update learning agents
        if agent_name in ["Q-Learning", "SARSA"]:
            agent.update(obs, action, reward, next_obs, done=done)
        
        total_reward += reward
    
    avg_reward = total_reward / 5
    print(f"  Avg Reward: {avg_reward:.2f}")
    
    if hasattr(agent, 'get_q_table_size'):
        print(f"  Q-table size: {agent.get_q_table_size()}")

print("\n" + "=" * 60)
print("✅ ALL AGENTS FUNCTIONAL!")
print("=" * 60)
print("\nAgents are ready for training!")
