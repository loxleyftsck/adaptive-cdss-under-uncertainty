"""
Phase 8: Q-Learning Training with 50-Drug Knowledge Base
Automated training script - no user interaction required
"""

import mlflow
import random
from src.environment.cdss_env import CDSSEnvironment
from src.agents.q_learning import QLearningAgent
from src.training.trainer import Trainer

def main():
    print("=" * 70)
    print("PHASE 8: Q-LEARNING TRAINING (50-DRUG KB)")
    print("=" * 70)
    
    # Configuration
    config = {
        "agent_type": "Q-Learning",
        "episodes": 5000,
        "learning_rate": 0.1,
        "discount_factor": 0.95,
        "epsilon": 0.2,
        "missing_rate": 0.4,
        "seed": 42,
        "knowledge_base": "50-drug expanded",
        "phase": "Phase 8"
    }
    
    print(f"\n📊 Configuration:")
    for k, v in config.items():
        print(f"  {k}: {v}")
    
    # Set seeds
    random.seed(config["seed"])
    
    # Initialize environment with 50-drug KB
    print(f"\n🔧 Initializing environment...")
    env = CDSSEnvironment(
        missing_rate=config["missing_rate"],
        seed=config["seed"]
    )
    print(f"✅ Environment ready: {len(env.kb.drugs)} drugs, {len(env.kb.interactions)} DDIs")
    
    # Initialize Q-Learning agent
    print(f"\n🤖 Initializing Q-Learning agent...")
    agent = QLearningAgent(
        action_space=env.action_space,
        learning_rate=config["learning_rate"],
        discount_factor=config["discount_factor"],
        epsilon=config["epsilon"]
    )
    print(f"✅ Agent initialized")
    
    # Initialize trainer
    print(f"\n🏋️ Initializing trainer...")

    trainer = Trainer(
        agent=agent,
        environment=env,
        mlflow_experiment="Phase8_QLearning_50Drug"
    )
    print(f"✅ Trainer ready")
    
    # Start MLflow run
    with mlflow.start_run(run_name="Q-Learning_Phase8_50Drug_5000ep"):
        # Log configuration
        mlflow.log_params(config)
        
        # Train
        print(f"\n🚀 Starting training: {config['episodes']} episodes...")
        print(f"MLflow tracking enabled: {mlflow.get_tracking_uri()}")
        print(f"Check progress at: http://127.0.0.1:5000")
        print("-" * 70)
        
        trainer.train(
            num_episodes=config["episodes"],
            checkpoint_interval=500,
            verbose=True
        )
        
        print("\n" + "=" * 70)
        print("✅ TRAINING COMPLETE!")
        print("=" * 70)
        
        # Final stats
        print(f"\n📊 Final Statistics:")
        print(f"  Total episodes: {config['episodes']}")
        print(f"  Q-table size: {len(agent.q_table)} states")
        print(f"  Epsilon (final): {agent.epsilon}")
        
        # Log final Q-table size
        mlflow.log_metric("final_q_table_size", len(agent.q_table))
        
        print(f"\n✅ Phase 8 Q-Learning training complete!")
        print(f"Check MLflow UI for detailed metrics and learning curves")

if __name__ == "__main__":
    main()
