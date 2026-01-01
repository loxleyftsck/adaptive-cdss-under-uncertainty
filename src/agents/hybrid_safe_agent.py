"""🛡️ Hybrid Safety Shield - Rule-Based Safety Net + RL Decision Making

Combines deterministic clinical rules with learned RL policies for
production-safe medical CDSS deployment.
"""

from typing import Optional
from src.environment.observation_model import Observation
from src.knowledge.knowledge_base import KnowledgeBase
from src.agents.q_learning import QLearningAgent


# Action constants
ACTION_APPROVE = 0
ACTION_WARN = 1
ACTION_SUGGEST_ALT = 2
ACTION_REQUEST_DATA = 3


class HybridSafeAgent:
    """Hybrid agent combining rule-based safety checks with RL policy.
    
    Architecture:
    1. Safety Check Layer (Clinical Rules) - FIRST
       - Checks for known severe DDIs
       - Checks for critical contraindications
       - VETO power over RL decisions
       
    2. RL Decision Layer (Q-Learning) - SECOND
       - Handles edge cases and uncertainty
       - Adapts to partial information
       - Provides nuanced decisions
       
    3. Override Logic - FINAL
       - If RL suggests dangerous action, override with safe default
       - Never allow high-risk approvals
       
    Args:
        rl_agent: Trained RL agent (Q-Learning or SARSA)
        knowledge_base: Clinical knowledge for safety rules
        safety_threshold: Risk threshold for veto (default: 8 = severe)
    """
    
    def __init__(
        self,
        rl_agent: QLearningAgent,
        knowledge_base: KnowledgeBase,
        safety_threshold: int = 8,
    ):
        """Initialize hybrid agent.
        
        Args:
            rl_agent: Trained RL agent
            knowledge_base: KB for safety checks
            safety_threshold: Minimum risk to trigger safety veto
        """
        self.rl_agent = rl_agent
        self.kb = knowledge_base
        self.safety_threshold = safety_threshold
        
        # Statistics
        self.total_decisions = 0
        self.safety_vetoes = 0
        self.rl_decisions = 0
        self.safety_layer_catches = 0
    
    def choose_action(
        self, observation: Observation, training: bool = False
    ) -> int:
        """Choose action with safety-first hybrid logic.
        
        Process:
        1. Calculate visible risk from observation
        2. Check safety rules (VETO dangerous actions)
        3. If safe, use RL policy
        4. If RL suggests dangerous action, override
        
        Args:
            observation: Partial patient observation
            training: Whether in training mode (ignored - always safe)
            
        Returns:
            Action index (0-3), guaranteed safe
        """
        self.total_decisions += 1
        
        # LAYER 1: SAFETY CHECK (Rule-Based)
        visible_risk = self._calculate_visible_risk(observation)
        
        # CRITICAL SAFETY RULE: If severe DDI visible, MUST warn/suggest alternative
        if visible_risk >= self.safety_threshold:
            self.safety_layer_catches += 1
            # Never approve high-risk prescriptions
            return ACTION_SUGGEST_ALT  # Always suggest alternative for severe risk
        
        # LAYER 2: RL DECISION (if safety check passed)
        rl_action = self.rl_agent.choose_action(observation, training=False)
        
        # LAYER 3: OVERRIDE LOGIC (veto dangerous RL decisions)
        # If RL wants to approve, but moderate risk exists, upgrade to warn
        if rl_action == ACTION_APPROVE and visible_risk >= 3:
            self.safety_vetoes += 1
            return ACTION_WARN  # Safe override: at least warn
        
        # RL decision is safe, use it
        self.rl_decisions += 1
        return rl_action
    
    def _calculate_visible_risk(self, observation: Observation) -> int:
        """Calculate risk from visible information only.
        
        This is the safety net - catches all KNOWN risks.
        
        Args:
            observation: Partial observation
            
        Returns:
            Total visible risk (DDI + contraindications)
        """
        # Check DDIs from visible medications
        ddi_risk, _ = self.kb.get_total_risk(observation.medications, [])
        
        # Check contraindications from visible conditions
        contra_risk = 0
        for drug in observation.medications:
            for condition in observation.visible_conditions:
                contra = self.kb.check_contraindication(drug, condition)
                if contra:
                    contra_risk += abs(contra["penalty"])
        
        return ddi_risk + contra_risk
    
    def update(self, *args, **kwargs):
        """No-op update (hybrid doesn't learn - wraps trained agent)."""
        return 0.0
    
    def get_statistics(self):
        """Get safety layer statistics.
        
        Returns:
            Dictionary with hybrid agent stats
        """
        return {
            "total_decisions": self.total_decisions,
            "safety_layer_catches": self.safety_layer_catches,
            "safety_vetoes": self.safety_vetoes,
            "rl_decisions": self.rl_decisions,
            "safety_intervention_rate": (
                (self.safety_layer_catches + self.safety_vetoes) / self.total_decisions
                if self.total_decisions > 0 else 0.0
            )
        }
    
    def __repr__(self) -> str:
        """String representation."""
        return (
            f"HybridSafeAgent(RL={self.rl_agent.__class__.__name__}, "
            f"threshold={self.safety_threshold}, "
            f"vetoes={self.safety_vetoes}/{self.total_decisions})"
        )
