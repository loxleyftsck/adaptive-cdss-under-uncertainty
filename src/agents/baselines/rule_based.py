"""Rule-based CDSS baseline using pharmacological knowledge."""

from src.environment.observation_model import Observation
from src.knowledge.knowledge_base import KnowledgeBase


# Action constants
ACTION_APPROVE = 0
ACTION_WARN = 1
ACTION_SUGGEST_ALT = 2
ACTION_REQUEST_DATA = 3


class RuleBasedCDSS:
    """Rule-based clinical decision support system.
    
    Implements deterministic rules based on visible risk:
    1. If data completeness < 60%: REQUEST_DATA
    2. If DDI/contraindication risk ≥ 8: SUGGEST_ALT
    3. If risk 3-7: WARN
    4. If risk 1-2: APPROVE (with caution)
    5. If risk 0: APPROVE
    
    Represents current state-of-the-art CDSS without learning.
    
    Args:
        knowledge_base: Pharmacological knowledge for risk assessment
    """
    
    def __init__(self, knowledge_base: KnowledgeBase):
        """Initialize rule-based system with knowledge base.
        
        Args:
            knowledge_base: KnowledgeBase for DDI/contraindication lookup
        """
        self.kb = knowledge_base
    
    def choose_action(self, observation: Observation, training: bool = True) -> int:
        """Apply deterministic rules to choose action.
        
        Args:
            observation: Partial patient observation
            training: Training flag (ignored, rules are deterministic)
            
        Returns:
            Action index based on rules
        """
        # Rule 1: Request more data if completeness low
        if observation.data_completeness < 0.6:
            return ACTION_REQUEST_DATA
        
        # Calculate visible risk (based on partial information)
        visible_risk = self._calculate_visible_risk(observation)
        
        # Rule 2: Suggest alternative if severe risk
        if visible_risk >= 8:
            return ACTION_SUGGEST_ALT
        
        # Rule 3: Warn if moderate risk
        if visible_risk >= 3:
            return ACTION_WARN
        
        # Rule 4 & 5: Approve (whether minor risk or no risk)
        return ACTION_APPROVE
    
    def _calculate_visible_risk(self, observation: Observation) -> int:
        """Calculate risk based on visible information only.
        
        Args:
            observation: Partial observation
            
        Returns:
            Total visible risk (DDI + contraindications)
        """
        # Get DDI risk from visible medications
        ddi_risk, _ = self.kb.get_total_risk(observation.medications, [])
        
        # Get contraindication risk from visible conditions
        contra_risk = 0
        for drug in observation.medications:
            for condition in observation.visible_conditions:
                contra = self.kb.check_contraindication(drug, condition)
                if contra:
                    contra_risk += abs(contra["penalty"])
        
        return ddi_risk + contra_risk
    
    def update(self, *args, **kwargs) -> float:
        """No-op update (rule-based system doesn't learn).
        
        Returns:
            0.0 (no learning)
        """
        return 0.0
    
    def __repr__(self) -> str:
        return "RuleBasedCDSS(deterministic rules based on visible risk)"
