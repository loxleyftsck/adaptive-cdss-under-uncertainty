"""Reward function for safety-centered prescription decisions.

This module implements the reward function R(s,a) that balances:
- Severe interaction detection (high rewards for correct warnings)
- Alert fatigue mitigation (penalties for false alarms)
- Information seeking (rewards for requesting data when uncertain)
"""

from typing import Tuple
from src.environment.patient_generator import Patient
from src.environment.observation_model import Observation
from src.knowledge.knowledge_base import KnowledgeBase


# Action constants
ACTION_APPROVE = 0
ACTION_WARN = 1
ACTION_SUGGEST_ALT = 2
ACTION_REQUEST_DATA = 3


class RewardFunction:
    """Calculates safety-centered rewards for prescription decisions.
    
    Reward design priorities:
    1. Severe penalties for missing high-risk interactions (false negatives)
    2. Moderate rewards for correct risk flagging
    3. Small penalty for false alarms (alert fatigue)
    4. Incentive for data requests when uncertainty is high
    
    Args:
        knowledge_base: Pharmacological knowledge for risk calculation
    """
    
    def __init__(self, knowledge_base: KnowledgeBase):
        """Initialize reward function with knowledge base.
        
        Args:
            knowledge_base: KnowledgeBase instance for DDI/contraindication lookup
        """
        self.kb = knowledge_base
    
    def compute(
        self,
        patient: Patient,
        action: int,
        observation: Observation
    ) -> float:
        """Calculate reward for action given true patient state.
        
        Reward logic by action:
        - APPROVE: +2 if safe, large negative if risky
        - WARN: +3 if moderate/high risk, -1 if false alarm
        - SUGGEST_ALT: +4 if severe risk, -1 if unnecessary
        - REQUEST_DATA: +2 if low completeness, -0.5 if already complete
        
        Args:
            patient: True patient state (hidden from agent)
            action: Action taken (0-3)
            observation: Partial observation (what agent saw)
            
        Returns:
            Scalar reward value
        """
        # Calculate true total risk
        ddi_risk, contra_risk = self._compute_true_risk(patient)
        total_risk = ddi_risk + contra_risk
        
        if action == ACTION_APPROVE:
            return self._reward_approve(total_risk)
        elif action == ACTION_WARN:
            return self._reward_warn(total_risk)
        elif action == ACTION_SUGGEST_ALT:
            return self._reward_suggest_alt(total_risk)
        elif action == ACTION_REQUEST_DATA:
            return self._reward_request_data(observation.data_completeness)
        else:
            raise ValueError(f"Invalid action: {action}")
    
    def _compute_true_risk(self, patient: Patient) -> Tuple[int, int]:
        """Calculate true DDI and contraindication risk.
        
        Args:
            patient: True patient state
            
        Returns:
            Tuple of (ddi_risk, contraindication_risk)
        """
        # Map hidden states to condition names
        conditions = patient.conditions.copy()
        if patient.renal_function == "impaired":
            conditions.append("renal_impairment")
        if patient.hepatic_function == "impaired":
            conditions.append("hepatic_impairment")
        
        return self.kb.get_total_risk(patient.medications, conditions)
    
    def _reward_approve(self, total_risk: int) -> float:
        """Reward for APPROVE action.
        
        Safe approval: +2
        Risky approval: -2 to -12 (severe penalty for missed risks)
        
        Args:
            total_risk: Total risk from DDIs and contraindications
            
        Returns:
            Reward value
        """
        if total_risk == 0:
            return 2.0  # Safe prescription approved
        else:
            # Penalty proportional to risk (max -12 for very high risk)
            return -2.0 - min(total_risk, 10)
    
    def _reward_warn(self, total_risk: int) -> float:
        """Reward for WARN action.
        
        Correct warning (risk ≥ 5): +3
        Minor risk flagged (1-4): +1
        False alarm (risk = 0): -1
        
        Args:
            total_risk: Total risk from DDIs and contraindications
            
        Returns:
            Reward value
        """
        if total_risk >= 5:
            return 3.0  # Correctly flagged moderate/high risk
        elif total_risk > 0:
            return 1.0  # Flagged minor risk (acceptable)
        else:
            return -1.0  # False alarm (alert fatigue)
    
    def _reward_suggest_alt(self, total_risk: int) -> float:
        """Reward for SUGGEST_ALT action.
        
        Severe risk prevented (≥8): +4
        Moderate risk (5-7): +2
        Unnecessary (risk <5): -1
        
        Args:
            total_risk: Total risk from DDIs and contraindications
            
        Returns:
            Reward value
        """
        if total_risk >= 8:
            return 4.0  # Excellent - prevented severe interaction
        elif total_risk >= 5:
            return 2.0  # Good - suggested alternative for moderate risk
        else:
            return -1.0  # Unnecessary (conservative but impractical)
    
    def _reward_request_data(self, completeness: float) -> float:
        """Reward for REQUEST_DATA action.
        
        High uncertainty (completeness <0.7): +2 (good decision to wait)
        Moderate uncertainty (0.7-0.85): +0.5 (acceptable)
        Low uncertainty (>0.85): -0.5 (unnecessary delay)
        
        Args:
            completeness: Data completeness (0-1)
            
        Returns:
            Reward value
        """
        if completeness < 0.7:
            return 2.0  # Wise to request more data
        elif completeness < 0.85:
            return 0.5  # Acceptable caution
        else:
            return -0.5  # Unnecessary delay
    
    def get_action_name(self, action: int) -> str:
        """Get human-readable action name.
        
        Args:
            action: Action index (0-3)
            
        Returns:
            Action name string
        """
        names = {
            ACTION_APPROVE: "APPROVE",
            ACTION_WARN: "WARN",
            ACTION_SUGGEST_ALT: "SUGGEST_ALT",
            ACTION_REQUEST_DATA: "REQUEST_DATA",
        }
        return names.get(action, "UNKNOWN")
