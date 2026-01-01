"""Perfect oracle baseline - upper bound on performance."""

from src.environment.observation_model import Observation
from src.environment.patient_generator import Patient
from src.knowledge.knowledge_base import KnowledgeBase


# Action constants
ACTION_APPROVE = 0
ACTION_WARN = 1
ACTION_SUGGEST_ALT = 2
ACTION_REQUEST_DATA = 3


class PerfectOracle:
    """Oracle policy with access to true patient state (cheating!).
    
    Provides theoretical upper bound on performance by making
    optimal decisions based on complete information.
    
    This is NOT realistic (agent has access to hidden state) but
    serves as performance ceiling for comparison.
    
    Args:
        knowledge_base: Pharmacological knowledge
    """
    
    def __init__(self, knowledge_base: KnowledgeBase):
        """Initialize oracle with knowledge base.
        
        Args:
            knowledge_base: KnowledgeBase for risk calculation
        """
        self.kb = knowledge_base
        self.true_patient: Patient = None  # Will be set externally
    
    def set_true_patient(self, patient: Patient) -> None:
        """Set true patient state (called by environment).
        
        Args:
            patient: True patient state with complete information
        """
        self.true_patient = patient
    
    def choose_action(self, observation: Observation, training: bool = True) -> int:
        """Choose optimal action based on TRUE patient state.
        
        Args:
            observation: Partial observation (ignored - oracle cheats!)
            training: Training flag (ignored)
            
        Returns:
            Optimal action based on complete information
        """
        if self.true_patient is None:
            raise RuntimeError("Oracle requires true patient state via set_true_patient()")
        
        # Calculate TRUE risk (including hidden states)
        true_risk = self._calculate_true_risk()
        
        # Optimal decision rules based on complete information
        if true_risk >= 8:
            return ACTION_SUGGEST_ALT  # Severe risk
        elif true_risk >= 3:
            return ACTION_WARN  # Moderate risk
        else:
            return ACTION_APPROVE  # Safe or minor risk
    
    def _calculate_true_risk(self) -> int:
        """Calculate true risk using complete patient information.
        
        Returns:
            Total true risk (DDI + contraindications)
        """
        # Include hidden physiological states as conditions
        conditions = self.true_patient.conditions.copy()
        if self.true_patient.renal_function == "impaired":
            conditions.append("renal_impairment")
        if self.true_patient.hepatic_function == "impaired":
            conditions.append("hepatic_impairment")
        
        ddi_risk, contra_risk = self.kb.get_total_risk(
            self.true_patient.medications, conditions
        )
        
        return ddi_risk + contra_risk
    
    def update(self, *args, **kwargs) -> float:
        """No-op update (oracle doesn't learn).
        
        Returns:
            0.0
        """
        return 0.0
    
    def __repr__(self) -> str:
        return "PerfectOracle(has access to true patient state)"
