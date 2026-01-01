"""CDSS Environment - Gym-compatible RL environment for prescription safety.

This module provides the main environment class that integrates all components:
patient generation, observation modeling, and reward calculation.
"""

from typing import Dict, Optional, Tuple, Any
from src.environment.patient_generator import Patient, PatientGenerator
from src.environment.observation_model import Observation, ObservationModel
from src.environment.reward import RewardFunction
from src.knowledge.knowledge_base import KnowledgeBase


class CDSSEnvironment:
    """Reinforcement Learning environment for prescription decision-making.
    
    Implements a Gym-like interface for RL agents to learn safe prescription
    decisions under partial observability (POMDP). Each episode represents
    one prescription decision.
    
    Action Space: {0: APPROVE, 1: WARN, 2: SUGGEST_ALT, 3: REQUEST_DATA}
    
    Observation Space: Partial patient information (age, visible conditions,
                       medications, optional lab results, data completeness)
    
    Args:
        knowledge_path: Path to knowledge base directory
        missing_rate: Probability of hiding condition (default: 0.4 = 40%)
        seed: Random seed for reproducibility
    """
    
    def __init__(
        self,
        knowledge_path: str = "knowledge/",
        missing_rate: float = 0.4,
        seed: Optional[int] = None,
    ):
        """Initialize CDSS environment.
        
        Args:
            knowledge_path: Path to directory with knowledge JSON files
            missing_rate: Fraction of conditions to hide (0-1)
            seed: Optional random seed
        """
        # Load knowledge base
        self.kb = KnowledgeBase(knowledge_path)
        
        # Initialize components
        self.patient_gen = PatientGenerator(
            available_drugs=self.kb.get_all_drugs(),
            available_conditions=[
                "diabetes",
                "hypertension",
                "heart_failure",
                "atrial_fibrillation",
            ],
            seed=seed,
        )
        
        self.obs_model = ObservationModel(
            missing_rate=missing_rate,
            lab_availability=0.6,
            seed=seed,
        )
        
        self.reward_fn = RewardFunction(self.kb)
        
        # State tracking
        self.current_patient: Optional[Patient] = None
        self.current_observation: Optional[Observation] = None
        self.episode_count = 0
    
    def reset(self) -> Observation:
        """Start a new episode with a fresh patient.
        
        Returns:
            Initial observation (partial patient information)
        """
        # Generate new patient
        self.current_patient = self.patient_gen.generate()
        
        # Create observation (partial view)
        self.current_observation = self.obs_model.observe(self.current_patient)
        
        self.episode_count += 1
        
        return self.current_observation
    
    def step(self, action: int) -> Tuple[None, float, bool, Dict[str, Any]]:
        """Execute action and return reward.
        
        Args:
            action: Action index (0-3)
            
        Returns:
            Tuple of (next_observation, reward, done, info)
            - next_observation: None (episodic, ends after one decision)
            - reward: Scalar reward value
            - done: Always True (single-step episodes)
            - info: Dictionary with episode metadata
            
        Raises:
            RuntimeError: If reset() not called before step()
        """
        if self.current_patient is None:
            raise RuntimeError("Call reset() before step()")
        
        # Calculate reward based on true patient state
        reward = self.reward_fn.compute(
            self.current_patient,
            action,
            self.current_observation
        )
        
        # Calculate true risk for logging
        ddi_risk, contra_risk = self.reward_fn._compute_true_risk(
            self.current_patient
        )
        
        # Episode ends after one decision (episodic formulation)
        done = True
        
        # Info dictionary for logging/debugging
        info = {
            "patient_id": self.current_patient.patient_id,
            "true_risk": ddi_risk + contra_risk,
            "ddi_risk": ddi_risk,
            "contra_risk": contra_risk,
            "action": action,
            "action_name": self.reward_fn.get_action_name(action),
            "data_completeness": self.current_observation.data_completeness,
            "n_medications": len(self.current_patient.medications),
            "n_conditions": len(self.current_patient.conditions),
            "n_visible_conditions": len(self.current_observation.visible_conditions),
        }
        
        return None, reward, done, info
    
    def render(self, mode: str = "human") -> Optional[str]:
        """Render current environment state.
        
        Args:
            mode: Rendering mode ("human" for text, "ansi" for string)
            
        Returns:
            String representation if mode="ansi", None otherwise
        """
        if self.current_patient is None:
            return "Environment not initialized. Call reset()."
        
        output = []
        output.append("=" * 60)
        output.append(f"Episode {self.episode_count}")
        output.append("=" * 60)
        output.append(f"\nPatient ID: {self.current_patient.patient_id}")
        output.append(f"Age: {self.current_patient.age}")
        output.append(f"\nMedications: {', '.join(self.current_patient.medications)}")
        output.append(f"True Conditions: {', '.join(self.current_patient.conditions)}")
        output.append(f"Visible Conditions: {', '.join(self.current_observation.visible_conditions)}")
        output.append(f"\nHidden States:")
        output.append(f"  Renal function: {self.current_patient.renal_function}")
        output.append(f"  Hepatic function: {self.current_patient.hepatic_function}")
        output.append(f"\nData Completeness: {self.current_observation.data_completeness:.2%}")
        
        # Calculate true risk
        ddi_risk, contra_risk = self.reward_fn._compute_true_risk(
            self.current_patient
        )
        output.append(f"\nTrue Risk: {ddi_risk + contra_risk} (DDI: {ddi_risk}, Contra: {contra_risk})")
        output.append("=" * 60)
        
        text = "\n".join(output)
        
        if mode == "human":
            print(text)
            return None
        else:  # ansi or other
            return text
    
    def set_missing_rate(self, rate: float) -> None:
        """Update missing data rate (for robustness testing).
        
        Args:
            rate: New missing rate (0-1)
        """
        self.obs_model.set_missing_rate(rate)
    
    def get_action_space_size(self) -> int:
        """Get number of possible actions.
        
        Returns:
            Action space size (4)
        """
        return 4
    
    def __repr__(self) -> str:
        """String representation of environment."""
        return (
            f"CDSSEnvironment("
            f"drugs={len(self.kb.drugs)}, "
            f"missing_rate={self.obs_model.missing_rate:.2f}, "
            f"episodes={self.episode_count})"
        )
