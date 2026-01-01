"""Observation model for simulating partial observability.

This module simulates incomplete EHR data by stochastically masking
patient information, creating the observation space (Ω) in our POMDP.
"""

import random
from dataclasses import dataclass, field
from typing import List, Optional
from src.environment.patient_generator import Patient


@dataclass
class Observation:
    """Partial observation of patient state (what the agent sees).
    
    This is the observation (o ∈ Ω) in POMDP formulation - incomplete
    view of true patient state due to missing EHR data.
    
    Attributes:
        age: Patient age (always observable)
        visible_conditions: Subset of true conditions (some may be hidden)
        medications: Current medications (always observable)
        lab_results: Lab results if available, else None
        data_completeness: Fraction of available information (0-1)
    """
    age: int
    visible_conditions: List[str] = field(default_factory=list)
    medications: List[str] = field(default_factory=list)
    lab_results: Optional[dict] = None
    data_completeness: float = 1.0
    
    def __repr__(self) -> str:
        return (
            f"Observation(age={self.age}, "
            f"visible_conditions={len(self.visible_conditions)}, "
            f"meds={len(self.medications)}, "
            f"completeness={self.data_completeness:.2f})"
        )


class ObservationModel:
    """Simulates partial observability via stochastic data masking.
    
    Models the observation function P(o|s) in POMDP, where some patient
    information is randomly hidden to simulate incomplete EHR data.
    
    Args:
        missing_rate: Probability that a condition is hidden (0-1)
        lab_availability: Probability lab results are available
        seed: Random seed for reproducibility
    """
    
    def __init__(
        self,
        missing_rate: float = 0.4,
        lab_availability: float = 0.6,
        seed: Optional[int] = None,
    ):
        """Initialize observation model with missing data parameters.
        
        Args:
            missing_rate: Fraction of conditions to hide (default: 40%)
            lab_availability: Probability lab results available (default: 60%)
            seed: Optional random seed
            
        Raises:
            ValueError: If parameters out of valid range [0,1]
        """
        if not 0 <= missing_rate <= 1:
            raise ValueError("missing_rate must be in [0, 1]")
        if not 0 <= lab_availability <= 1:
            raise ValueError("lab_availability must be in [0, 1]")
        
        self.missing_rate = missing_rate
        self.lab_availability = lab_availability
        self.rng = random.Random(seed)
    
    def observe(self, patient: Patient) -> Observation:
        """Generate partial observation from true patient state.
        
        Applies stochastic masking to simulate incomplete EHR:
        - Age: Always visible
        - Medications: Always visible  
        - Conditions: Some hidden based on missing_rate
        - Lab results: Available based on lab_availability
        
        Args:
            patient: True patient state
            
        Returns:
            Observation with partial information
        """
        # Age and medications always observable
        age = patient.age
        medications = patient.medications.copy()
        
        # Stochastically mask conditions
        visible_conditions = [
            cond for cond in patient.conditions
            if self.rng.random() > self.missing_rate
        ]
        
        # Lab results availability
        lab_results = None
        if patient.lab_results_available and self.rng.random() < self.lab_availability:
            lab_results = {
                "renal_function": patient.renal_function,
                "hepatic_function": patient.hepatic_function,
            }
        
        # Calculate data completeness
        total_info = len(patient.conditions) + (2 if patient.lab_results_available else 0)
        observed_info = len(visible_conditions) + (2 if lab_results else 0)
        completeness = observed_info / total_info if total_info > 0 else 1.0
        
        return Observation(
            age=age,
            visible_conditions=visible_conditions,
            medications=medications,
            lab_results=lab_results,
            data_completeness=completeness,
        )
    
    def set_missing_rate(self, rate: float) -> None:
        """Update missing data rate (useful for robustness testing).
        
        Args:
            rate: New missing rate (0-1)
            
        Raises:
            ValueError: If rate out of range
        """
        if not 0 <= rate <= 1:
            raise ValueError("missing_rate must be in [0, 1]")
        self.missing_rate = rate
