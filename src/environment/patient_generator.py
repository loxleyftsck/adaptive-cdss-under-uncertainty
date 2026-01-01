"""Patient generation module for synthetic medical data.

This module generates realistic synthetic patients with:
- Demographics (age)
- Medical conditions
- Current medications
- Hidden physiological states (renal/hepatic function)
"""

import random
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class Patient:
    """Represents a synthetic patient with complete medical information.
    
    This represents the TRUE state in our POMDP formulation - complete
    clinical reality that is partially observable to the agent.
    
    Attributes:
        patient_id: Unique identifier
        age: Patient age in years (18-85)
        conditions: List of medical conditions
        medications: List of current medications (drug IDs)
        renal_function: Renal function status
        hepatic_function: Hepatic function status
        lab_results_available: Whether lab results are available
    """
    patient_id: int
    age: int
    conditions: List[str] = field(default_factory=list)
    medications: List[str] = field(default_factory=list)
    renal_function: str = "normal"  # "normal" or "impaired"
    hepatic_function: str = "normal"  # "normal" or "impaired"
    lab_results_available: bool = False
    
    def __repr__(self) -> str:
        return (
            f"Patient(id={self.patient_id}, age={self.age}, "
            f"conditions={len(self.conditions)}, meds={len(self.medications)})"
        )


class PatientGenerator:
    """Generates synthetic patients for CDSS simulation.
    
    Creates realistic patient profiles by sampling from configured
    distributions. Patients have demographics, conditions, medications,
    and hidden physiological states.
    
    Args:
        available_drugs: List of drug IDs that can be prescribed
        available_conditions: List of condition names
        seed: Random seed for reproducibility
    """
    
    def __init__(
        self,
        available_drugs: List[str],
        available_conditions: List[str],
        seed: Optional[int] = None,
    ):
        """Initialize patient generator with available drugs and conditions.
        
        Args:
            available_drugs: List of drug IDs from knowledge base
            available_conditions: List of condition names
            seed: Optional random seed for reproducibility
        """
        self.available_drugs = available_drugs
        self.available_conditions = available_conditions
        self.rng = random.Random(seed)
        self._patient_counter = 0
    
    def generate(self) -> Patient:
        """Generate a single synthetic patient.
        
        Returns:
            Patient object with randomly sampled attributes
        """
        self._patient_counter += 1
        
        # Sample demographics
        age = self.rng.randint(18, 85)
        
        # Sample conditions (0-3)
        n_conditions = self.rng.randint(0, 3)
        conditions = self.rng.sample(
            self.available_conditions,
            min(n_conditions, len(self.available_conditions))
        )
        
        # Sample medications (1-4)
        n_medications = self.rng.randint(1, 4)
        medications = self.rng.sample(
            self.available_drugs,
            min(n_medications, len(self.available_drugs))
        )
        
        # Sample hidden physiological states
        renal_function = (
            "impaired" if self.rng.random() < 0.3 else "normal"
        )
        hepatic_function = (
            "impaired" if self.rng.random() < 0.2 else "normal"
        )
        
        # Lab results availability (60% chance)
        lab_results_available = self.rng.random() < 0.6
        
        return Patient(
            patient_id=self._patient_counter,
            age=age,
            conditions=conditions,
            medications=medications,
            renal_function=renal_function,
            hepatic_function=hepatic_function,
            lab_results_available=lab_results_available,
        )
    
    def generate_batch(self, n: int) -> List[Patient]:
        """Generate multiple patients.
        
        Args:
            n: Number of patients to generate
            
        Returns:
            List of Patient objects
        """
        return [self.generate() for _ in range(n)]
    
    def reset_counter(self) -> None:
        """Reset patient ID counter (useful for testing)."""
        self._patient_counter = 0
