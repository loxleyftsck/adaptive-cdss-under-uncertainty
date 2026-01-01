"""Knowledge Base module for pharmacological data.

This module provides access to drug information, drug-drug interactions,
and contraindications used by the CDSS environment.
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any


class KnowledgeBase:
    """Manages pharmacological knowledge for prescription decision-making.
    
    Loads and provides access to:
    - Drug metadata (mechanism, indications, monitoring)
    - Drug-drug interactions (DDI) with severity levels
    - Drug-condition contraindications
    
    Attributes:
        drugs: Dictionary of drug information indexed by drug ID
        interactions: List of drug-drug interaction dictionaries
        contraindications: List of drug-condition contraindication dictionaries
    """
    
    def __init__(self, knowledge_path: str = "knowledge/"):
        """Initialize knowledge base by loading JSON files.
        
        Args:
            knowledge_path: Path to directory containing JSON knowledge files
            
        Raises:
            FileNotFoundError: If knowledge files are missing
            json.JSONDecodeError: If JSON files are malformed
        """
        self.knowledge_path = Path(knowledge_path)
        self.drugs: Dict[str, Dict] = {}
        self.interactions: List[Dict] = []
        self.contraindications: List[Dict] = []
        
        self._load_all()
        self._validate()
    
    def _load_all(self) -> None:
        """Load all knowledge files from disk."""
        self.drugs = self._load_json("drugs.json")["drugs"]
        self.interactions = self._load_json("interactions.json")["interactions"]
        self.contraindications = self._load_json("contraindications.json")[
            "contraindications"
        ]
        
        # Index drugs by ID for fast lookup
        self.drugs = {drug["id"]: drug for drug in self.drugs}
    
    def _load_json(self, filename: str) -> Dict:
        """Load and parse a JSON file.
        
        Args:
            filename: Name of JSON file to load
            
        Returns:
            Parsed JSON as dictionary
            
        Raises:
            FileNotFoundError: If file does not exist
            json.JSONDecodeError: If JSON is malformed
        """
        filepath = self.knowledge_path / filename
        if not filepath.exists():
            raise FileNotFoundError(f"Knowledge file not found: {filepath}")
        
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)
    
    def _validate(self) -> None:
        """Validate knowledge base consistency.
        
        Raises:
            ValueError: If validation fails
        """
        # Check drugs exist
        if not self.drugs:
            raise ValueError("No drugs loaded")
        
        # Validate interactions reference existing drugs
        for interaction in self.interactions:
            drug_pair = interaction["drug_pair"]
            for drug_id in drug_pair:
                if drug_id not in self.drugs:
                    raise ValueError(
                        f"Interaction references unknown drug: {drug_id}"
                    )
        
        # Validate contraindications reference existing drugs
        for contra in self.contraindications:
            drug_id = contra["drug"]
            if drug_id not in self.drugs:
                raise ValueError(
                    f"Contraindication references unknown drug: {drug_id}"
                )
    
    def get_drug(self, drug_id: str) -> Optional[Dict]:
        """Retrieve drug information by ID.
        
        Args:
            drug_id: Drug identifier (e.g., 'warfarin')
            
        Returns:
            Drug information dictionary, or None if not found
        """
        return self.drugs.get(drug_id)
    
    def get_interaction(
        self, drug1: str, drug2: str
    ) -> Optional[Dict]:
        """Check if two drugs have a known interaction.
        
        Args:
            drug1: First drug ID
            drug2: Second drug ID
            
        Returns:
            Interaction dictionary if found, None otherwise
        """
        # Normalize order (interactions are bidirectional)
        pair1 = sorted([drug1, drug2])
        
        for interaction in self.interactions:
            pair2 = sorted(interaction["drug_pair"])
            if pair1 == pair2:
                return interaction
        
        return None
    
    def check_all_interactions(
        self, medications: List[str]
    ) -> List[Dict]:
        """Find all interactions in a medication list.
        
        Args:
            medications: List of drug IDs
            
        Returns:
            List of all detected interactions
        """
        detected = []
        
        # Check all pairs
        for i, drug1 in enumerate(medications):
            for drug2 in medications[i + 1 :]:
                interaction = self.get_interaction(drug1, drug2)
                if interaction:
                    detected.append(interaction)
        
        return detected
    
    def check_contraindication(
        self, drug: str, condition: str
    ) -> Optional[Dict]:
        """Check if drug is contraindicated for a condition.
        
        Args:
            drug: Drug ID
            condition: Condition name (e.g., 'renal_impairment')
            
        Returns:
            Contraindication dictionary if found, None otherwise
        """
        for contra in self.contraindications:
            if contra["drug"] == drug and contra["condition"] == condition:
                return contra
        
        return None
    
    def get_total_risk(
        self, medications: List[str], conditions: List[str]
    ) -> Tuple[int, int]:
        """Calculate total risk from DDIs and contraindications.
        
        Args:
            medications: List of drug IDs
            conditions: List of condition names
            
        Returns:
            Tuple of (ddi_risk, contraindication_risk) as penalty sums
        """
        ddi_risk = 0
        contra_risk = 0
        
        # Sum DDI penalties
        interactions = self.check_all_interactions(medications)
        for interaction in interactions:
            ddi_risk += abs(interaction["penalty"])
        
        # Sum contraindication penalties
        for drug in medications:
            for condition in conditions:
                contra = self.check_contraindication(drug, condition)
                if contra:
                    contra_risk += abs(contra["penalty"])
        
        return ddi_risk, contra_risk
    
    def get_all_drugs(self) -> List[str]:
        """Get list of all drug IDs.
        
        Returns:
            List of drug ID strings
        """
        return list(self.drugs.keys())
    
    def __repr__(self) -> str:
        """String representation of knowledge base."""
        return (
            f"KnowledgeBase("
            f"drugs={len(self.drugs)}, "
            f"interactions={len(self.interactions)}, "
            f"contraindications={len(self.contraindications)})"
        )
