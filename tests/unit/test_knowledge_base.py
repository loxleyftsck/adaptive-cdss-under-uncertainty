"""Unit tests for KnowledgeBase class."""

import pytest
import json
import tempfile
from pathlib import Path
from src.knowledge.knowledge_base import KnowledgeBase


@pytest.fixture
def temp_knowledge_dir():
    """Create temporary knowledge directory with test data."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        
        # Create test drugs
        drugs = {
            "drugs": [
                {"id": "drug_a", "name": "Drug A", "drug_class": "Test"},
                {"id": "drug_b", "name": "Drug B", "drug_class": "Test"},
                {"id": "drug_c", "name": "Drug C", "drug_class": "Test"},
            ],
            "metadata": {"version": "1.0.0"}
        }
        
        # Create test interactions
        interactions = {
            "interactions": [
                {
                    "id": "ddi_001",
                    "drug_pair": ["drug_a", "drug_b"],
                    "severity": "high",
                    "penalty": -10,
                },
                {
                    "id": "ddi_002",
                    "drug_pair": ["drug_b", "drug_c"],
                    "severity": "low",
                    "penalty": -2,
                },
            ],
            "metadata": {"total_interactions": 2}
        }
        
        # Create test contraindications
        contraindications = {
            "contraindications": [
                {
                    "id": "contra_001",
                    "drug": "drug_a",
                    "condition": "renal_impairment",
                    "penalty": -8,
                },
            ],
            "metadata": {"total_contraindications": 1}
        }
        
        # Write JSON files
        for filename, data in [
            ("drugs.json", drugs),
            ("interactions.json", interactions),
            ("contraindications.json", contraindications),
        ]:
            with open(tmpdir / filename, "w") as f:
                json.dump(data, f)
        
        yield str(tmpdir)


def test_knowledge_base_initialization(temp_knowledge_dir):
    """Test KnowledgeBase loads successfully."""
    kb = KnowledgeBase(temp_knowledge_dir)
    
    assert len(kb.drugs) == 3
    assert len(kb.interactions) == 2
    assert len(kb.contraindications) == 1


def test_get_drug(temp_knowledge_dir):
    """Test retrieving drug by ID."""
    kb = KnowledgeBase(temp_knowledge_dir)
    
    drug = kb.get_drug("drug_a")
    assert drug is not None
    assert drug["name"] == "Drug A"
    
    # Non-existent drug
    assert kb.get_drug("drug_z") is None


def test_get_interaction(temp_knowledge_dir):
    """Test interaction lookup."""
    kb = KnowledgeBase(temp_knowledge_dir)
    
    # Forward order
    interaction = kb.get_interaction("drug_a", "drug_b")
    assert interaction is not None
    assert interaction["severity"] == "high"
    assert interaction["penalty"] == -10
    
    # Reverse order (should work - bidirectional)
    interaction_rev = kb.get_interaction("drug_b", "drug_a")
    assert interaction_rev is not None
    assert interaction_rev["id"] == interaction["id"]
    
    # No interaction
    assert kb.get_interaction("drug_a", "drug_c") is None


def test_check_all_interactions(temp_knowledge_dir):
    """Test checking multiple medications for interactions."""
    kb = KnowledgeBase(temp_knowledge_dir)
    
    # Single interaction
    medications = ["drug_a", "drug_b"]
    interactions = kb.check_all_interactions(medications)
    assert len(interactions) == 1
    assert interactions[0]["id"] == "ddi_001"
    
    # Two interactions (transitive)
    medications = ["drug_a", "drug_b", "drug_c"]
    interactions = kb.check_all_interactions(medications)
    assert len(interactions) == 2  # drug_a+drug_b, drug_b+drug_c
    
    # No interactions
    medications = ["drug_a"]
    interactions = kb.check_all_interactions(medications)
    assert len(interactions) == 0


def test_check_contraindication(temp_knowledge_dir):
    """Test contraindication checking."""
    kb = KnowledgeBase(temp_knowledge_dir)
    
    contra = kb.check_contraindication("drug_a", "renal_impairment")
    assert contra is not None
    assert contra["penalty"] == -8
    
    # No contraindication
    assert kb.check_contraindication("drug_b", "renal_impairment") is None


def test_get_total_risk(temp_knowledge_dir):
    """Test total risk calculation."""
    kb = KnowledgeBase(temp_knowledge_dir)
    
    # DDI only
    medications = ["drug_a", "drug_b"]
    conditions = []
    ddi_risk, contra_risk = kb.get_total_risk(medications, conditions)
    assert ddi_risk == 10  # abs(-10)
    assert contra_risk == 0
    
    # Contraindication only
    medications = ["drug_a"]
    conditions = ["renal_impairment"]
    ddi_risk, contra_risk = kb.get_total_risk(medications, conditions)
    assert ddi_risk == 0
    assert contra_risk == 8  # abs(-8)
    
    # Both DDI and contraindication
    medications = ["drug_a", "drug_b"]
    conditions = ["renal_impairment"]
    ddi_risk, contra_risk = kb.get_total_risk(medications, conditions)
    assert ddi_risk == 10
    assert contra_risk == 8


def test_get_all_drugs(temp_knowledge_dir):
    """Test getting all drug IDs."""
    kb = KnowledgeBase(temp_knowledge_dir)
    
    drugs = kb.get_all_drugs()
    assert len(drugs) == 3
    assert set(drugs) == {"drug_a", "drug_b", "drug_c"}


def test_missing_file():
    """Test error when knowledge file missing."""
    with pytest.raises(FileNotFoundError):
        KnowledgeBase("nonexistent_path/")


def test_validation_fails_invalid_drug_reference(temp_knowledge_dir):
    """Test validation catches invalid drug references."""
    tmpdir = Path(temp_knowledge_dir)
    
    # Add interaction with non-existent drug
    with open(tmpdir / "interactions.json", "w") as f:
        json.dump({
            "interactions": [{
                "id": "bad",
                "drug_pair": ["drug_a", "drug_z"],  # drug_z doesn't exist
                "penalty": -10
            }],
            "metadata": {}
        }, f)
    
    with pytest.raises(ValueError, match="unknown drug"):
        KnowledgeBase(temp_knowledge_dir)


def test_repr(temp_knowledge_dir):
    """Test string representation."""
    kb = KnowledgeBase(temp_knowledge_dir)
    repr_str = repr(kb)
    
    assert "drugs=3" in repr_str
    assert "interactions=2" in repr_str
    assert "contraindications=1" in repr_str


# Integration test with real knowledge base
def test_real_knowledge_base():
    """Test loading actual knowledge base from knowledge/ directory."""
    try:
        kb = KnowledgeBase("knowledge/")
        
        # Verify data loaded
        assert len(kb.drugs) > 0
        assert len(kb.interactions) > 0
        assert len(kb.contraindications) > 0
        
        # Test specific drug
        warfarin = kb.get_drug("warfarin")
        assert warfarin is not None
        assert "anticoagulant" in warfarin["drug_class"].lower()
        
        # Test known interaction
        interaction = kb.get_interaction("warfarin", "aspirin")
        assert interaction is not None
        assert interaction["severity"] == "high"
        
    except FileNotFoundError:
        pytest.skip("Real knowledge base not available")
