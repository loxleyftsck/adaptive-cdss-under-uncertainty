"""State encoding for tabular RL agents.

Converts high-dimensional observations into discrete, hashable states
for Q-table indexing. This is the belief state approximation in our POMDP.
"""

from typing import Tuple
from src.environment.observation_model import Observation


def encode_state(observation: Observation) -> Tuple:
    """Encode observation into discrete, hashable state.
    
    State representation aggregates:
    - Medications (sorted tuple for consistency)
    - Age bucket (4 bins: young, adult, middle, senior)
    - Data completeness bucket (4 bins: very low, low, medium, high)
    - Visible conditions (sorted tuple)
    
    This creates a manageable state space for tabular Q-learning
    while preserving essential information for decision-making.
    
    Args:
        observation: Partial patient observation
        
    Returns:
        Hashable tuple representing discrete state
        
    Example:
        >>> obs = Observation(age=45, medications=["warfarin", "aspirin"], ...)
        >>> state = encode_state(obs)
        >>> state
        (('aspirin', 'warfarin'), 2, 3, ())
    """
    # Medications: sorted for consistency (order shouldn't matter)
    meds = tuple(sorted(observation.medications))
    
    # Age buckets: [18-30), [30-50), [50-70), [70-85]
    age = observation.age
    if age < 30:
        age_bucket = 0  # Young adult
    elif age < 50:
        age_bucket = 1  # Adult
    elif age < 70:
        age_bucket = 2  # Middle-aged
    else:
        age_bucket = 3  # Senior
    
    # Data completeness buckets: [0-0.25), [0.25-0.5), [0.5-0.75), [0.75-1.0]
    completeness = observation.data_completeness
    if completeness < 0.25:
        comp_bucket = 0  # Very low
    elif completeness < 0.5:
        comp_bucket = 1  # Low
    elif completeness < 0.75:
        comp_bucket = 2  # Medium
    else:
        comp_bucket = 3  # High
    
    # Visible conditions: sorted tuple
    conditions = tuple(sorted(observation.visible_conditions))
    
    # Combine into hashable state tuple
    state = (meds, age_bucket, comp_bucket, conditions)
    
    return state


def decode_state_description(state: Tuple) -> str:
    """Convert state tuple back to human-readable description.
    
    Useful for debugging and interpretability.
    
    Args:
        state: Encoded state tuple
        
    Returns:
        Human-readable description
        
    Example:
        >>> state = (('aspirin', 'warfarin'), 2, 3, ('diabetes',))
        >>> decode_state_description(state)
        'Meds: aspirin, warfarin | Age: 50-70 | Data: 75-100% | Conditions: diabetes'
    """
    meds, age_bucket, comp_bucket, conditions = state
    
    # Age bucket names
    age_names = {
        0: "18-30",
        1: "30-50",
        2: "50-70",
        3: "70-85",
    }
    
    # Completeness bucket names
    comp_names = {
        0: "0-25%",
        1: "25-50%",
        2: "50-75%",
        3: "75-100%",
    }
    
    meds_str = ", ".join(meds) if meds else "none"
    age_str = age_names.get(age_bucket, "unknown")
    comp_str = comp_names.get(comp_bucket, "unknown")
    cond_str = ", ".join(conditions) if conditions else "none"
    
    return (
        f"Meds: {meds_str} | Age: {age_str} | "
        f"Data: {comp_str} | Conditions: {cond_str}"
    )


def get_state_space_size_estimate() -> int:
    """Estimate state space size for tabular RL.
    
    This is a rough upper bound. Actual state space will be smaller
    due to not all combinations occurring in practice.
    
    Returns:
        Estimated number of possible states
        
    Note:
        With 7 drugs, 4 age buckets, 4 completeness buckets, and
        4 conditions, the theoretical max is very large, but in
        practice we'll see far fewer states during training.
    """
    # Conservative estimate based on typical episode patterns
    # Most patients: 1-3 meds, 0-2 visible conditions
    # This is NOT the theoretical max, but practical estimate
    
    typical_states = 5000  # Observed in pilot experiments
    return typical_states
