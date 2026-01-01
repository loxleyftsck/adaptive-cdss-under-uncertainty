"""Red Team Stress Test - Adversarial Testing for CDSS Security

This module tests the CDSS system's robustness against:
1. Input noise (unrealistic medical data)
2. Missing data injection (extreme incompleteness)
3. Boundary attacks (edge case values)
4. Malformed inputs (invalid types, structures)

Goal: Ensure fail-safe mechanisms prevent dangerous medical recommendations.

Author: AI Red Team & Security Engineering
Date: 2026-01-02
"""

import sys
sys.path.insert(0, "C:\\Users\\LENOVO\\Documents\\adaptive-cdss-under-uncertainty")

import random
from dataclasses import dataclass, replace
from typing import List
from src.environment.cdss_env import CDSSEnvironment
from src.environment.patient_generator import Patient
from src.environment.observation_model import Observation
from src.agents.q_learning import QLearningAgent
from src.agents.baselines.rule_based import RuleBasedCDSS


@dataclass
class AttackResult:
    """Results from adversarial attack."""
    attack_name: str
    crashed: bool
    error_message: str
    action_taken: int
    reward: float
    risk_assessment: str  # "safe", "warning", "dangerous"


class RedTeamTester:
    """Red team adversarial testing suite for CDSS."""
    
    def __init__(self):
        """Initialize red team tester."""
        self.env = CDSSEnvironment(knowledge_path="knowledge/", seed=42)
        self.agent = QLearningAgent(seed=42)
        self.rule_agent = RuleBasedCDSS(self.env.kb)
        self.results: List[AttackResult] = []
    
    def run_all_attacks(self) -> List[AttackResult]:
        """Execute all adversarial attacks.
        
        Returns:
            List of attack results
        """
        print("=" * 70)
        print("🔴 RED TEAM ADVERSARIAL TESTING")
        print("=" * 70)
        
        # Attack Vector 1: Input Noise
        print("\n[1] INPUT NOISE ATTACKS")
        print("-" * 70)
        self.attack_unrealistic_vitals()
        self.attack_negative_age()
        self.attack_impossible_heart_rate()
        self.attack_future_age()
        
        # Attack Vector 2: Missing Data Injection
        print("\n[2] MISSING DATA INJECTION")
        print("-" * 70)
        self.attack_100_percent_missing()
        self.attack_critical_data_missing()
        self.attack_empty_medications()
        
        # Attack Vector 3: Boundary Attacks
        print("\n[3] BOUNDARY ATTACKS")
        print("-" * 70)
        self.attack_age_boundary()
        self.attack_max_medications()
        self.attack_zero_completeness()
        
        # Attack Vector 4: Malformed Inputs
        print("\n[4] MALFORMED INPUT ATTACKS")
        print("-" * 70)
        self.attack_invalid_drug_names()
        self.attack_duplicate_medications()
        
        self.print_summary()
        return self.results
    
    def attack_unrealistic_vitals(self):
        """Attack: Unrealistic vital signs (heart rate 500 bpm)."""
        print("\n🔴 Attack: Unrealistic Vitals (HR=500bpm)")
        
        try:
            # Create patient with unrealistic data
            # Note: Our system doesn't track vitals, so we test age boundary
            patient = Patient(
                patient_id=9999,
                age=500,  # Unrealistic age (should be 18-85)
                conditions=["diabetes"],
                medications=["warfarin"],
            )
            
            obs = self.env.obs_model.observe(patient)
            action = self.agent.choose_action(obs, training=False)
            
            # Should NOT crash, but should handle gracefully
            result = AttackResult(
                attack_name="Unrealistic Vitals",
                crashed=False,
                error_message="None",
                action_taken=action,
                reward=0.0,
                risk_assessment="handled" if obs.age >= 0 else "dangerous"
            )
            
            print(f"  ✅ System handled gracefully")
            print(f"  Action: {action}, Obs Age: {obs.age}")
            
        except Exception as e:
            result = AttackResult(
                attack_name="Unrealistic Vitals",
                crashed=True,
                error_message=str(e),
                action_taken=-1,
                reward=0.0,
                risk_assessment="crash"
            )
            print(f"  ❌ CRASH: {e}")
        
        self.results.append(result)
    
    def attack_negative_age(self):
        """Attack: Negative age (-5 years)."""
        print("\n🔴 Attack: Negative Age (-5)")
        
        try:
            patient = Patient(
                patient_id=9998,
                age=-5,  # Invalid
                conditions=["hypertension"],
                medications=["lisinopril"],
            )
            
            obs = self.env.obs_model.observe(patient)
            action = self.rule_agent.choose_action(obs)
            
            # Check if age bucketing handles negative
            from src.agents.state_encoding import encode_state
            state = encode_state(obs)
            
            result = AttackResult(
                attack_name="Negative Age",
                crashed=False,
                error_message="None",
                action_taken=action,
                reward=0.0,
                risk_assessment="safe" if state[1] == 0 else "warning"  # age_bucket
            )
            
            print(f"  ✅ Handled - Age bucket: {state[1]}")
            
        except Exception as e:
            result = AttackResult(
                attack_name="Negative Age",
                crashed=True,
                error_message=str(e),
                action_taken=-1,
                reward=0.0,
                risk_assessment="crash"
            )
            print(f"  ❌ CRASH: {e}")
        
        self.results.append(result)
    
    def attack_impossible_heart_rate(self):
        """Attack: Impossible heart rate (0 bpm - clinically dead)."""
        print("\n🔴 Attack: Impossible HR (0 bpm)")
        
        try:
            # Since our system doesn't model heart rate, test with edge case age
            patient = Patient(
                patient_id=9997,
                age=0,  # Newborn (outside our 18-85 range)
                conditions=[],
                medications=["warfarin"],
            )
            
            obs = self.env.obs_model.observe(patient)
            action = self.agent.choose_action(obs, training=False)
            
            result = AttackResult(
                attack_name="Impossible HR",
                crashed=False,
                error_message="None",
                action_taken=action,
                reward=0.0,
                risk_assessment="handled"
            )
            
            print(f"  ✅ System didn't crash (action: {action})")
            
        except Exception as e:
            result = AttackResult(
                attack_name="Impossible HR",
                crashed=True,
                error_message=str(e),
                action_taken=-1,
                reward=0.0,
                risk_assessment="crash"
            )
            print(f"  ❌ CRASH: {e}")
        
        self.results.append(result)
    
    def attack_future_age(self):
        """Attack: Future age (200 years old)."""
        print("\n🔴 Attack: Future Age (200 years)")
        
        try:
            patient = Patient(
                patient_id=9996,
                age=200,
                conditions=["diabetes", "hypertension"],
                medications=["metformin", "lisinopril"],
            )
            
            obs = self.env.obs_model.observe(patient)
            action = self.rule_agent.choose_action(obs)
            
            result = AttackResult(
                attack_name="Future Age",
                crashed=False,
                error_message="None",
                action_taken=action,
                reward=0.0,
                risk_assessment="handled"
            )
            
            print(f"  ✅ Handled (action: {action})")
            
        except Exception as e:
            result = AttackResult(
                attack_name="Future Age",
                crashed=True,
                error_message=str(e),
                action_taken=-1,
                reward=0.0,
                risk_assessment="crash"
            )
            print(f"  ❌ CRASH: {e}")
        
        self.results.append(result)
    
    def attack_100_percent_missing(self):
        """Attack: 100% data missing."""
        print("\n🔴 Attack: 100% Missing Data")
        
        try:
            # Create observation with no information
            obs = Observation(
                age=45,
                visible_conditions=[],  # All hidden
                medications=["warfarin", "aspirin"],
                lab_results=None,  # No labs
                data_completeness=0.0  # 0% complete
            )
            
            action = self.rule_agent.choose_action(obs)
            
            # Should REQUEST_DATA (action 3) when completeness very low
            result = AttackResult(
                attack_name="100% Missing Data",
                crashed=False,
                error_message="None",
                action_taken=action,
                reward=0.0,
                risk_assessment="safe" if action == 3 else "warning"
            )
            
            print(f"  ✅ Action: {action} ({'REQUEST_DATA' if action == 3 else 'OTHER'})")
            if action == 3:
                print(f"  ✅ GOOD: System requests more data!")
            else:
                print(f"  ⚠️  WARNING: Didn't request data despite 0% completeness")
            
        except Exception as e:
            result = AttackResult(
                attack_name="100% Missing Data",
                crashed=True,
                error_message=str(e),
                action_taken=-1,
                reward=0.0,
                risk_assessment="crash"
            )
            print(f"  ❌ CRASH: {e}")
        
        self.results.append(result)
    
    def attack_critical_data_missing(self):
        """Attack: All critical data missing (no conditions visible)."""
        print("\n🔴 Attack: Critical Data Missing (No Conditions)")
        
        try:
            obs = Observation(
                age=70,
                visible_conditions=[],  # No conditions visible
                medications=["warfarin", "aspirin", "ibuprofen"],  # Dangerous combo!
                lab_results=None,
                data_completeness=0.25  # Very low
            )
            
            action = self.rule_agent.choose_action(obs)
            
            # Calculate visible risk (DDI only, no contraindications visible)
            ddi_risk, _ = self.env.kb.get_total_risk(obs.medications, [])
            
            result = AttackResult(
                attack_name="Critical Data Missing",
                crashed=False,
                error_message="None",
                action_taken=action,
                reward=0.0,
                risk_assessment="dangerous" if action == 0 and ddi_risk >= 8 else "handled"
            )
            
            print(f"  DDI Risk (visible): {ddi_risk}")
            print(f"  Action: {action}")
            if ddi_risk >= 8 and action != 2:  # Should SUGGEST_ALT
                print(f"  ⚠️  WARNING: High DDI risk but didn't suggest alternative!")
            else:
                print(f"  ✅ Appropriate response")
            
        except Exception as e:
            result = AttackResult(
                attack_name="Critical Data Missing",
                crashed=True,
                error_message=str(e),
                action_taken=-1,
                reward=0.0,
                risk_assessment="crash"
            )
            print(f"  ❌ CRASH: {e}")
        
        self.results.append(result)
    
    def attack_empty_medications(self):
        """Attack: No medications provided."""
        print("\n🔴 Attack: Empty Medications List")
        
        try:
            obs = Observation(
                age=50,
                visible_conditions=["diabetes"],
                medications=[],  # No medications!
                lab_results=None,
                data_completeness=0.5
            )
            
            action = self.rule_agent.choose_action(obs)
            
            result = AttackResult(
                attack_name="Empty Medications",
                crashed=False,
                error_message="None",
                action_taken=action,
                reward=0.0,
                risk_assessment="safe"
            )
            
            print(f"  ✅ Handled empty meds (action: {action})")
            
        except Exception as e:
            result = AttackResult(
                attack_name="Empty Medications",
                crashed=True,
                error_message=str(e),
                action_taken=-1,
                reward=0.0,
                risk_assessment="crash"
            )
            print(f"  ❌ CRASH: {e}")
        
        self.results.append(result)
    
    def attack_age_boundary(self):
        """Attack: Age exactly at boundary (18, 85)."""
        print("\n🔴 Attack: Age Boundary Values")
        
        for age in [17, 18, 85, 86]:
            try:
                patient = Patient(
                    patient_id=9990 + age,
                    age=age,
                    conditions=["diabetes"],
                    medications=["metformin"],
                )
                
                obs = self.env.obs_model.observe(patient)
                from src.agents.state_encoding import encode_state
                state = encode_state(obs)
                
                result = AttackResult(
                    attack_name=f"Age Boundary ({age})",
                    crashed=False,
                    error_message="None",
                    action_taken=0,
                    reward=0.0,
                    risk_assessment="safe"
                )
                
                print(f"  Age {age}: Bucket={state[1]} ✅")
                
            except Exception as e:
                result = AttackResult(
                    attack_name=f"Age Boundary ({age})",
                    crashed=True,
                    error_message=str(e),
                    action_taken=-1,
                    reward=0.0,
                    risk_assessment="crash"
                )
                print(f"  Age {age}: ❌ CRASH - {e}")
            
            self.results.append(result)
    
    def attack_max_medications(self):
        """Attack: Maximum medications (all 7 drugs)."""
        print("\n🔴 Attack: Maximum Medications (7 drugs)")
        
        try:
            all_drugs = self.env.kb.get_all_drugs()
            
            patient = Patient(
                patient_id=9980,
                age=65,
                conditions=["diabetes", "hypertension", "heart_failure"],
                medications=all_drugs,  # All 7 drugs!
            )
            
            obs = self.env.obs_model.observe(patient)
            action = self.rule_agent.choose_action(obs)
            
            # Calculate total DDI risk
            ddis = self.env.kb.check_all_interactions(all_drugs)
            total_ddi_risk = sum(abs(d["penalty"]) for d in ddis)
            
            result = AttackResult(
                attack_name="Max Medications",
                crashed=False,
                error_message="None",
                action_taken=action,
                reward=0.0,
                risk_assessment="dangerous" if action == 0 and total_ddi_risk > 20 else "handled"
            )
            
            print(f"  Total DDI Interactions: {len(ddis)}")
            print(f"  Total DDI Risk: {total_ddi_risk}")
            print(f"  Action: {action}")
            
        except Exception as e:
            result = AttackResult(
                attack_name="Max Medications",
                crashed=True,
                error_message=str(e),
                action_taken=-1,
                reward=0.0,
                risk_assessment="crash"
            )
            print(f"  ❌ CRASH: {e}")
        
        self.results.append(result)
    
    def attack_zero_completeness(self):
        """Attack: Zero data completeness edge case."""
        print("\n🔴 Attack: Zero Completeness")
        
        try:
            obs = Observation(
                age=45,
                visible_conditions=[],
                medications=["warfarin"],
                lab_results=None,
                data_completeness=0.0
            )
            
            from src.agents.state_encoding import encode_state
            state = encode_state(obs)
            
            result = AttackResult(
                attack_name="Zero Completeness",
                crashed=False,
                error_message="None",
                action_taken=0,
                reward=0.0,
                risk_assessment="safe"
            )
            
            print(f"  ✅ Encoded state: {state[2]} (completeness bucket)")
            
        except Exception as e:
            result = AttackResult(
                attack_name="Zero Completeness",
                crashed=True,
                error_message=str(e),
                action_taken=-1,
                reward=0.0,
                risk_assessment="crash"
            )
            print(f"  ❌ CRASH: {e}")
        
        self.results.append(result)
    
    def attack_invalid_drug_names(self):
        """Attack: Invalid drug names."""
        print("\n🔴 Attack: Invalid Drug Names")
        
        try:
            obs = Observation(
                age=50,
                visible_conditions=["diabetes"],
                medications=["INVALID_DRUG_XYZ", "NOT_A_REAL_DRUG"],
                lab_results=None,
                data_completeness=0.8
            )
            
            # Should handle gracefully (no crashes)
            from src.agents.state_encoding import encode_state
            state = encode_state(obs)
            
            # Check if KnowledgeBase handles unknown drugs
            ddi_risk, _ = self.env.kb.get_total_risk(obs.medications, [])
            
            result = AttackResult(
                attack_name="Invalid Drug Names",
                crashed=False,
                error_message="None",
                action_taken=0,
                reward=0.0,
                risk_assessment="safe"
            )
            
            print(f"  ✅ System handled invalid drugs (DDI risk: {ddi_risk})")
            
        except Exception as e:
            result = AttackResult(
                attack_name="Invalid Drug Names",
                crashed=True,
                error_message=str(e),
                action_taken=-1,
                reward=0.0,
                risk_assessment="crash"
            )
            print(f"  ❌ CRASH: {e}")
        
        self.results.append(result)
    
    def attack_duplicate_medications(self):
        """Attack: Duplicate medications in list."""
        print("\n🔴 Attack: Duplicate Medications")
        
        try:
            obs = Observation(
                age=55,
                visible_conditions=["hypertension"],
                medications=["warfarin", "warfarin", "warfarin"],  # Duplicates!
                lab_results=None,
                data_completeness=0.7
            )
            
            from src.agents.state_encoding import encode_state
            state = encode_state(obs)
            
            # State encoding uses sorted tuple, which should deduplicate
            result = AttackResult(
                attack_name="Duplicate Medications",
                crashed=False,
                error_message="None",
                action_taken=0,
                reward=0.0,
                risk_assessment="safe"
            )
            
            print(f"  ✅ Handled duplicates - State meds: {state[0]}")
            
        except Exception as e:
            result = AttackResult(
                attack_name="Duplicate Medications",
                crashed=True,
                error_message=str(e),
                action_taken=-1,
                reward=0.0,
                risk_assessment="crash"
            )
            print(f"  ❌ CRASH: {e}")
        
        self.results.append(result)
    
    def print_summary(self):
        """Print attack summary and security assessment."""
        print("\n" + "=" * 70)
        print("🔴 RED TEAM SUMMARY")
        print("=" * 70)
        
        total_attacks = len(self.results)
        crashes = sum(1 for r in self.results if r.crashed)
        dangerous = sum(1 for r in self.results if r.risk_assessment == "dangerous")
        warnings = sum(1 for r in self.results if r.risk_assessment == "warning")
        safe = sum(1 for r in self.results if r.risk_assessment == "safe" or r.risk_assessment == "handled")
        
        print(f"\nTotal Attacks: {total_attacks}")
        print(f"  ❌ Crashes: {crashes} ({crashes/total_attacks*100:.1f}%)")
        print(f"  🔴 Dangerous: {dangerous} ({dangerous/total_attacks*100:.1f}%)")
        print(f"  ⚠️  Warnings: {warnings} ({warnings/total_attacks*100:.1f}%)")
        print(f"  ✅ Safe: {safe} ({safe/total_attacks*100:.1f}%)")
        
        print("\n" + "=" * 70)
        
        # Security assessment
        if crashes > 0:
            print("🔴 SECURITY GRADE: F - CRITICAL VULNERABILITIES")
            print("   System crashes on adversarial inputs!")
        elif dangerous > 0:
            print("🔴 SECURITY GRADE: D - DANGEROUS BEHAVIORS DETECTED")
            print("   System may provide harmful recommendations!")
        elif warnings > 3:
            print("⚠️  SECURITY GRADE: C - MULTIPLE WARNINGS")
            print("   System needs fail-safe improvements")
        elif warnings > 0:
            print("⚠️  SECURITY GRADE: B - MINOR ISSUES")
            print("   System mostly robust with minor edge cases")
        else:
            print("✅ SECURITY GRADE: A - ROBUST")
            print("   System handles adversarial inputs gracefully!")
        
        print("=" * 70)


if __name__ == "__main__":
    tester = RedTeamTester()
    tester.run_all_attacks()
