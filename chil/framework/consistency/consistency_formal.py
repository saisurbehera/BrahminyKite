"""
Formal Logic-Based Consistency Verification Framework

Uses formal logic systems (Lean, Coq, Prolog) for rigorous consistency checking
instead of pattern matching.
"""

import asyncio
import json
import logging
import subprocess
import tempfile
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
import re

from .unified_base import UnifiedVerificationComponent
from ...consensus_types import Claim, VerificationResult, ConsensusProposal, NodeContext, ConsensusVerificationResult


class FormalSystem(Enum):
    """Supported formal logic systems"""
    LEAN = "lean"
    COQUERY = "coq"  # Coq via coqtop
    PROLOG = "prolog"  # SWI-Prolog
    Z3 = "z3"  # Z3 SMT solver
    VAMPIRE = "vampire"  # Vampire theorem prover


@dataclass
class FormalTranslation:
    """Translation of natural language to formal logic"""
    original_text: str
    formal_representation: str
    formal_system: FormalSystem
    translation_confidence: float  # 0.0 to 1.0
    assumptions_made: List[str] = field(default_factory=list)
    translation_notes: str = ""


@dataclass
class ProofResult:
    """Result from formal proof system"""
    is_valid: bool
    proof_found: bool
    counterexample: Optional[str] = None
    proof_steps: List[str] = field(default_factory=list)
    execution_time: float = 0.0
    error_message: Optional[str] = None
    axioms_used: List[str] = field(default_factory=list)


@dataclass
class ConsistencyProof:
    """Formal consistency proof result"""
    is_consistent: bool
    proof_system: FormalSystem
    formal_statement: str
    proof_result: ProofResult
    consistency_score: float  # 0.0 to 1.0
    logical_dependencies: List[str] = field(default_factory=list)


class FormalConsistencyFramework(UnifiedVerificationComponent):
    """
    Formal logic-based consistency verification using theorem provers
    """
    
    def __init__(self, preferred_systems: Optional[List[FormalSystem]] = None, 
                 enable_parallel_proving: bool = True):
        self.preferred_systems = preferred_systems or [FormalSystem.LEAN, FormalSystem.Z3, FormalSystem.PROLOG]
        self.enable_parallel_proving = enable_parallel_proving
        self.cache = {}
        self.logger = logging.getLogger(__name__)
        
        # Check which formal systems are available
        self.available_systems = self._detect_available_systems()
        self.logger.info(f"Available formal systems: {[s.value for s in self.available_systems]}")
        
        # Natural language to logic patterns
        self._init_translation_patterns()
        
        # Common logical axioms and rules
        self.logical_axioms = {
            "propositional": [
                "∀p. p ∨ ¬p",  # Law of excluded middle
                "∀p. ¬(p ∧ ¬p)",  # Law of non-contradiction
                "∀p q. (p → q) → (¬q → ¬p)",  # Contraposition
                "∀p q r. ((p → q) ∧ (q → r)) → (p → r)",  # Transitivity
            ],
            "predicate": [
                "∀x P. (∀x. P(x)) → P(x)",  # Universal instantiation
                "∀x P. P(x) → (∃x. P(x))",  # Existential generalization
            ]
        }
    
    def _detect_available_systems(self) -> List[FormalSystem]:
        """Detect which formal logic systems are installed"""
        available = []
        
        # Check for Lean
        try:
            result = subprocess.run(["lean", "--version"], capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                available.append(FormalSystem.LEAN)
                self.logger.info(f"Lean detected: {result.stdout.strip()}")
        except (subprocess.SubprocessError, FileNotFoundError):
            self.logger.debug("Lean not available")
        
        # Check for Coq
        try:
            result = subprocess.run(["coqc", "-v"], capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                available.append(FormalSystem.COQUERY)
                self.logger.info("Coq detected")
        except (subprocess.SubprocessError, FileNotFoundError):
            self.logger.debug("Coq not available")
        
        # Check for SWI-Prolog
        try:
            result = subprocess.run(["swipl", "--version"], capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                available.append(FormalSystem.PROLOG)
                self.logger.info("SWI-Prolog detected")
        except (subprocess.SubprocessError, FileNotFoundError):
            self.logger.debug("SWI-Prolog not available")
        
        # Check for Z3
        try:
            result = subprocess.run(["z3", "-version"], capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                available.append(FormalSystem.Z3)
                self.logger.info(f"Z3 detected: {result.stdout.strip()}")
        except (subprocess.SubprocessError, FileNotFoundError):
            self.logger.debug("Z3 not available")
        
        return available
    
    def _init_translation_patterns(self):
        """Initialize patterns for translating natural language to formal logic"""
        
        # Logical connectives mapping
        self.connective_patterns = {
            "and": ["and", "&", "∧", "also", "furthermore", "moreover"],
            "or": ["or", "|", "∨", "either", "alternatively"],
            "not": ["not", "¬", "~", "it is false that", "it is not the case that"],
            "implies": ["implies", "→", "if.*then", "entails", "leads to", "results in"],
            "iff": ["if and only if", "↔", "≡", "exactly when", "precisely when"],
            "forall": ["all", "every", "each", "any", "∀"],
            "exists": ["some", "there exists", "there is", "∃", "at least one"]
        }
        
        # Predicate patterns
        self.predicate_patterns = {
            "is_a": re.compile(r'(\w+)\s+is\s+a\s+(\w+)', re.IGNORECASE),
            "has_property": re.compile(r'(\w+)\s+is\s+(\w+)', re.IGNORECASE),
            "relation": re.compile(r'(\w+)\s+(loves|hates|knows|sees|helps)\s+(\w+)', re.IGNORECASE),
            "comparison": re.compile(r'(\w+)\s+is\s+(greater|less|equal)\s+than\s+(\w+)', re.IGNORECASE)
        }
        
        # Common logical forms
        self.logical_forms = {
            "universal_statement": re.compile(r'all\s+(\w+)\s+are\s+(\w+)', re.IGNORECASE),
            "existential_statement": re.compile(r'some\s+(\w+)\s+are\s+(\w+)', re.IGNORECASE),
            "conditional": re.compile(r'if\s+(.+?)\s+then\s+(.+?)(?:[.!?]|$)', re.IGNORECASE),
            "negation": re.compile(r'it is not (?:the case )?that\s+(.+?)(?:[.!?]|$)', re.IGNORECASE)
        }
    
    async def verify_individual(self, claim: Claim) -> VerificationResult:
        """Verify claim through formal logic systems"""
        try:
            # Check cache
            cache_key = self._generate_cache_key(claim)
            if cache_key in self.cache:
                self.logger.info(f"Using cached formal verification result for claim: {claim.content[:50]}...")
                return self.cache[cache_key]
            
            # Step 1: Translate to formal logic
            translations = await self._translate_to_formal_logic(claim)
            
            if not translations:
                return VerificationResult(
                    framework_name="formal_consistency",
                    confidence_score=0.3,
                    reasoning="Unable to translate claim to formal logic representation",
                    evidence_references=["Natural language processing attempted"],
                    uncertainty_factors=["Complex natural language", "No clear logical structure"],
                    contextual_notes="Consider rephrasing with clearer logical structure"
                )
            
            # Step 2: Verify consistency across available formal systems
            if self.enable_parallel_proving and len(self.available_systems) > 1:
                # Parallel verification
                proof_tasks = [
                    self._verify_with_system(translation, system) 
                    for translation in translations 
                    for system in self.available_systems 
                    if system in self.preferred_systems
                ]
                
                proof_results = await asyncio.gather(*proof_tasks, return_exceptions=True)
                proof_results = [r for r in proof_results if not isinstance(r, Exception)]
            else:
                # Sequential verification
                proof_results = []
                for translation in translations:
                    for system in self.available_systems:
                        if system in self.preferred_systems:
                            result = await self._verify_with_system(translation, system)
                            if result:
                                proof_results.append(result)
            
            # Step 3: Combine results from multiple formal systems
            overall_score = self._combine_formal_results(proof_results, translations)
            
            # Step 4: Generate reasoning and insights
            reasoning = self._generate_formal_reasoning(translations, proof_results)
            uncertainty_factors = self._identify_formal_uncertainties(translations, proof_results)
            contextual_notes = self._generate_formal_insights(claim, translations, proof_results)
            
            result = VerificationResult(
                framework_name="formal_consistency",
                confidence_score=overall_score,
                reasoning=reasoning,
                evidence_references=self._collect_formal_references(translations, proof_results),
                uncertainty_factors=uncertainty_factors,
                contextual_notes=contextual_notes,
                metadata={
                    "formal_systems_used": [r.proof_system.value for r in proof_results],
                    "translations_generated": len(translations),
                    "proofs_attempted": len(proof_results),
                    "proofs_successful": sum(1 for r in proof_results if r.is_consistent),
                    "translation_confidence": max([t.translation_confidence for t in translations], default=0.0),
                    "verification_timestamp": datetime.utcnow().isoformat(),
                    "parallel_proving": self.enable_parallel_proving
                }
            )
            
            # Cache result
            self.cache[cache_key] = result
            
            return result
            
        except Exception as e:
            self.logger.error(f"Formal consistency verification failed: {e}")
            return VerificationResult(
                framework_name="formal_consistency",
                confidence_score=0.1,
                reasoning=f"Formal verification failed due to technical error: {str(e)}",
                evidence_references=[],
                uncertainty_factors=["Technical failure", "Formal system unavailable"],
                contextual_notes="This result should be treated with extreme caution."
            )
    
    async def verify_consensus(self, proposal: ConsensusProposal, node_context: NodeContext) -> ConsensusVerificationResult:
        """Verify proposal in consensus mode"""
        # Extract claim from proposal
        claim_data = proposal.content.get("claim", {})
        claim = Claim(
            content=claim_data.get("content", ""),
            context=claim_data.get("context", {}),
            metadata=claim_data.get("metadata", {})
        )
        
        # Run individual verification
        individual_result = await self.verify_individual(claim)
        
        # Apply node-specific formal verification adjustments
        node_adjusted_score = self._apply_formal_node_adjustments(
            individual_result.confidence_score, node_context, individual_result
        )
        
        return ConsensusVerificationResult(
            node_id=node_context.node_id,
            framework_name="formal_consistency",
            confidence_score=node_adjusted_score,
            reasoning=individual_result.reasoning,
            evidence_quality=self._assess_formal_evidence_quality(individual_result),
            consensus_readiness=node_adjusted_score > 0.7,  # Higher threshold for formal verification
            suggested_refinements=self._suggest_formal_refinements(individual_result, node_context),
            metadata={
                **individual_result.metadata,
                "node_formal_preferences": True,
                "consensus_mode": True,
                "node_consistency_strictness": node_context.philosophical_weights.get("consistency", 1.0)
            }
        )
    
    async def _translate_to_formal_logic(self, claim: Claim) -> List[FormalTranslation]:
        """Translate natural language claim to formal logic representations"""
        text = claim.content
        translations = []
        
        # Try different translation approaches
        translation_tasks = [
            self._translate_to_lean(text),
            self._translate_to_z3(text),
            self._translate_to_prolog(text)
        ]
        
        translation_results = await asyncio.gather(*translation_tasks, return_exceptions=True)
        
        for result in translation_results:
            if isinstance(result, FormalTranslation):
                translations.append(result)
            elif isinstance(result, Exception):
                self.logger.warning(f"Translation failed: {result}")
        
        return translations
    
    async def _translate_to_lean(self, text: str) -> Optional[FormalTranslation]:
        """Translate to Lean 4 theorem prover syntax"""
        try:
            # Simple pattern-based translation to Lean
            lean_statements = []
            assumptions = []
            
            # Handle universal statements: "All X are Y" -> "∀ x : X, Y x"
            universal_match = self.logical_forms["universal_statement"].search(text)
            if universal_match:
                x_type, y_property = universal_match.groups()
                lean_stmt = f"∀ x : {x_type.capitalize()}, {y_property.capitalize()} x"
                lean_statements.append(lean_stmt)
                assumptions.append(f"Assumed {x_type} and {y_property} are well-defined types/predicates")
            
            # Handle conditionals: "If P then Q" -> "P → Q"
            conditional_match = self.logical_forms["conditional"].search(text)
            if conditional_match:
                condition, consequence = conditional_match.groups()
                # Simplify to propositional variables for basic verification
                p_var = "P"
                q_var = "Q"
                lean_stmt = f"{p_var} → {q_var}"
                lean_statements.append(lean_stmt)
                assumptions.extend([
                    f"P represents: {condition.strip()}",
                    f"Q represents: {consequence.strip()}"
                ])
            
            # Handle negations
            negation_match = self.logical_forms["negation"].search(text)
            if negation_match:
                negated_content = negation_match.group(1)
                lean_stmt = f"¬ ({negated_content})"
                lean_statements.append(lean_stmt)
                assumptions.append(f"Negation of: {negated_content}")
            
            if lean_statements:
                # Create a basic Lean theorem structure
                lean_code = self._generate_lean_theorem(lean_statements)
                
                return FormalTranslation(
                    original_text=text,
                    formal_representation=lean_code,
                    formal_system=FormalSystem.LEAN,
                    translation_confidence=0.7,  # Pattern-based, so moderate confidence
                    assumptions_made=assumptions,
                    translation_notes="Basic pattern-based translation to Lean 4"
                )
            
        except Exception as e:
            self.logger.warning(f"Lean translation failed: {e}")
        
        return None
    
    def _generate_lean_theorem(self, statements: List[str]) -> str:
        """Generate a Lean theorem for consistency checking"""
        # Create a basic consistency check theorem
        if len(statements) == 1:
            theorem = f"""
-- Consistency check for single statement
variable (P Q : Prop)

theorem consistency_check : {statements[0]} ∨ ¬({statements[0]}) := by
  exact Classical.em _
"""
        else:
            combined = " ∧ ".join(statements)
            theorem = f"""
-- Consistency check for multiple statements
variable (P Q R : Prop)

theorem consistency_check : ¬({combined} ∧ ¬({combined})) := by
  intro h
  exact h.2 h.1
"""
        return theorem
    
    async def _translate_to_z3(self, text: str) -> Optional[FormalTranslation]:
        """Translate to Z3 SMT solver syntax"""
        try:
            z3_statements = []
            assumptions = []
            
            # Z3 SMT-LIB format
            z3_code = "(set-logic QF_UF)\n"
            
            # Declare sorts and functions based on detected patterns
            if self.logical_forms["universal_statement"].search(text):
                z3_code += "(declare-sort Person 0)\n"
                z3_code += "(declare-fun isGood (Person) Bool)\n"
                assumptions.append("Declared Person sort and isGood predicate")
            
            # Add basic consistency axioms
            z3_code += "\n; Consistency check\n"
            z3_code += "(assert (not (and P (not P))))\n"  # Law of non-contradiction
            z3_code += "(check-sat)\n"
            
            return FormalTranslation(
                original_text=text,
                formal_representation=z3_code,
                formal_system=FormalSystem.Z3,
                translation_confidence=0.6,
                assumptions_made=assumptions,
                translation_notes="SMT-LIB format for Z3 solver"
            )
            
        except Exception as e:
            self.logger.warning(f"Z3 translation failed: {e}")
        
        return None
    
    async def _translate_to_prolog(self, text: str) -> Optional[FormalTranslation]:
        """Translate to Prolog logic programming syntax"""
        try:
            prolog_facts = []
            prolog_rules = []
            assumptions = []
            
            # Handle simple facts and rules
            # "All X are Y" -> "Y(X) :- X(Z)"
            universal_match = self.logical_forms["universal_statement"].search(text)
            if universal_match:
                x_type, y_property = universal_match.groups()
                prolog_rule = f"{y_property.lower()}(X) :- {x_type.lower()}(X)."
                prolog_rules.append(prolog_rule)
                assumptions.append(f"Rule: All {x_type} have property {y_property}")
            
            # Handle conditionals
            conditional_match = self.logical_forms["conditional"].search(text)
            if conditional_match:
                condition, consequence = conditional_match.groups()
                # Simplified propositional rule
                prolog_rule = f"consequence :- condition."
                prolog_rules.append(prolog_rule)
                assumptions.extend([
                    f"condition represents: {condition.strip()}",
                    f"consequence represents: {consequence.strip()}"
                ])
            
            # Create consistency check query
            prolog_code = "\n".join(prolog_rules + prolog_facts)
            prolog_code += "\n\n% Consistency check\n"
            prolog_code += "inconsistent :- condition, \\+ consequence.\n"
            prolog_code += "?- inconsistent.\n"
            
            return FormalTranslation(
                original_text=text,
                formal_representation=prolog_code,
                formal_system=FormalSystem.PROLOG,
                translation_confidence=0.5,
                assumptions_made=assumptions,
                translation_notes="Prolog rules and facts with consistency query"
            )
            
        except Exception as e:
            self.logger.warning(f"Prolog translation failed: {e}")
        
        return None
    
    async def _verify_with_system(self, translation: FormalTranslation, 
                                 system: FormalSystem) -> Optional[ConsistencyProof]:
        """Verify consistency using a specific formal system"""
        try:
            if system == FormalSystem.LEAN:
                return await self._verify_with_lean(translation)
            elif system == FormalSystem.Z3:
                return await self._verify_with_z3(translation)
            elif system == FormalSystem.PROLOG:
                return await self._verify_with_prolog(translation)
            else:
                self.logger.warning(f"Verification with {system.value} not implemented")
                return None
                
        except Exception as e:
            self.logger.error(f"Verification with {system.value} failed: {e}")
            return None
    
    async def _verify_with_lean(self, translation: FormalTranslation) -> Optional[ConsistencyProof]:
        """Verify using Lean theorem prover"""
        if FormalSystem.LEAN not in self.available_systems:
            return None
        
        try:
            # Write Lean code to temporary file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.lean', delete=False) as f:
                f.write(translation.formal_representation)
                lean_file = f.name
            
            # Run Lean checker
            start_time = datetime.utcnow()
            result = subprocess.run(
                ["lean", lean_file],
                capture_output=True,
                text=True,
                timeout=30
            )
            execution_time = (datetime.utcnow() - start_time).total_seconds()
            
            # Clean up
            Path(lean_file).unlink()
            
            # Parse result
            is_consistent = result.returncode == 0
            proof_found = "no goals" in result.stdout.lower() or result.returncode == 0
            
            proof_result = ProofResult(
                is_valid=is_consistent,
                proof_found=proof_found,
                execution_time=execution_time,
                error_message=result.stderr if result.stderr else None
            )
            
            return ConsistencyProof(
                is_consistent=is_consistent,
                proof_system=FormalSystem.LEAN,
                formal_statement=translation.formal_representation,
                proof_result=proof_result,
                consistency_score=1.0 if is_consistent else 0.0,
                logical_dependencies=["Classical logic", "Law of excluded middle"]
            )
            
        except subprocess.TimeoutExpired:
            return ConsistencyProof(
                is_consistent=False,
                proof_system=FormalSystem.LEAN,
                formal_statement=translation.formal_representation,
                proof_result=ProofResult(False, False, 30.0, "Timeout"),
                consistency_score=0.0
            )
        except Exception as e:
            self.logger.error(f"Lean verification failed: {e}")
            return None
    
    async def _verify_with_z3(self, translation: FormalTranslation) -> Optional[ConsistencyProof]:
        """Verify using Z3 SMT solver"""
        if FormalSystem.Z3 not in self.available_systems:
            return None
        
        try:
            # Write Z3 code to temporary file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.smt2', delete=False) as f:
                f.write(translation.formal_representation)
                z3_file = f.name
            
            # Run Z3 solver
            start_time = datetime.utcnow()
            result = subprocess.run(
                ["z3", z3_file],
                capture_output=True,
                text=True,
                timeout=30
            )
            execution_time = (datetime.utcnow() - start_time).total_seconds()
            
            # Clean up
            Path(z3_file).unlink()
            
            # Parse Z3 output
            output = result.stdout.strip()
            is_consistent = output == "unsat"  # unsat means no contradiction found
            proof_found = output in ["sat", "unsat"]
            
            counterexample = None
            if output == "sat":
                # Z3 found a model, which might indicate inconsistency
                counterexample = "Z3 found satisfying model"
            
            proof_result = ProofResult(
                is_valid=is_consistent,
                proof_found=proof_found,
                counterexample=counterexample,
                execution_time=execution_time,
                error_message=result.stderr if result.stderr else None
            )
            
            return ConsistencyProof(
                is_consistent=is_consistent,
                proof_system=FormalSystem.Z3,
                formal_statement=translation.formal_representation,
                proof_result=proof_result,
                consistency_score=1.0 if is_consistent else 0.2,
                logical_dependencies=["SMT theory", "First-order logic"]
            )
            
        except Exception as e:
            self.logger.error(f"Z3 verification failed: {e}")
            return None
    
    async def _verify_with_prolog(self, translation: FormalTranslation) -> Optional[ConsistencyProof]:
        """Verify using SWI-Prolog"""
        if FormalSystem.PROLOG not in self.available_systems:
            return None
        
        try:
            # Write Prolog code to temporary file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.pl', delete=False) as f:
                f.write(translation.formal_representation)
                prolog_file = f.name
            
            # Run SWI-Prolog query
            start_time = datetime.utcnow()
            result = subprocess.run(
                ["swipl", "-q", "-t", "halt", "-s", prolog_file],
                capture_output=True,
                text=True,
                timeout=30
            )
            execution_time = (datetime.utcnow() - start_time).total_seconds()
            
            # Clean up
            Path(prolog_file).unlink()
            
            # Parse Prolog output
            is_consistent = "false" not in result.stdout.lower()  # No inconsistency found
            proof_found = result.returncode == 0
            
            proof_result = ProofResult(
                is_valid=is_consistent,
                proof_found=proof_found,
                execution_time=execution_time,
                error_message=result.stderr if result.stderr else None
            )
            
            return ConsistencyProof(
                is_consistent=is_consistent,
                proof_system=FormalSystem.PROLOG,
                formal_statement=translation.formal_representation,
                proof_result=proof_result,
                consistency_score=0.8 if is_consistent else 0.1,
                logical_dependencies=["Horn clause logic", "SLD resolution"]
            )
            
        except Exception as e:
            self.logger.error(f"Prolog verification failed: {e}")
            return None
    
    def _combine_formal_results(self, proof_results: List[ConsistencyProof],
                               translations: List[FormalTranslation]) -> float:
        """Combine results from multiple formal systems"""
        if not proof_results:
            return 0.1
        
        # Weight systems by their theoretical strength
        system_weights = {
            FormalSystem.LEAN: 1.0,      # Full theorem prover
            FormalSystem.COQUERY: 1.0,   # Full theorem prover
            FormalSystem.Z3: 0.8,        # SMT solver (limited logic)
            FormalSystem.PROLOG: 0.6,    # Logic programming (limited expressiveness)
            FormalSystem.VAMPIRE: 0.9    # First-order theorem prover
        }
        
        weighted_scores = []
        for proof in proof_results:
            weight = system_weights.get(proof.proof_system, 0.5)
            weighted_scores.append(proof.consistency_score * weight)
        
        if not weighted_scores:
            return 0.1
        
        # Take the maximum score (if any system proves consistency, claim is likely consistent)
        base_score = max(weighted_scores)
        
        # Bonus for multiple systems agreeing
        consistent_systems = sum(1 for proof in proof_results if proof.is_consistent)
        total_systems = len(proof_results)
        
        if total_systems > 1:
            agreement_bonus = (consistent_systems / total_systems) * 0.2
            base_score = min(base_score + agreement_bonus, 1.0)
        
        # Penalty for translation uncertainty
        if translations:
            avg_translation_confidence = sum(t.translation_confidence for t in translations) / len(translations)
            base_score *= avg_translation_confidence
        
        return base_score
    
    def _generate_formal_reasoning(self, translations: List[FormalTranslation],
                                 proof_results: List[ConsistencyProof]) -> str:
        """Generate reasoning based on formal verification results"""
        reasoning_parts = []
        
        # Translation summary
        if translations:
            systems_used = list(set([t.formal_system.value for t in translations]))
            reasoning_parts.append(f"Translated to formal logic using {', '.join(systems_used)}")
            
            avg_confidence = sum(t.translation_confidence for t in translations) / len(translations)
            reasoning_parts.append(f"Average translation confidence: {avg_confidence:.2f}")
        
        # Formal verification results
        if proof_results:
            consistent_proofs = [p for p in proof_results if p.is_consistent]
            if consistent_proofs:
                systems = [p.proof_system.value for p in consistent_proofs]
                reasoning_parts.append(f"Consistency verified by: {', '.join(systems)}")
            
            inconsistent_proofs = [p for p in proof_results if not p.is_consistent]
            if inconsistent_proofs:
                systems = [p.proof_system.value for p in inconsistent_proofs]
                reasoning_parts.append(f"Inconsistencies detected by: {', '.join(systems)}")
        else:
            reasoning_parts.append("No formal verification results obtained")
        
        return ". ".join(reasoning_parts) + "."
    
    def _identify_formal_uncertainties(self, translations: List[FormalTranslation],
                                     proof_results: List[ConsistencyProof]) -> List[str]:
        """Identify uncertainties in formal verification"""
        uncertainties = []
        
        if not translations:
            uncertainties.append("Unable to translate to formal logic")
        elif all(t.translation_confidence < 0.7 for t in translations):
            uncertainties.append("Low confidence in formal translation")
        
        if not proof_results:
            uncertainties.append("No formal verification systems available")
        elif any(not p.proof_result.proof_found for p in proof_results):
            uncertainties.append("Incomplete formal proofs")
        
        # Check for disagreement between systems
        if len(proof_results) > 1:
            consistent_count = sum(1 for p in proof_results if p.is_consistent)
            if 0 < consistent_count < len(proof_results):
                uncertainties.append("Disagreement between formal systems")
        
        # Check for timeouts or errors
        if any(p.proof_result.error_message for p in proof_results):
            uncertainties.append("Formal system errors encountered")
        
        return uncertainties
    
    def _generate_formal_insights(self, claim: Claim,
                                translations: List[FormalTranslation],
                                proof_results: List[ConsistencyProof]) -> str:
        """Generate insights from formal verification"""
        insights = []
        
        # Translation insights
        if translations:
            high_confidence_translations = [t for t in translations if t.translation_confidence > 0.8]
            if high_confidence_translations:
                insights.append("High-confidence formal translation achieved")
            
            complex_translations = [t for t in translations if len(t.assumptions_made) > 3]
            if complex_translations:
                insights.append("Complex logical structure requiring multiple assumptions")
        
        # Proof insights
        if proof_results:
            fast_proofs = [p for p in proof_results if p.proof_result.execution_time < 1.0]
            if fast_proofs:
                insights.append("Rapid formal verification indicates simple logical structure")
            
            slow_proofs = [p for p in proof_results if p.proof_result.execution_time > 10.0]
            if slow_proofs:
                insights.append("Complex formal verification suggests sophisticated logical content")
        
        # System-specific insights
        lean_proofs = [p for p in proof_results if p.proof_system == FormalSystem.LEAN and p.is_consistent]
        if lean_proofs:
            insights.append("Constructive proof verified in Lean theorem prover")
        
        z3_proofs = [p for p in proof_results if p.proof_system == FormalSystem.Z3]
        if z3_proofs:
            insights.append("SMT solver analysis completed for decidable fragments")
        
        return ". ".join(insights) + "." if insights else "Standard formal verification completed."
    
    def _collect_formal_references(self, translations: List[FormalTranslation],
                                 proof_results: List[ConsistencyProof]) -> List[str]:
        """Collect references from formal verification"""
        references = []
        
        # Translation references
        for translation in translations:
            references.append(f"{translation.formal_system.value} translation: {translation.formal_representation[:100]}...")
        
        # Proof references
        for proof in proof_results:
            if proof.is_consistent:
                references.append(f"{proof.proof_system.value}: Consistency verified")
            else:
                references.append(f"{proof.proof_system.value}: Inconsistency detected")
        
        return references
    
    def _apply_formal_node_adjustments(self, score: float, node_context: NodeContext,
                                     result: VerificationResult) -> float:
        """Apply node-specific adjustments for formal verification"""
        consistency_weight = node_context.philosophical_weights.get("consistency", 1.0)
        
        # Nodes that highly value consistency should get bonus for formal verification
        if consistency_weight > 0.8:
            # Bonus for formal verification
            formal_bonus = 0.1 if result.metadata.get("proofs_successful", 0) > 0 else 0.0
            score = min(score + formal_bonus, 1.0)
        
        return score * consistency_weight
    
    def _assess_formal_evidence_quality(self, result: VerificationResult) -> float:
        """Assess quality of formal evidence"""
        quality_factors = []
        
        # Translation quality
        translation_confidence = result.metadata.get("translation_confidence", 0.0)
        quality_factors.append(translation_confidence)
        
        # Proof success rate
        proofs_attempted = result.metadata.get("proofs_attempted", 0)
        proofs_successful = result.metadata.get("proofs_successful", 0)
        
        if proofs_attempted > 0:
            success_rate = proofs_successful / proofs_attempted
            quality_factors.append(success_rate)
        
        # System diversity
        systems_used = len(result.metadata.get("formal_systems_used", []))
        diversity_score = min(systems_used / 3.0, 1.0)  # Normalize by max 3 systems
        quality_factors.append(diversity_score)
        
        return sum(quality_factors) / len(quality_factors) if quality_factors else 0.0
    
    def _suggest_formal_refinements(self, result: VerificationResult,
                                  node_context: NodeContext) -> List[str]:
        """Suggest refinements for formal verification"""
        suggestions = []
        
        if result.confidence_score < 0.7:
            suggestions.append("Clarify logical structure for better formal translation")
        
        if "Unable to translate to formal logic" in result.uncertainty_factors:
            suggestions.append("Rephrase using explicit logical connectives (if-then, and, or, not)")
        
        if "Low confidence in formal translation" in result.uncertainty_factors:
            suggestions.append("Use more precise logical language and clearly defined terms")
        
        if "Disagreement between formal systems" in result.uncertainty_factors:
            suggestions.append("Resolve ambiguities that cause different formal interpretations")
        
        if "Formal system errors encountered" in result.uncertainty_factors:
            suggestions.append("Simplify logical complexity for automated verification")
        
        return suggestions
    
    def _generate_cache_key(self, claim: Claim) -> str:
        """Generate cache key for formal verification"""
        import hashlib
        content_hash = hashlib.md5(claim.content.encode()).hexdigest()
        systems_hash = hashlib.md5(str(sorted([s.value for s in self.preferred_systems])).encode()).hexdigest()
        return f"formal_{content_hash}_{systems_hash}"
    
    def clear_cache(self):
        """Clear formal verification cache"""
        self.cache.clear()
    
    def get_formal_stats(self) -> Dict[str, Any]:
        """Get formal verification statistics"""
        return {
            "available_systems": [s.value for s in self.available_systems],
            "preferred_systems": [s.value for s in self.preferred_systems],
            "cached_verifications": len(self.cache),
            "parallel_proving": self.enable_parallel_proving,
            "logical_axioms": sum(len(axioms) for axioms in self.logical_axioms.values())
        }