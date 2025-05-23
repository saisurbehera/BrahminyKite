"""
LLM + Formal Verification Integration

Combines Large Language Models with formal verification systems for:
1. Natural language → formal logic translation
2. Proof explanation and interpretation
3. Counterexample analysis and refinement
4. Multi-step reasoning verification
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum

from .consistency_formal import FormalConsistencyFramework, FormalTranslation, ConsistencyProof, FormalSystem
from .unified_base import UnifiedVerificationComponent
from ...consensus_types import Claim, VerificationResult, ConsensusProposal, NodeContext, ConsensusVerificationResult


class LLMRole(Enum):
    """Different roles LLMs play in formal verification"""
    TRANSLATOR = "translator"  # NL → Formal logic
    EXPLAINER = "explainer"   # Formal results → NL
    REFINER = "refiner"       # Improve translations
    VALIDATOR = "validator"   # Check translation quality
    REASONER = "reasoner"     # Multi-step reasoning


@dataclass
class LLMTranslation:
    """LLM-generated formal translation"""
    original_text: str
    formal_code: str
    target_system: FormalSystem
    confidence: float
    reasoning_steps: List[str] = field(default_factory=list)
    assumptions: List[str] = field(default_factory=list)
    uncertainty_notes: str = ""


@dataclass
class ProofExplanation:
    """LLM explanation of formal proof results"""
    formal_result: str
    natural_language_explanation: str
    key_insights: List[str] = field(default_factory=list)
    implications: List[str] = field(default_factory=list)
    confidence: float = 0.0


@dataclass
class ReasoningChain:
    """Multi-step reasoning chain verified formally"""
    steps: List[str] = field(default_factory=list)
    formal_proofs: List[ConsistencyProof] = field(default_factory=list)
    overall_validity: bool = False
    weak_links: List[int] = field(default_factory=list)  # Indices of weak reasoning steps


class LLMFormalIntegration(UnifiedVerificationComponent):
    """
    Integration of LLMs with formal verification systems
    """
    
    def __init__(self, llm_provider: str = "anthropic", model_name: str = "claude-3",
                 formal_framework: Optional[FormalConsistencyFramework] = None):
        self.llm_provider = llm_provider
        self.model_name = model_name
        self.formal_framework = formal_framework or FormalConsistencyFramework()
        self.cache = {}
        self.logger = logging.getLogger(__name__)
        
        # LLM-specific prompts for formal verification
        self._init_prompts()
        
        # Quality thresholds
        self.translation_confidence_threshold = 0.7
        self.proof_explanation_threshold = 0.8
    
    def _init_prompts(self):
        """Initialize LLM prompts for different formal verification tasks"""
        
        self.prompts = {
            "nl_to_lean": """
You are an expert in formal logic and the Lean theorem prover. Translate the following natural language claim into valid Lean 4 code.

Guidelines:
1. Use proper Lean 4 syntax
2. Define necessary types and predicates
3. Create a theorem that captures the logical structure
4. Include consistency checks where appropriate
5. Explain your translation choices

Natural language claim: "{claim}"

Provide your response in this format:
```lean
-- Your Lean 4 code here
```

Translation reasoning:
1. [Step 1 explanation]
2. [Step 2 explanation]
...

Assumptions made:
- [Assumption 1]
- [Assumption 2]
...

Confidence (0-100): [Your confidence in this translation]
""",

            "nl_to_z3": """
You are an expert in formal logic and SMT solving. Translate the following natural language claim into Z3 SMT-LIB format.

Guidelines:
1. Use proper SMT-LIB syntax
2. Declare appropriate sorts and functions
3. Add assertions that capture the logical structure
4. Include consistency checks
5. Use the most appropriate logic (QF_UF, LIA, etc.)

Natural language claim: "{claim}"

Provide your response in this format:
```smt2
; Your SMT-LIB code here
```

Translation reasoning:
1. [Step 1 explanation]
2. [Step 2 explanation]
...

Logic used: [QF_UF/LIA/etc.]
Confidence (0-100): [Your confidence in this translation]
""",

            "explain_proof": """
You are an expert in formal logic. Explain the following formal verification result in clear, natural language.

Formal system: {system}
Formal code: {formal_code}
Verification result: {result}
Execution details: {details}

Provide an explanation that covers:
1. What the formal code represents
2. What the verification result means
3. Key logical insights
4. Practical implications
5. Any limitations or caveats

Make it understandable to someone familiar with logic but not necessarily the specific formal system.
""",

            "refine_translation": """
You are an expert in formal logic. The following translation has been attempted but may need improvement.

Original claim: "{original}"
Current formal translation: {current_translation}
Target system: {system}
Issues identified: {issues}
Formal verification result: {verification_result}

Please provide an improved translation that addresses the identified issues.

Guidelines:
1. Fix any syntax errors
2. Improve logical accuracy
3. Handle edge cases better
4. Maintain the original meaning
5. Explain your improvements

Improved translation:
```{system_syntax}
-- Your improved code here
```

Improvements made:
1. [Improvement 1]
2. [Improvement 2]
...
""",

            "multi_step_reasoning": """
You are an expert in formal logic and reasoning. Analyze the following claim that may involve multiple reasoning steps.

Claim: "{claim}"

Tasks:
1. Break down the claim into individual logical steps
2. Identify the logical structure connecting these steps
3. Highlight any assumptions or gaps
4. Assess the overall logical validity

Provide your analysis in this format:

Reasoning steps:
1. [Step 1]
2. [Step 2]
...

Logical structure:
- [How steps connect]
- [Key logical operators]
- [Dependencies between steps]

Assumptions:
- [Implicit assumption 1]
- [Implicit assumption 2]
...

Potential issues:
- [Issue 1]
- [Issue 2]
...

Overall assessment: [Valid/Invalid/Uncertain with explanation]
"""
        }
    
    async def verify_individual(self, claim: Claim) -> VerificationResult:
        """Verify claim using LLM + formal verification integration"""
        try:
            # Check cache
            cache_key = self._generate_cache_key(claim)
            if cache_key in self.cache:
                return self.cache[cache_key]
            
            # Step 1: LLM-assisted multi-step reasoning analysis
            reasoning_analysis = await self._analyze_reasoning_structure(claim)
            
            # Step 2: LLM-generated formal translations (multiple systems)
            llm_translations = await self._generate_llm_translations(claim)
            
            # Step 3: Formal verification of translations
            formal_results = []
            for translation in llm_translations:
                formal_translation = FormalTranslation(
                    original_text=translation.original_text,
                    formal_representation=translation.formal_code,
                    formal_system=translation.target_system,
                    translation_confidence=translation.confidence,
                    assumptions_made=translation.assumptions
                )
                
                proof_result = await self.formal_framework._verify_with_system(
                    formal_translation, translation.target_system
                )
                if proof_result:
                    formal_results.append(proof_result)
            
            # Step 4: LLM explanation of formal results
            explanations = []
            for result in formal_results:
                explanation = await self._explain_formal_result(result)
                if explanation:
                    explanations.append(explanation)
            
            # Step 5: Iterative refinement if needed
            if formal_results and not all(r.is_consistent for r in formal_results):
                refined_translations = await self._refine_translations(
                    llm_translations, formal_results
                )
                
                # Re-verify refined translations
                for refined in refined_translations:
                    formal_translation = FormalTranslation(
                        original_text=refined.original_text,
                        formal_representation=refined.formal_code,
                        formal_system=refined.target_system,
                        translation_confidence=refined.confidence,
                        assumptions_made=refined.assumptions
                    )
                    
                    proof_result = await self.formal_framework._verify_with_system(
                        formal_translation, refined.target_system
                    )
                    if proof_result:
                        formal_results.append(proof_result)
            
            # Step 6: Combine LLM insights with formal verification
            overall_score = self._combine_llm_formal_results(
                reasoning_analysis, llm_translations, formal_results, explanations
            )
            
            reasoning = self._generate_integrated_reasoning(
                reasoning_analysis, llm_translations, formal_results, explanations
            )
            
            uncertainty_factors = self._identify_integrated_uncertainties(
                reasoning_analysis, llm_translations, formal_results
            )
            
            contextual_notes = self._generate_integrated_insights(
                claim, reasoning_analysis, explanations
            )
            
            result = VerificationResult(
                framework_name="llm_formal_integration",
                confidence_score=overall_score,
                reasoning=reasoning,
                evidence_references=self._collect_integrated_references(llm_translations, formal_results),
                uncertainty_factors=uncertainty_factors,
                contextual_notes=contextual_notes,
                metadata={
                    "llm_provider": self.llm_provider,
                    "model_name": self.model_name,
                    "llm_translations_generated": len(llm_translations),
                    "formal_verifications_completed": len(formal_results),
                    "explanations_generated": len(explanations),
                    "reasoning_steps_identified": len(reasoning_analysis.steps) if reasoning_analysis else 0,
                    "multi_step_reasoning": reasoning_analysis.overall_validity if reasoning_analysis else False,
                    "verification_timestamp": datetime.utcnow().isoformat()
                }
            )
            
            # Cache result
            self.cache[cache_key] = result
            return result
            
        except Exception as e:
            self.logger.error(f"LLM-formal integration failed: {e}")
            return VerificationResult(
                framework_name="llm_formal_integration",
                confidence_score=0.1,
                reasoning=f"LLM-formal verification failed: {str(e)}",
                evidence_references=[],
                uncertainty_factors=["Technical failure", "LLM or formal system unavailable"],
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
        
        # Apply node-specific adjustments for LLM+formal integration
        node_adjusted_score = individual_result.confidence_score * node_context.philosophical_weights.get("consistency", 1.0)
        
        return ConsensusVerificationResult(
            node_id=node_context.node_id,
            framework_name="llm_formal_integration",
            confidence_score=node_adjusted_score,
            reasoning=individual_result.reasoning,
            evidence_quality=self._assess_integrated_evidence_quality(individual_result),
            consensus_readiness=node_adjusted_score > 0.75,  # Higher threshold for complex verification
            suggested_refinements=self._suggest_integrated_refinements(individual_result, node_context),
            metadata={
                **individual_result.metadata,
                "integration_mode": True,
                "consensus_mode": True
            }
        )
    
    async def _analyze_reasoning_structure(self, claim: Claim) -> Optional[ReasoningChain]:
        """Use LLM to analyze multi-step reasoning structure"""
        try:
            # This would call the actual LLM API
            # For now, simulate the response
            llm_response = await self._call_llm(
                self.prompts["multi_step_reasoning"].format(claim=claim.content)
            )
            
            # Parse LLM response into structured reasoning chain
            reasoning_chain = self._parse_reasoning_analysis(llm_response, claim.content)
            
            return reasoning_chain
            
        except Exception as e:
            self.logger.warning(f"Reasoning structure analysis failed: {e}")
            return None
    
    async def _generate_llm_translations(self, claim: Claim) -> List[LLMTranslation]:
        """Generate formal logic translations using LLM"""
        translations = []
        
        # Generate translations for available formal systems
        available_systems = self.formal_framework.available_systems
        
        for system in available_systems:
            try:
                if system == FormalSystem.LEAN:
                    prompt = self.prompts["nl_to_lean"].format(claim=claim.content)
                elif system == FormalSystem.Z3:
                    prompt = self.prompts["nl_to_z3"].format(claim=claim.content)
                else:
                    continue  # Skip systems without LLM prompts
                
                # Call LLM for translation
                llm_response = await self._call_llm(prompt)
                
                # Parse LLM response
                translation = self._parse_translation_response(llm_response, system, claim.content)
                if translation:
                    translations.append(translation)
                    
            except Exception as e:
                self.logger.warning(f"LLM translation to {system.value} failed: {e}")
        
        return translations
    
    async def _explain_formal_result(self, formal_result: ConsistencyProof) -> Optional[ProofExplanation]:
        """Use LLM to explain formal verification results"""
        try:
            prompt = self.prompts["explain_proof"].format(
                system=formal_result.proof_system.value,
                formal_code=formal_result.formal_statement,
                result="Consistent" if formal_result.is_consistent else "Inconsistent",
                details=f"Score: {formal_result.consistency_score}, Time: {formal_result.proof_result.execution_time}s"
            )
            
            llm_response = await self._call_llm(prompt)
            
            # Parse explanation
            explanation = self._parse_explanation_response(llm_response, formal_result)
            
            return explanation
            
        except Exception as e:
            self.logger.warning(f"Proof explanation failed: {e}")
            return None
    
    async def _refine_translations(self, original_translations: List[LLMTranslation],
                                 formal_results: List[ConsistencyProof]) -> List[LLMTranslation]:
        """Use LLM to refine translations based on formal verification feedback"""
        refined_translations = []
        
        for translation in original_translations:
            # Find corresponding formal result
            formal_result = next(
                (r for r in formal_results if r.proof_system == translation.target_system),
                None
            )
            
            if formal_result and not formal_result.is_consistent:
                try:
                    # Generate refinement prompt
                    issues = []
                    if formal_result.proof_result.error_message:
                        issues.append(f"Error: {formal_result.proof_result.error_message}")
                    if not formal_result.proof_result.proof_found:
                        issues.append("No proof found")
                    
                    prompt = self.prompts["refine_translation"].format(
                        original=translation.original_text,
                        current_translation=translation.formal_code,
                        system=translation.target_system.value,
                        issues="; ".join(issues),
                        verification_result=f"Inconsistent (score: {formal_result.consistency_score})"
                    )
                    
                    llm_response = await self._call_llm(prompt)
                    
                    # Parse refined translation
                    refined = self._parse_refined_translation(llm_response, translation)
                    if refined:
                        refined_translations.append(refined)
                        
                except Exception as e:
                    self.logger.warning(f"Translation refinement failed: {e}")
        
        return refined_translations
    
    async def _call_llm(self, prompt: str) -> str:
        """Call LLM API (placeholder for actual implementation)"""
        # In a real implementation, this would call the actual LLM API
        # For now, simulate realistic responses
        
        if "translate" in prompt.lower() and "lean" in prompt.lower():
            return """
```lean
-- Formal representation of the claim
variable (P Q : Prop)

theorem claim_consistency : P → Q := by
  intro h
  sorry  -- Proof would be completed here
```

Translation reasoning:
1. Identified conditional structure "if P then Q"
2. Represented as implication in propositional logic
3. Created theorem to check consistency

Assumptions made:
- P and Q represent well-defined propositions
- Classical logic applies

Confidence (0-100): 75
"""
        elif "explain" in prompt.lower():
            return """
The formal verification result shows that the claim has a consistent logical structure. 
The theorem prover was able to verify that the conditional statement "if P then Q" 
is logically sound within the given framework.

Key insights:
- The logical structure is valid
- No contradictions were detected
- The implication relationship is properly formed

Practical implications:
- The reasoning in the original claim follows sound logical principles
- The argument structure supports the conclusion
- No logical fallacies were identified in the formal representation
"""
        else:
            return """
Reasoning steps:
1. Initial premise identification
2. Logical connector analysis  
3. Conclusion validation

Logical structure:
- Conditional reasoning pattern identified
- Proper premise-conclusion relationship
- Valid logical operators used

Overall assessment: Valid logical structure with minor uncertainty about implicit assumptions
"""
    
    def _parse_reasoning_analysis(self, llm_response: str, original_claim: str) -> ReasoningChain:
        """Parse LLM reasoning analysis into structured format"""
        # Simple parsing - in real implementation would be more sophisticated
        lines = llm_response.split('\n')
        
        steps = []
        overall_validity = False
        
        # Extract reasoning steps
        in_steps_section = False
        for line in lines:
            if "reasoning steps:" in line.lower():
                in_steps_section = True
                continue
            elif in_steps_section and line.strip().startswith(('1.', '2.', '3.', '4.', '5.')):
                step = line.strip()[2:].strip()  # Remove number and whitespace
                steps.append(step)
            elif "overall assessment:" in line.lower():
                overall_validity = "valid" in line.lower() and "invalid" not in line.lower()
                break
        
        return ReasoningChain(
            steps=steps,
            overall_validity=overall_validity,
            formal_proofs=[],  # Will be populated later
            weak_links=[]
        )
    
    def _parse_translation_response(self, llm_response: str, system: FormalSystem, 
                                  original_text: str) -> Optional[LLMTranslation]:
        """Parse LLM translation response"""
        try:
            # Extract formal code
            if system == FormalSystem.LEAN:
                code_match = re.search(r'```lean\n(.*?)\n```', llm_response, re.DOTALL)
            elif system == FormalSystem.Z3:
                code_match = re.search(r'```smt2\n(.*?)\n```', llm_response, re.DOTALL)
            else:
                return None
            
            if not code_match:
                return None
            
            formal_code = code_match.group(1).strip()
            
            # Extract confidence
            confidence_match = re.search(r'confidence.*?(\d+)', llm_response, re.IGNORECASE)
            confidence = float(confidence_match.group(1)) / 100.0 if confidence_match else 0.5
            
            # Extract reasoning steps
            reasoning_steps = []
            reasoning_section = re.search(r'translation reasoning:(.*?)(?:assumptions|confidence|$)', 
                                        llm_response, re.DOTALL | re.IGNORECASE)
            if reasoning_section:
                for line in reasoning_section.group(1).split('\n'):
                    line = line.strip()
                    if line and line.startswith(('1.', '2.', '3.', '4.', '5.')):
                        reasoning_steps.append(line[2:].strip())
            
            # Extract assumptions
            assumptions = []
            assumptions_section = re.search(r'assumptions made:(.*?)(?:confidence|$)', 
                                          llm_response, re.DOTALL | re.IGNORECASE)
            if assumptions_section:
                for line in assumptions_section.group(1).split('\n'):
                    line = line.strip()
                    if line and line.startswith('-'):
                        assumptions.append(line[1:].strip())
            
            return LLMTranslation(
                original_text=original_text,
                formal_code=formal_code,
                target_system=system,
                confidence=confidence,
                reasoning_steps=reasoning_steps,
                assumptions=assumptions
            )
            
        except Exception as e:
            self.logger.warning(f"Failed to parse translation response: {e}")
            return None
    
    def _parse_explanation_response(self, llm_response: str, 
                                  formal_result: ConsistencyProof) -> ProofExplanation:
        """Parse LLM explanation response"""
        # Extract key insights
        insights = []
        insights_section = re.search(r'key insights:(.*?)(?:practical implications|$)', 
                                   llm_response, re.DOTALL | re.IGNORECASE)
        if insights_section:
            for line in insights_section.group(1).split('\n'):
                line = line.strip()
                if line and line.startswith('-'):
                    insights.append(line[1:].strip())
        
        # Extract implications
        implications = []
        implications_section = re.search(r'practical implications:(.*?)$', 
                                       llm_response, re.DOTALL | re.IGNORECASE)
        if implications_section:
            for line in implications_section.group(1).split('\n'):
                line = line.strip()
                if line and line.startswith('-'):
                    implications.append(line[1:].strip())
        
        return ProofExplanation(
            formal_result=str(formal_result.is_consistent),
            natural_language_explanation=llm_response[:500],  # First 500 chars
            key_insights=insights,
            implications=implications,
            confidence=0.8  # Default confidence for explanations
        )
    
    def _parse_refined_translation(self, llm_response: str, 
                                 original_translation: LLMTranslation) -> Optional[LLMTranslation]:
        """Parse refined translation from LLM response"""
        # Similar to _parse_translation_response but for refinements
        # This would extract the improved code and explanations
        return None  # Placeholder
    
    def _combine_llm_formal_results(self, reasoning_analysis: Optional[ReasoningChain],
                                   llm_translations: List[LLMTranslation],
                                   formal_results: List[ConsistencyProof],
                                   explanations: List[ProofExplanation]) -> float:
        """Combine LLM insights with formal verification results"""
        scores = []
        weights = []
        
        # Reasoning structure score
        if reasoning_analysis and reasoning_analysis.overall_validity:
            scores.append(0.8)
            weights.append(0.2)
        elif reasoning_analysis:
            scores.append(0.4)
            weights.append(0.2)
        
        # Translation quality score
        if llm_translations:
            avg_translation_confidence = sum(t.confidence for t in llm_translations) / len(llm_translations)
            scores.append(avg_translation_confidence)
            weights.append(0.3)
        
        # Formal verification score
        if formal_results:
            consistent_results = [r for r in formal_results if r.is_consistent]
            formal_score = len(consistent_results) / len(formal_results)
            scores.append(formal_score)
            weights.append(0.4)
        
        # Explanation confidence
        if explanations:
            avg_explanation_confidence = sum(e.confidence for e in explanations) / len(explanations)
            scores.append(avg_explanation_confidence)
            weights.append(0.1)
        
        if not scores:
            return 0.1
        
        return sum(s * w for s, w in zip(scores, weights)) / sum(weights)
    
    def _generate_integrated_reasoning(self, reasoning_analysis: Optional[ReasoningChain],
                                     llm_translations: List[LLMTranslation],
                                     formal_results: List[ConsistencyProof],
                                     explanations: List[ProofExplanation]) -> str:
        """Generate integrated reasoning combining LLM and formal insights"""
        parts = []
        
        # LLM reasoning analysis
        if reasoning_analysis:
            if reasoning_analysis.overall_validity:
                parts.append(f"LLM analysis identified {len(reasoning_analysis.steps)} valid reasoning steps")
            else:
                parts.append(f"LLM analysis found potential issues in {len(reasoning_analysis.steps)} reasoning steps")
        
        # Translation summary
        if llm_translations:
            systems = [t.target_system.value for t in llm_translations]
            avg_conf = sum(t.confidence for t in llm_translations) / len(llm_translations)
            parts.append(f"Generated formal translations for {', '.join(systems)} with {avg_conf:.2f} average confidence")
        
        # Formal verification summary
        if formal_results:
            consistent_count = sum(1 for r in formal_results if r.is_consistent)
            parts.append(f"Formal verification: {consistent_count}/{len(formal_results)} systems found consistency")
        
        # Explanation insights
        if explanations:
            all_insights = []
            for exp in explanations:
                all_insights.extend(exp.key_insights)
            if all_insights:
                parts.append(f"Key insights: {'; '.join(all_insights[:2])}")
        
        return ". ".join(parts) + "."
    
    def _identify_integrated_uncertainties(self, reasoning_analysis: Optional[ReasoningChain],
                                         llm_translations: List[LLMTranslation],
                                         formal_results: List[ConsistencyProof]) -> List[str]:
        """Identify uncertainties from integrated analysis"""
        uncertainties = []
        
        # LLM reasoning uncertainties
        if reasoning_analysis and not reasoning_analysis.overall_validity:
            uncertainties.append("LLM identified potential reasoning issues")
        
        if reasoning_analysis and reasoning_analysis.weak_links:
            uncertainties.append("Weak logical connections detected")
        
        # Translation uncertainties
        if llm_translations:
            low_confidence = [t for t in llm_translations if t.confidence < self.translation_confidence_threshold]
            if low_confidence:
                uncertainties.append("Low confidence in formal translations")
        
        # Formal verification uncertainties
        if formal_results:
            inconsistent_results = [r for r in formal_results if not r.is_consistent]
            if inconsistent_results:
                systems = [r.proof_system.value for r in inconsistent_results]
                uncertainties.append(f"Inconsistencies found by: {', '.join(systems)}")
        
        return uncertainties
    
    def _generate_integrated_insights(self, claim: Claim,
                                    reasoning_analysis: Optional[ReasoningChain],
                                    explanations: List[ProofExplanation]) -> str:
        """Generate insights from integrated LLM+formal analysis"""
        insights = []
        
        # Reasoning complexity
        if reasoning_analysis and len(reasoning_analysis.steps) > 3:
            insights.append("Complex multi-step reasoning detected requiring careful verification")
        
        # Translation challenges
        if explanations:
            for exp in explanations:
                if exp.implications:
                    insights.append(f"Formal verification implications: {exp.implications[0]}")
                    break
        
        # Integration benefits
        insights.append("LLM-assisted translation enabled formal verification of natural language claim")
        
        return ". ".join(insights) + "."
    
    def _collect_integrated_references(self, llm_translations: List[LLMTranslation],
                                     formal_results: List[ConsistencyProof]) -> List[str]:
        """Collect references from integrated analysis"""
        references = []
        
        # LLM translation references
        for translation in llm_translations:
            references.append(f"{translation.target_system.value} translation (conf: {translation.confidence:.2f})")
        
        # Formal verification references
        for result in formal_results:
            status = "Consistent" if result.is_consistent else "Inconsistent"
            references.append(f"{result.proof_system.value}: {status}")
        
        return references
    
    def _assess_integrated_evidence_quality(self, result: VerificationResult) -> float:
        """Assess quality of integrated evidence"""
        quality_factors = []
        
        # LLM translation quality
        translations_generated = result.metadata.get("llm_translations_generated", 0)
        if translations_generated > 0:
            quality_factors.append(min(translations_generated / 2.0, 1.0))
        
        # Formal verification quality
        verifications_completed = result.metadata.get("formal_verifications_completed", 0)
        if verifications_completed > 0:
            quality_factors.append(min(verifications_completed / 2.0, 1.0))
        
        # Multi-step reasoning
        multi_step_reasoning = result.metadata.get("multi_step_reasoning", False)
        quality_factors.append(0.8 if multi_step_reasoning else 0.5)
        
        return sum(quality_factors) / len(quality_factors) if quality_factors else 0.0
    
    def _suggest_integrated_refinements(self, result: VerificationResult,
                                      node_context: NodeContext) -> List[str]:
        """Suggest refinements for integrated verification"""
        suggestions = []
        
        if result.confidence_score < 0.7:
            suggestions.append("Clarify logical structure for better LLM translation")
        
        if "Low confidence in formal translations" in result.uncertainty_factors:
            suggestions.append("Use more precise logical language and standard terminology")
        
        if "LLM identified potential reasoning issues" in result.uncertainty_factors:
            suggestions.append("Address logical gaps or strengthen reasoning connections")
        
        return suggestions
    
    def _generate_cache_key(self, claim: Claim) -> str:
        """Generate cache key for integrated verification"""
        import hashlib
        content_hash = hashlib.md5(claim.content.encode()).hexdigest()
        model_hash = hashlib.md5(f"{self.llm_provider}_{self.model_name}".encode()).hexdigest()
        return f"llm_formal_{content_hash}_{model_hash}"
    
    def clear_cache(self):
        """Clear verification cache"""
        self.cache.clear()
        self.formal_framework.clear_cache()
    
    def get_integration_stats(self) -> Dict[str, Any]:
        """Get integration statistics"""
        return {
            "llm_provider": self.llm_provider,
            "model_name": self.model_name,
            "cached_verifications": len(self.cache),
            "formal_systems_available": len(self.formal_framework.available_systems),
            "translation_confidence_threshold": self.translation_confidence_threshold,
            "proof_explanation_threshold": self.proof_explanation_threshold
        }