"""
Core Ideal Verifier System

Main orchestrator that integrates all components, frameworks, and systems
for comprehensive multi-framework verification.
"""

import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Any, Optional
import json

from .frameworks import VerificationFramework, VerificationResult, Claim, Domain
from .components import (
    EmpiricalVerifier, ContextualVerifier, ConsistencyVerifier,
    PowerDynamicsVerifier, UtilityVerifier, EvolutionaryVerifier
)
from .meta import MetaVerificationSystem
from .systems import DebateSystem


class IdealVerifier:
    """
    Main verifier system integrating all components and philosophical principles
    
    This system implements the philosophical foundations from philophicalbasis.md:
    - Multi-framework integration
    - Value pluralism acknowledgment
    - Reflective equilibrium
    - Power dynamics awareness
    - Adaptive learning
    """
    
    def __init__(self, enable_debate: bool = True, enable_parallel: bool = True):
        # Initialize all verification components
        self.components = {
            "empirical": EmpiricalVerifier(),
            "contextual": ContextualVerifier(),
            "consistency": ConsistencyVerifier(),
            "power_dynamics": PowerDynamicsVerifier(),
            "utility": UtilityVerifier(),
            "evolutionary": EvolutionaryVerifier()
        }
        
        # Initialize meta-systems
        self.meta_verifier = MetaVerificationSystem()
        self.debate_system = DebateSystem()
        
        # Configuration
        self.enable_debate = enable_debate
        self.enable_parallel = enable_parallel
        self.max_workers = min(len(self.components), 6)
        self.verification_timeout = 30  # seconds
        
        # Logging
        self.logger = logging.getLogger(__name__)
        self.verification_history = []
    
    def verify(self, claim: Claim) -> Dict[str, Any]:
        """
        Comprehensive verification of a claim using all components and frameworks
        
        Args:
            claim: The claim to verify
            
        Returns:
            Dict containing verification results, explanations, and metadata
        """
        self.logger.info(f"Starting verification for claim: {claim.content[:100]}...")
        
        try:
            # Step 1: Run all verification components
            component_results = self._run_verification_components(claim)
            
            if not component_results:
                return self._create_error_response("No verification components completed successfully")
            
            # Step 2: Identify applicable frameworks
            applicable_frameworks = self._identify_applicable_frameworks()
            
            # Step 3: Conduct adversarial debate if enabled
            debate_result = None
            if self.enable_debate and len(applicable_frameworks) > 1:
                initial_scores = self._extract_framework_scores(component_results)
                debate_result = self.debate_system.conduct_debate(claim, applicable_frameworks, initial_scores)
            
            # Step 4: Meta-verification to resolve conflicts
            meta_result = self.meta_verifier.resolve_conflicts(component_results, claim)
            
            # Step 5: Calculate final verification score
            final_score = self._calculate_final_score(meta_result, debate_result)
            
            # Step 6: Generate comprehensive explanation
            explanation = self._generate_explanation(claim, component_results, meta_result, debate_result)
            
            # Step 7: Create final verification response
            verification_response = {
                "claim": claim.content,
                "domain": claim.domain.value,
                "final_score": final_score,
                "confidence_interval": meta_result.confidence_interval,
                "dominant_framework": meta_result.framework.value,
                "meta_result": self._serialize_verification_result(meta_result),
                "component_results": {
                    name: self._serialize_verification_result(result) 
                    for name, result in zip(self.components.keys(), component_results)
                },
                "debate_result": debate_result,
                "explanation": explanation,
                "metadata": {
                    "verification_timestamp": str(np.datetime64('now')),
                    "components_used": len(component_results),
                    "frameworks_considered": len(applicable_frameworks),
                    "debate_conducted": debate_result is not None,
                    "uncertainty_level": meta_result.metadata.get("uncertainty_level", 0.0),
                    "consensus_level": meta_result.metadata.get("consensus_level", 0.0)
                }
            }
            
            # Step 8: Record for learning
            self._record_verification(claim, verification_response)
            
            return verification_response
            
        except Exception as e:
            self.logger.error(f"Verification failed: {str(e)}")
            return self._create_error_response(f"Verification error: {str(e)}")
    
    def _run_verification_components(self, claim: Claim) -> List[VerificationResult]:
        """Run all verification components on the claim"""
        if self.enable_parallel:
            return self._run_parallel_verification(claim)
        else:
            return self._run_sequential_verification(claim)
    
    def _run_parallel_verification(self, claim: Claim) -> List[VerificationResult]:
        """Run verification components in parallel"""
        results = []
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all component verification tasks
            futures = {
                executor.submit(component.verify, claim): name 
                for name, component in self.components.items()
                if component.validate_claim(claim)
            }
            
            # Collect results
            for future in futures:
                component_name = futures[future]
                try:
                    result = future.result(timeout=self.verification_timeout)
                    results.append(result)
                    self.logger.debug(f"Component {component_name} completed: score={result.score:.3f}")
                except Exception as e:
                    self.logger.error(f"Component {component_name} failed: {e}")
        
        return results
    
    def _run_sequential_verification(self, claim: Claim) -> List[VerificationResult]:
        """Run verification components sequentially"""
        results = []
        
        for name, component in self.components.items():
            if not component.validate_claim(claim):
                continue
                
            try:
                result = component.verify(claim)
                results.append(result)
                self.logger.debug(f"Component {name} completed: score={result.score:.3f}")
            except Exception as e:
                self.logger.error(f"Component {name} failed: {e}")
        
        return results
    
    def _identify_applicable_frameworks(self) -> List[VerificationFramework]:
        """Identify all frameworks that could apply to verification"""
        applicable_frameworks = set()
        
        for component in self.components.values():
            applicable_frameworks.update(component.get_applicable_frameworks())
        
        return list(applicable_frameworks)
    
    def _extract_framework_scores(self, component_results: List[VerificationResult]) -> Dict[VerificationFramework, float]:
        """Extract framework scores from component results for debate initialization"""
        framework_scores = {}
        
        for result in component_results:
            if result.framework not in framework_scores:
                framework_scores[result.framework] = []
            framework_scores[result.framework].append(result.score)
        
        # Average scores for each framework
        return {
            framework: sum(scores) / len(scores) 
            for framework, scores in framework_scores.items()
        }
    
    def _calculate_final_score(self, meta_result: VerificationResult, 
                              debate_result: Optional[Dict[str, Any]]) -> float:
        """Calculate final verification score integrating all analyses"""
        base_score = meta_result.score
        
        if debate_result:
            # Adjust based on debate consensus
            consensus_level = debate_result.get("final_consensus", 0.5)
            debate_winner_score = max(debate_result.get("final_scores", {}).values()) if debate_result.get("final_scores") else 0.5
            
            # Weight meta-result more heavily, but consider debate outcome
            final_score = (base_score * 0.7 + 
                          debate_winner_score * 0.2 + 
                          consensus_level * 0.1)
        else:
            final_score = base_score
        
        return max(0.0, min(1.0, final_score))
    
    def _generate_explanation(self, claim: Claim, component_results: List[VerificationResult],
                             meta_result: VerificationResult, debate_result: Optional[Dict[str, Any]]) -> str:
        """Generate comprehensive human-readable explanation"""
        explanation_parts = [
            f"# Verification Analysis: {claim.domain.value.title()} Domain",
            f"**Final Score:** {meta_result.score:.3f} (using {meta_result.framework.value} framework)",
            ""
        ]
        
        # Component analysis
        explanation_parts.append("## Component Analysis")
        component_names = list(self.components.keys())
        
        for i, result in enumerate(component_results):
            comp_name = component_names[i] if i < len(component_names) else f"component_{i}"
            explanation_parts.append(f"- **{comp_name.title()}:** {result.score:.3f} ({result.framework.value})")
        
        # Meta-verification insights
        explanation_parts.extend([
            "",
            "## Meta-Verification Insights",
            f"- **Dominant Framework:** {meta_result.framework.value}",
            f"- **Uncertainty Level:** ±{meta_result.metadata.get('uncertainty_level', 0.0):.3f}",
            f"- **Consensus Level:** {meta_result.metadata.get('consensus_level', 0.0):.3f}"
        ])
        
        # Debate analysis
        if debate_result:
            explanation_parts.extend([
                "",
                "## Debate Analysis",
                f"- **Participants:** {', '.join(debate_result['session']['participants'])}",
                f"- **Rounds Conducted:** {len(debate_result['session']['rounds'])}",
                f"- **Final Consensus:** {debate_result['final_consensus']:.3f}",
                f"- **Debate Winner:** {debate_result['winner']}"
            ])
        
        # Philosophical considerations
        explanation_parts.extend([
            "",
            "## Philosophical Considerations",
            self._get_philosophical_insight(claim, meta_result),
            "",
            "## Confidence Assessment"
        ])
        
        # Confidence breakdown
        conf_lower, conf_upper = meta_result.confidence_interval
        uncertainty = conf_upper - conf_lower
        
        if uncertainty < 0.1:
            confidence_desc = "High confidence"
        elif uncertainty < 0.2:
            confidence_desc = "Moderate confidence"
        else:
            confidence_desc = "Lower confidence"
        
        explanation_parts.append(f"**{confidence_desc}** (uncertainty: ±{uncertainty:.3f})")
        
        return "\n".join(explanation_parts)
    
    def _get_philosophical_insight(self, claim: Claim, meta_result: VerificationResult) -> str:
        """Generate domain-appropriate philosophical insight"""
        framework = meta_result.framework
        domain = claim.domain
        
        insights = {
            (Domain.EMPIRICAL, VerificationFramework.POSITIVIST): 
                "This empirical claim is best verified through objective, measurable evidence and scientific methodology.",
            
            (Domain.AESTHETIC, VerificationFramework.INTERPRETIVIST): 
                "This aesthetic claim requires cultural context and interpretive understanding rather than objective measurement.",
            
            (Domain.ETHICAL, VerificationFramework.COHERENCE): 
                "This ethical claim is evaluated based on its internal consistency and coherence with established moral frameworks.",
            
            (Domain.SOCIAL, VerificationFramework.CONSTRUCTIVIST): 
                "This social claim must be understood within its power structures and social construction of meaning.",
            
            (Domain.LOGICAL, VerificationFramework.COHERENCE): 
                "This logical claim is verified through formal consistency and systematic coherence analysis."
        }
        
        specific_insight = insights.get((domain, framework))
        if specific_insight:
            return specific_insight
        
        # Fallback general insight
        return f"This {domain.value} claim is analyzed using {framework.value} verification principles, emphasizing appropriate evidence types and validation methods."
    
    def _serialize_verification_result(self, result: VerificationResult) -> Dict[str, Any]:
        """Convert VerificationResult to serializable dict"""
        return {
            "score": result.score,
            "framework": result.framework.value,
            "component": result.component,
            "evidence": result.evidence,
            "confidence_interval": result.confidence_interval,
            "metadata": result.metadata
        }
    
    def _create_error_response(self, error_message: str) -> Dict[str, Any]:
        """Create standardized error response"""
        return {
            "error": True,
            "message": error_message,
            "final_score": 0.0,
            "confidence_interval": (0.0, 0.0),
            "explanation": f"Verification failed: {error_message}"
        }
    
    def _record_verification(self, claim: Claim, verification_response: Dict[str, Any]):
        """Record verification for learning and analysis"""
        record = {
            "claim_content": claim.content,
            "domain": claim.domain.value,
            "final_score": verification_response["final_score"],
            "dominant_framework": verification_response["dominant_framework"],
            "components_used": verification_response["metadata"]["components_used"],
            "uncertainty": verification_response["metadata"]["uncertainty_level"],
            "timestamp": verification_response["metadata"]["verification_timestamp"]
        }
        
        self.verification_history.append(record)
        
        # Keep history manageable
        if len(self.verification_history) > 1000:
            self.verification_history = self.verification_history[-1000:]
    
    def learn_from_feedback(self, claim: Claim, verification_result: Dict[str, Any], 
                           human_feedback: Dict[str, Any]):
        """Learn and adapt from human feedback"""
        try:
            # Update meta-verifier weights
            if "framework_preference" in human_feedback:
                framework = VerificationFramework(human_feedback["framework_preference"])
                framework_feedback = {framework: human_feedback.get("weight_adjustment", 1.1)}
                self.meta_verifier.update_framework_weights(framework_feedback)
            
            # Update domain preferences
            if "domain_adjustments" in human_feedback:
                for domain_name, adjustments in human_feedback["domain_adjustments"].items():
                    domain = Domain(domain_name)
                    framework_adjustments = {
                        VerificationFramework(f): adj for f, adj in adjustments.items()
                    }
                    self.meta_verifier.update_domain_preferences(domain, framework_adjustments)
            
            # Update evolutionary component
            evolutionary_component = self.components.get("evolutionary")
            if evolutionary_component:
                evolutionary_component.update_verification_rules(human_feedback)
            
            self.logger.info("Updated verifier based on human feedback")
            
        except Exception as e:
            self.logger.error(f"Failed to process feedback: {e}")
    
    def get_verification_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics about verification performance"""
        stats = {
            "system_config": {
                "components_count": len(self.components),
                "debate_enabled": self.enable_debate,
                "parallel_enabled": self.enable_parallel,
                "max_workers": self.max_workers
            },
            "verification_history": {
                "total_verifications": len(self.verification_history),
                "recent_average_score": self._calculate_recent_average_score(),
                "domain_distribution": self._get_domain_distribution(),
                "framework_usage": self._get_framework_usage()
            },
            "component_statistics": {
                name: component.get_statistics() 
                for name, component in self.components.items()
                if hasattr(component, 'get_statistics')
            },
            "meta_verifier_stats": self.meta_verifier.get_statistics(),
            "debate_system_stats": self.debate_system.get_statistics()
        }
        
        return stats
    
    def _calculate_recent_average_score(self) -> float:
        """Calculate average score from recent verifications"""
        if not self.verification_history:
            return 0.0
        
        recent_records = self.verification_history[-50:]  # Last 50 verifications
        scores = [record["final_score"] for record in recent_records]
        return sum(scores) / len(scores)
    
    def _get_domain_distribution(self) -> Dict[str, int]:
        """Get distribution of claims by domain"""
        if not self.verification_history:
            return {}
        
        distribution = {}
        for record in self.verification_history:
            domain = record["domain"]
            distribution[domain] = distribution.get(domain, 0) + 1
        
        return distribution
    
    def _get_framework_usage(self) -> Dict[str, int]:
        """Get usage frequency of different frameworks"""
        if not self.verification_history:
            return {}
        
        usage = {}
        for record in self.verification_history:
            framework = record["dominant_framework"]
            usage[framework] = usage.get(framework, 0) + 1
        
        return usage
    
    def export_configuration(self) -> Dict[str, Any]:
        """Export current verifier configuration"""
        return {
            "framework_weights": {f.value: w for f, w in self.meta_verifier.framework_weights.items()},
            "domain_preferences": {
                d.value: {f.value: w for f, w in prefs.items()}
                for d, prefs in self.meta_verifier.domain_preferences.items()
            },
            "system_settings": {
                "enable_debate": self.enable_debate,
                "enable_parallel": self.enable_parallel,
                "max_workers": self.max_workers,
                "verification_timeout": self.verification_timeout
            }
        }
    
    def import_configuration(self, config: Dict[str, Any]):
        """Import verifier configuration"""
        try:
            # Update framework weights
            if "framework_weights" in config:
                self.meta_verifier.framework_weights = {
                    VerificationFramework(f): w for f, w in config["framework_weights"].items()
                }
            
            # Update domain preferences
            if "domain_preferences" in config:
                self.meta_verifier.domain_preferences = {
                    Domain(d): {VerificationFramework(f): w for f, w in prefs.items()}
                    for d, prefs in config["domain_preferences"].items()
                }
            
            # Update system settings
            if "system_settings" in config:
                settings = config["system_settings"]
                self.enable_debate = settings.get("enable_debate", self.enable_debate)
                self.enable_parallel = settings.get("enable_parallel", self.enable_parallel)
                self.max_workers = settings.get("max_workers", self.max_workers)
                self.verification_timeout = settings.get("verification_timeout", self.verification_timeout)
            
            self.logger.info("Configuration imported successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to import configuration: {e}")
            raise


# For numpy import in core.py
import numpy as np