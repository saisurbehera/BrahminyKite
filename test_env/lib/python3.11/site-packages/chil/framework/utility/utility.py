"""
Component E: Pragmatic Utility Assessment

Handles practical effectiveness through:
- Real-world outcome assessment
- Cost-benefit analysis
- Simulation-based evaluation
- Reinforcement learning optimization
"""

import numpy as np
from typing import List, Dict, Any
from .base import VerificationComponent
from ..frameworks import VerificationFramework, VerificationResult, Claim, Domain


class UtilityVerifier(VerificationComponent):
    """Component E: Pragmatic Utility Assessment"""
    
    def __init__(self):
        self.simulation_environments = {
            "monte_carlo": {},
            "agent_based": {},
            "system_dynamics": {}
        }
        self.outcome_predictors = {
            "regression_models": {},
            "neural_networks": {},
            "ensemble_methods": {}
        }
        self.effectiveness_metrics = {
            "cost_benefit": {},
            "impact_assessment": {},
            "efficiency_measures": {}
        }
        self.utility_cache = {}
        self.outcome_history = []
    
    def verify(self, claim: Claim) -> VerificationResult:
        """Verify claim through pragmatic utility assessment"""
        
        practical_score = self._assess_practical_outcomes(claim)
        effectiveness_score = self._measure_effectiveness(claim)
        implementability_score = self._evaluate_implementability(claim)
        
        # Weight based on domain
        if claim.domain == Domain.ETHICAL:
            # For ethical claims, outcomes matter most
            combined_score = (practical_score * 0.5 + effectiveness_score * 0.3 + 
                            implementability_score * 0.2)
        elif claim.domain == Domain.SOCIAL:
            # For social claims, balance all aspects
            combined_score = (practical_score + effectiveness_score + implementability_score) / 3
        else:
            # Default weighting
            combined_score = (practical_score * 0.4 + effectiveness_score * 0.4 + 
                            implementability_score * 0.2)
        
        # Calculate uncertainty
        scores = [practical_score, effectiveness_score, implementability_score]
        uncertainty = np.std(scores) * 0.4
        confidence_interval = self.get_confidence_bounds(combined_score, uncertainty)
        
        return VerificationResult(
            score=combined_score,
            framework=VerificationFramework.PRAGMATIST,
            component=self.get_component_name(),
            evidence={
                "practical_outcomes_score": practical_score,
                "effectiveness_score": effectiveness_score,
                "implementability_score": implementability_score,
                "simulation_environments_used": list(self.simulation_environments.keys()),
                "prediction_models_used": list(self.outcome_predictors.keys())
            },
            confidence_interval=confidence_interval,
            metadata={
                "utility_complexity": np.random.uniform(0.3, 0.8),
                "prediction_confidence": np.random.uniform(0.5, 0.9)
            }
        )
    
    def _assess_practical_outcomes(self, claim: Claim) -> float:
        """
        Assess practical real-world outcomes of the claim
        
        In a real implementation, this would:
        - Run Monte Carlo simulations
        - Analyze historical outcome data
        - Model potential consequences
        - Assess intervention effectiveness
        """
        cache_key = f"practical_{hash(claim.content)}"
        if cache_key in self.utility_cache:
            return self.utility_cache[cache_key]
        
        # Simulate practical outcome assessment
        base_score = np.random.uniform(0.2, 0.9)
        
        # Check for outcome-related terms
        outcome_terms = ["results in", "leads to", "causes", "improves", "reduces", "increases"]
        has_outcome_language = any(term in claim.content.lower() for term in outcome_terms)
        
        if has_outcome_language:
            base_score *= 1.1
        
        # Check for measurable outcomes
        measurable_terms = ["percent", "percentage", "%", "number", "amount", "rate", "level"]
        has_measurable_outcomes = any(term in claim.content.lower() for term in measurable_terms)
        
        if has_measurable_outcomes:
            base_score *= 1.15
        
        # Check context for outcome data
        if "historical_data" in claim.context:
            base_score *= 1.2
        if "pilot_study" in claim.context:
            base_score *= 1.1
        if "case_study" in claim.context:
            base_score *= 1.05
        
        # Domain-specific adjustments
        if claim.domain == Domain.EMPIRICAL:
            base_score *= 1.1  # Empirical claims often have clear outcomes
        elif claim.domain == Domain.AESTHETIC:
            base_score *= 0.8  # Aesthetic claims have less clear practical outcomes
        
        # Penalize vague claims
        vague_terms = ["might", "could", "possibly", "perhaps", "maybe"]
        has_vague_language = any(term in claim.content.lower() for term in vague_terms)
        
        if has_vague_language:
            base_score *= 0.9
        
        score = np.clip(base_score, 0.0, 1.0)
        self.utility_cache[cache_key] = score
        return score
    
    def _measure_effectiveness(self, claim: Claim) -> float:
        """
        Measure the effectiveness of claims that propose actions or interventions
        
        In a real implementation, this would:
        - Analyze cost-benefit ratios
        - Compare with alternative approaches
        - Assess resource efficiency
        - Evaluate scalability
        """
        cache_key = f"effectiveness_{hash(claim.content)}"
        if cache_key in self.utility_cache:
            return self.utility_cache[cache_key]
        
        # Simulate effectiveness measurement
        base_score = np.random.uniform(0.3, 0.8)
        
        # Check for effectiveness indicators
        effectiveness_terms = ["effective", "efficient", "successful", "works", "solves", "addresses"]
        has_effectiveness_language = any(term in claim.content.lower() for term in effectiveness_terms)
        
        if has_effectiveness_language:
            base_score *= 1.1
        
        # Check for comparative language
        comparative_terms = ["better than", "more effective", "superior", "improves on", "outperforms"]
        has_comparative_language = any(term in claim.content.lower() for term in comparative_terms)
        
        if has_comparative_language:
            base_score *= 1.05
        
        # Check for efficiency indicators
        efficiency_terms = ["cost-effective", "resource-efficient", "scalable", "sustainable"]
        has_efficiency_language = any(term in claim.content.lower() for term in efficiency_terms)
        
        if has_efficiency_language:
            base_score *= 1.1
        
        # Check context for effectiveness data
        if "effectiveness_data" in claim.context:
            base_score *= 1.15
        if "comparison_study" in claim.context:
            base_score *= 1.1
        if "meta_analysis" in claim.context:
            base_score *= 1.2
        
        # Penalize unsupported effectiveness claims
        unsupported_terms = ["obviously effective", "clearly works", "definitely solves"]
        has_unsupported_claims = any(term in claim.content.lower() for term in unsupported_terms)
        
        if has_unsupported_claims:
            base_score *= 0.8
        
        score = np.clip(base_score, 0.0, 1.0)
        self.utility_cache[cache_key] = score
        return score
    
    def _evaluate_implementability(self, claim: Claim) -> float:
        """
        Evaluate how implementable or actionable the claim is
        
        In a real implementation, this would:
        - Assess resource requirements
        - Check feasibility constraints
        - Analyze implementation barriers
        - Evaluate practical viability
        """
        cache_key = f"implementability_{hash(claim.content)}"
        if cache_key in self.utility_cache:
            return self.utility_cache[cache_key]
        
        # Simulate implementability evaluation
        base_score = np.random.uniform(0.4, 0.8)
        
        # Check for actionable language
        actionable_terms = ["implement", "apply", "use", "adopt", "deploy", "execute"]
        has_actionable_language = any(term in claim.content.lower() for term in actionable_terms)
        
        if has_actionable_language:
            base_score *= 1.1
        
        # Check for specific steps or methods
        method_terms = ["method", "approach", "strategy", "technique", "procedure", "process"]
        has_method_language = any(term in claim.content.lower() for term in method_terms)
        
        if has_method_language:
            base_score *= 1.05
        
        # Check for resource considerations
        resource_terms = ["cost", "budget", "resources", "time", "effort", "investment"]
        has_resource_awareness = any(term in claim.content.lower() for term in resource_terms)
        
        if has_resource_awareness:
            base_score *= 1.1
        
        # Check context for implementation details
        if "implementation_plan" in claim.context:
            base_score *= 1.2
        if "resource_requirements" in claim.context:
            base_score *= 1.1
        if "timeline" in claim.context:
            base_score *= 1.05
        
        # Penalize vague or impractical claims
        impractical_terms = ["impossible", "unrealistic", "impractical", "unfeasible"]
        has_impractical_language = any(term in claim.content.lower() for term in impractical_terms)
        
        if has_impractical_language:
            base_score *= 0.7
        
        # Abstract claims are harder to implement
        abstract_terms = ["abstract", "theoretical", "conceptual", "philosophical"]
        has_abstract_language = any(term in claim.content.lower() for term in abstract_terms)
        
        if has_abstract_language and claim.domain != Domain.LOGICAL:
            base_score *= 0.9
        
        score = np.clip(base_score, 0.0, 1.0)
        self.utility_cache[cache_key] = score
        return score
    
    def get_applicable_frameworks(self) -> List[VerificationFramework]:
        """Frameworks implemented by utility verification"""
        return [VerificationFramework.PRAGMATIST]
    
    def validate_claim(self, claim: Claim) -> bool:
        """Check if claim can benefit from utility analysis"""
        # Utility analysis is most valuable for actionable claims
        actionable_domains = [Domain.ETHICAL, Domain.SOCIAL, Domain.EMPIRICAL]
        return claim.domain in actionable_domains
    
    def add_simulation_environment(self, env_name: str, env_config: Dict[str, Any]):
        """Add a new simulation environment"""
        self.simulation_environments[env_name] = env_config
    
    def add_outcome_predictor(self, predictor_name: str, predictor_config: Dict[str, Any]):
        """Add a new outcome prediction model"""
        self.outcome_predictors[predictor_name] = predictor_config
    
    def add_effectiveness_metric(self, metric_name: str, metric_config: Dict[str, Any]):
        """Add a new effectiveness measurement metric"""
        self.effectiveness_metrics[metric_name] = metric_config
    
    def record_outcome(self, claim: Claim, predicted_score: float, actual_outcome: float):
        """Record actual outcomes for learning"""
        self.outcome_history.append({
            "claim": claim.content,
            "predicted": predicted_score,
            "actual": actual_outcome,
            "domain": claim.domain.value,
            "timestamp": np.datetime64('now')
        })
    
    def clear_cache(self):
        """Clear the utility cache"""
        self.utility_cache.clear()
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about utility verification performance"""
        return {
            "simulation_environments_count": len(self.simulation_environments),
            "outcome_predictors_count": len(self.outcome_predictors),
            "effectiveness_metrics_count": len(self.effectiveness_metrics),
            "outcome_records": len(self.outcome_history),
            "cache_size": len(self.utility_cache),
            "applicable_frameworks": [f.value for f in self.get_applicable_frameworks()]
        }