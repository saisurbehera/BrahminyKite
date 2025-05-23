"""
Meta-Verification System

Handles conflict resolution between verification frameworks using:
- Pareto-optimal solutions
- Reflective equilibrium
- Multi-objective optimization
- Domain-appropriate weighting
"""

import numpy as np
from typing import Dict, List, Any, Optional
from .frameworks import VerificationFramework, VerificationResult, Claim, Domain, get_domain_characteristics


class MetaVerificationSystem:
    """
    Higher-order system to arbitrate between verification frameworks
    Implements reflective equilibrium and multi-objective optimization
    """
    
    def __init__(self):
        self.framework_weights = {
            framework: 1.0 for framework in VerificationFramework
        }
        self.domain_preferences = self._initialize_domain_preferences()
        self.conflict_resolution_history = []
        self.learning_rate = 0.1
        self.adaptation_threshold = 0.1
    
    def _initialize_domain_preferences(self) -> Dict[Domain, Dict[VerificationFramework, float]]:
        """Initialize domain-specific framework preferences based on philosophical foundations"""
        return {
            Domain.EMPIRICAL: {
                VerificationFramework.POSITIVIST: 1.0,
                VerificationFramework.CORRESPONDENCE: 0.9,
                VerificationFramework.PRAGMATIST: 0.7,
                VerificationFramework.COHERENCE: 0.6,
                VerificationFramework.INTERPRETIVIST: 0.3,
                VerificationFramework.CONSTRUCTIVIST: 0.2
            },
            Domain.AESTHETIC: {
                VerificationFramework.INTERPRETIVIST: 1.0,
                VerificationFramework.CONSTRUCTIVIST: 0.9,
                VerificationFramework.COHERENCE: 0.7,
                VerificationFramework.PRAGMATIST: 0.6,
                VerificationFramework.CORRESPONDENCE: 0.3,
                VerificationFramework.POSITIVIST: 0.2
            },
            Domain.ETHICAL: {
                VerificationFramework.COHERENCE: 1.0,
                VerificationFramework.PRAGMATIST: 0.8,
                VerificationFramework.CONSTRUCTIVIST: 0.7,
                VerificationFramework.INTERPRETIVIST: 0.6,
                VerificationFramework.CORRESPONDENCE: 0.5,
                VerificationFramework.POSITIVIST: 0.4
            },
            Domain.LOGICAL: {
                VerificationFramework.COHERENCE: 1.0,
                VerificationFramework.POSITIVIST: 0.9,
                VerificationFramework.CORRESPONDENCE: 0.8,
                VerificationFramework.PRAGMATIST: 0.6,
                VerificationFramework.INTERPRETIVIST: 0.4,
                VerificationFramework.CONSTRUCTIVIST: 0.3
            },
            Domain.SOCIAL: {
                VerificationFramework.INTERPRETIVIST: 1.0,
                VerificationFramework.CONSTRUCTIVIST: 0.9,
                VerificationFramework.PRAGMATIST: 0.8,
                VerificationFramework.COHERENCE: 0.6,
                VerificationFramework.POSITIVIST: 0.5,
                VerificationFramework.CORRESPONDENCE: 0.4
            },
            Domain.CREATIVE: {
                VerificationFramework.INTERPRETIVIST: 1.0,
                VerificationFramework.PRAGMATIST: 0.8,
                VerificationFramework.CONSTRUCTIVIST: 0.7,
                VerificationFramework.COHERENCE: 0.5,
                VerificationFramework.CORRESPONDENCE: 0.3,
                VerificationFramework.POSITIVIST: 0.2
            }
        }
    
    def resolve_conflicts(self, results: List[VerificationResult], claim: Claim) -> VerificationResult:
        """
        Resolve conflicts between different verification frameworks
        Uses Pareto-optimal solutions and reflective equilibrium
        """
        if not results:
            raise ValueError("No verification results to resolve")
        
        # Group results by framework
        framework_results = self._group_by_framework(results)
        
        # Calculate weighted scores considering domain appropriateness
        weighted_scores = self._calculate_weighted_scores(framework_results, claim.domain)
        
        # Apply reflective equilibrium
        equilibrium_scores = self._apply_reflective_equilibrium(weighted_scores, claim)
        
        # Find Pareto-optimal solution
        optimal_framework, final_score = self._find_pareto_optimal(equilibrium_scores, framework_results)
        
        # Combine evidence from all frameworks
        combined_evidence = self._combine_evidence(results)
        
        # Calculate uncertainty considering all frameworks
        uncertainty = self._calculate_meta_uncertainty(results, weighted_scores)
        
        # Create meta-verification result
        meta_result = VerificationResult(
            score=final_score,
            framework=optimal_framework,
            component="meta_verification",
            evidence=combined_evidence,
            confidence_interval=(
                max(0, final_score - uncertainty),
                min(1, final_score + uncertainty)
            ),
            metadata={
                "framework_scores": {f.value: s for f, s in weighted_scores.items()},
                "equilibrium_adjustments": len(self.conflict_resolution_history),
                "uncertainty_level": uncertainty,
                "frameworks_considered": len(framework_results),
                "dominant_weight": max(weighted_scores.values()) if weighted_scores else 0,
                "consensus_level": self._calculate_consensus_level(weighted_scores)
            }
        )
        
        # Record for learning
        self._record_resolution(claim, framework_results, optimal_framework, final_score)
        
        return meta_result
    
    def _group_by_framework(self, results: List[VerificationResult]) -> Dict[VerificationFramework, List[VerificationResult]]:
        """Group verification results by their frameworks"""
        framework_results = {}
        for result in results:
            if result.framework not in framework_results:
                framework_results[result.framework] = []
            framework_results[result.framework].append(result)
        return framework_results
    
    def _calculate_weighted_scores(self, framework_results: Dict[VerificationFramework, List[VerificationResult]], 
                                 domain: Domain) -> Dict[VerificationFramework, float]:
        """Calculate weighted scores for each framework"""
        weighted_scores = {}
        domain_weights = self.domain_preferences.get(domain, {})
        
        for framework, results_list in framework_results.items():
            # Average score for this framework
            avg_score = np.mean([r.score for r in results_list])
            
            # Apply domain preference weight
            domain_weight = domain_weights.get(framework, 1.0)
            
            # Apply global framework weight
            global_weight = self.framework_weights[framework]
            
            # Combine weights
            final_weight = domain_weight * global_weight
            weighted_scores[framework] = avg_score * final_weight
        
        return weighted_scores
    
    def _apply_reflective_equilibrium(self, weighted_scores: Dict[VerificationFramework, float], 
                                    claim: Claim) -> Dict[VerificationFramework, float]:
        """
        Apply reflective equilibrium to balance principles with concrete judgments
        
        This implements Rawls' concept of adjusting between abstract principles
        and particular judgments until they reach coherent balance.
        """
        equilibrium_scores = weighted_scores.copy()
        
        # Check for outliers that might indicate need for adjustment
        if len(weighted_scores) > 1:
            scores = list(weighted_scores.values())
            mean_score = np.mean(scores)
            std_score = np.std(scores)
            
            # Identify outliers (scores > 2 standard deviations from mean)
            for framework, score in weighted_scores.items():
                if abs(score - mean_score) > 2 * std_score:
                    # Moderate outliers toward the mean (reflective equilibrium)
                    adjustment_factor = 0.8  # Move 20% toward mean
                    equilibrium_scores[framework] = score * adjustment_factor + mean_score * (1 - adjustment_factor)
        
        # Historical learning: adjust based on past performance
        for framework in equilibrium_scores:
            historical_adjustment = self._get_historical_adjustment(framework, claim)
            equilibrium_scores[framework] *= historical_adjustment
        
        return equilibrium_scores
    
    def _find_pareto_optimal(self, scores: Dict[VerificationFramework, float], 
                           framework_results: Dict[VerificationFramework, List[VerificationResult]]) -> tuple[VerificationFramework, float]:
        """
        Find Pareto-optimal solution among frameworks
        
        A Pareto-optimal solution is one where we cannot improve one criterion
        without worsening another.
        """
        if not scores:
            raise ValueError("No scores to optimize")
        
        # Simple Pareto optimization: find framework with highest weighted score
        # In a more complex implementation, this would consider multiple objectives
        best_framework = max(scores, key=scores.get)
        best_score = scores[best_framework]
        
        # Adjust score based on result quality metrics
        if best_framework in framework_results:
            results = framework_results[best_framework]
            
            # Consider confidence intervals
            avg_confidence_width = np.mean([
                r.confidence_interval[1] - r.confidence_interval[0] 
                for r in results
            ])
            
            # Penalize wide confidence intervals (high uncertainty)
            confidence_penalty = min(avg_confidence_width * 0.1, 0.1)
            best_score = max(0, best_score - confidence_penalty)
        
        return best_framework, best_score
    
    def _combine_evidence(self, results: List[VerificationResult]) -> Dict[str, Any]:
        """Combine evidence from all verification results"""
        combined_evidence = {
            "component_scores": {},
            "total_components": len(results),
            "evidence_sources": set(),
            "verification_methods": set()
        }
        
        for result in results:
            # Collect component scores
            combined_evidence["component_scores"][result.component] = result.score
            
            # Collect evidence sources
            for key, value in result.evidence.items():
                if isinstance(value, (list, set)):
                    combined_evidence["evidence_sources"].update(value)
                elif isinstance(value, str):
                    combined_evidence["evidence_sources"].add(value)
            
            # Collect verification methods
            combined_evidence["verification_methods"].add(result.framework.value)
        
        # Convert sets to lists for JSON serialization
        combined_evidence["evidence_sources"] = list(combined_evidence["evidence_sources"])
        combined_evidence["verification_methods"] = list(combined_evidence["verification_methods"])
        
        return combined_evidence
    
    def _calculate_meta_uncertainty(self, results: List[VerificationResult], 
                                   weighted_scores: Dict[VerificationFramework, float]) -> float:
        """Calculate meta-level uncertainty considering all frameworks"""
        if not results:
            return 0.5
        
        # Uncertainty from score variance
        all_scores = [r.score for r in results]
        score_variance = np.var(all_scores)
        
        # Uncertainty from confidence interval widths
        interval_widths = [r.confidence_interval[1] - r.confidence_interval[0] for r in results]
        avg_interval_width = np.mean(interval_widths)
        
        # Uncertainty from framework disagreement
        if len(weighted_scores) > 1:
            framework_disagreement = np.std(list(weighted_scores.values()))
        else:
            framework_disagreement = 0
        
        # Combine uncertainty sources
        meta_uncertainty = (score_variance * 0.4 + 
                          avg_interval_width * 0.4 + 
                          framework_disagreement * 0.2)
        
        return min(meta_uncertainty, 0.5)  # Cap at 0.5
    
    def _calculate_consensus_level(self, weighted_scores: Dict[VerificationFramework, float]) -> float:
        """Calculate level of consensus between frameworks"""
        if len(weighted_scores) <= 1:
            return 1.0
        
        scores = list(weighted_scores.values())
        # High consensus = low standard deviation
        std_dev = np.std(scores)
        max_possible_std = np.sqrt(np.var([0, 1]))  # Max std for 0-1 range
        
        consensus = 1.0 - (std_dev / max_possible_std)
        return max(0, consensus)
    
    def _get_historical_adjustment(self, framework: VerificationFramework, claim: Claim) -> float:
        """Get historical performance adjustment for a framework"""
        if not self.conflict_resolution_history:
            return 1.0
        
        # Find similar historical cases
        similar_cases = [
            record for record in self.conflict_resolution_history[-50:]  # Last 50 cases
            if record["claim_domain"] == claim.domain
        ]
        
        if not similar_cases:
            return 1.0
        
        # Count how often this framework was chosen for similar cases
        framework_choices = [record["chosen_framework"] for record in similar_cases]
        framework_frequency = framework_choices.count(framework) / len(framework_choices)
        
        # Moderate adjustment based on historical success
        adjustment = 0.9 + (framework_frequency * 0.2)  # Range: 0.9 to 1.1
        return adjustment
    
    def _record_resolution(self, claim: Claim, framework_results: Dict[VerificationFramework, List[VerificationResult]], 
                          chosen_framework: VerificationFramework, final_score: float):
        """Record conflict resolution for learning"""
        resolution_record = {
            "claim_domain": claim.domain,
            "claim_length": len(claim.content),
            "frameworks_used": list(framework_results.keys()),
            "chosen_framework": chosen_framework,
            "final_score": final_score,
            "num_frameworks": len(framework_results),
            "timestamp": np.datetime64('now')
        }
        
        self.conflict_resolution_history.append(resolution_record)
        
        # Keep history manageable
        if len(self.conflict_resolution_history) > 1000:
            self.conflict_resolution_history = self.conflict_resolution_history[-1000:]
    
    def update_framework_weights(self, feedback: Dict[VerificationFramework, float]):
        """Update framework weights based on performance feedback"""
        for framework, adjustment in feedback.items():
            old_weight = self.framework_weights[framework]
            new_weight = old_weight * (1 + self.learning_rate * (adjustment - 1))
            
            # Keep weights in reasonable bounds
            self.framework_weights[framework] = np.clip(new_weight, 0.1, 2.0)
    
    def update_domain_preferences(self, domain: Domain, framework_adjustments: Dict[VerificationFramework, float]):
        """Update domain-specific framework preferences"""
        if domain not in self.domain_preferences:
            self.domain_preferences[domain] = {f: 1.0 for f in VerificationFramework}
        
        for framework, adjustment in framework_adjustments.items():
            old_preference = self.domain_preferences[domain].get(framework, 1.0)
            new_preference = old_preference * (1 + self.learning_rate * (adjustment - 1))
            
            # Keep preferences in reasonable bounds
            self.domain_preferences[domain][framework] = np.clip(new_preference, 0.1, 2.0)
    
    def analyze_framework_performance(self) -> Dict[str, Any]:
        """Analyze performance of different frameworks across domains"""
        if not self.conflict_resolution_history:
            return {"message": "No resolution history available"}
        
        analysis = {
            "framework_usage": {},
            "domain_preferences": {},
            "average_scores": {},
            "consensus_trends": []
        }
        
        # Framework usage frequency
        all_frameworks = [record["chosen_framework"] for record in self.conflict_resolution_history]
        for framework in VerificationFramework:
            analysis["framework_usage"][framework.value] = all_frameworks.count(framework)
        
        # Domain-specific preferences
        for domain in Domain:
            domain_records = [r for r in self.conflict_resolution_history if r["claim_domain"] == domain]
            if domain_records:
                domain_frameworks = [r["chosen_framework"] for r in domain_records]
                most_common = max(set(domain_frameworks), key=domain_frameworks.count)
                analysis["domain_preferences"][domain.value] = most_common.value
        
        # Average scores by framework
        for framework in VerificationFramework:
            framework_records = [r for r in self.conflict_resolution_history if r["chosen_framework"] == framework]
            if framework_records:
                avg_score = np.mean([r["final_score"] for r in framework_records])
                analysis["average_scores"][framework.value] = avg_score
        
        return analysis
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get meta-verification system statistics"""
        return {
            "framework_weights": {f.value: w for f, w in self.framework_weights.items()},
            "resolution_history_size": len(self.conflict_resolution_history),
            "domains_configured": len(self.domain_preferences),
            "learning_rate": self.learning_rate,
            "adaptation_threshold": self.adaptation_threshold
        }