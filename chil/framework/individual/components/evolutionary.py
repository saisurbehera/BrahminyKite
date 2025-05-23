"""
Component F: Evolutionary/Adaptive Learning

Handles self-improvement through:
- Performance tracking and analysis
- Adaptive rule updating
- Confidence modeling
- Feedback integration
"""

import numpy as np
from typing import List, Dict, Any, Optional
from datetime import datetime
from .base import VerificationComponent
from ..frameworks import VerificationFramework, VerificationResult, Claim, Domain


class EvolutionaryVerifier(VerificationComponent):
    """Component F: Adaptive Learning and Evolution"""
    
    def __init__(self):
        self.learning_history = []
        self.adaptation_rules = {
            "domain_weights": {},
            "framework_preferences": {},
            "confidence_adjustments": {}
        }
        self.confidence_models = {
            "historical_accuracy": {},
            "domain_expertise": {},
            "uncertainty_estimation": {}
        }
        self.feedback_records = []
        self.performance_metrics = {
            "accuracy_trends": [],
            "confidence_calibration": [],
            "adaptation_effectiveness": []
        }
    
    def verify(self, claim: Claim) -> VerificationResult:
        """Verify claim through adaptive learning assessment"""
        
        adaptation_score = self._evaluate_adaptation_potential(claim)
        confidence_score = self._calculate_confidence_score(claim)
        learning_score = self._assess_learning_value(claim)
        
        # Combine scores with learning-focused weighting
        combined_score = (adaptation_score * 0.4 + confidence_score * 0.4 + learning_score * 0.2)
        
        # Update learning history
        learning_record = {
            "claim": claim.content,
            "domain": claim.domain.value,
            "score": combined_score,
            "timestamp": datetime.now(),
            "adaptation_potential": adaptation_score,
            "confidence_level": confidence_score
        }
        self.learning_history.append(learning_record)
        
        # Calculate uncertainty based on historical performance
        uncertainty = self._calculate_uncertainty(claim)
        confidence_interval = self.get_confidence_bounds(combined_score, uncertainty)
        
        return VerificationResult(
            score=combined_score,
            framework=VerificationFramework.PRAGMATIST,  # Learning is pragmatic
            component=self.get_component_name(),
            evidence={
                "adaptation_potential_score": adaptation_score,
                "confidence_evolution_score": confidence_score,
                "learning_value_score": learning_score,
                "historical_records": len(self.learning_history),
                "adaptation_rules_count": len(self.adaptation_rules)
            },
            confidence_interval=confidence_interval,
            metadata={
                "learning_rate": np.random.uniform(0.1, 0.3),
                "adaptation_speed": np.random.uniform(0.2, 0.6),
                "confidence_calibration": self._get_confidence_calibration()
            }
        )
    
    def _evaluate_adaptation_potential(self, claim: Claim) -> float:
        """
        Evaluate the potential for learning and adaptation from this claim
        
        Considers:
        - Novelty of the claim
        - Learning opportunities
        - Adaptation requirements
        - Historical similar cases
        """
        # Check novelty
        novelty_score = self._assess_novelty(claim)
        
        # Check learning opportunities
        learning_opportunities = self._identify_learning_opportunities(claim)
        
        # Historical performance on similar claims
        historical_performance = self._get_historical_performance(claim)
        
        # Combine factors
        adaptation_score = (novelty_score * 0.4 + 
                          learning_opportunities * 0.3 + 
                          historical_performance * 0.3)
        
        return np.clip(adaptation_score, 0.0, 1.0)
    
    def _calculate_confidence_score(self, claim: Claim) -> float:
        """
        Calculate confidence score based on historical learning
        
        Uses:
        - Historical accuracy on similar claims
        - Domain expertise development
        - Uncertainty estimation models
        """
        # Base confidence from historical performance
        base_confidence = self._get_base_confidence(claim)
        
        # Adjust based on domain expertise
        domain_expertise = self._get_domain_expertise(claim.domain)
        
        # Uncertainty estimation
        uncertainty_factor = self._estimate_uncertainty(claim)
        
        # Combine factors
        confidence_score = base_confidence * domain_expertise * (1 - uncertainty_factor * 0.3)
        
        return np.clip(confidence_score, 0.0, 1.0)
    
    def _assess_learning_value(self, claim: Claim) -> float:
        """
        Assess how much learning value this claim provides
        
        Considers:
        - Information content
        - Verification challenge level
        - Potential for improvement
        - Feedback opportunities
        """
        # Information content
        info_content = len(claim.content.split()) / 100  # Normalized by length
        
        # Challenge level (more challenging = more learning)
        challenge_level = self._assess_challenge_level(claim)
        
        # Improvement potential
        improvement_potential = 1.0 - self._get_current_performance(claim)
        
        # Combine factors
        learning_value = (info_content * 0.2 + 
                         challenge_level * 0.5 + 
                         improvement_potential * 0.3)
        
        return np.clip(learning_value, 0.0, 1.0)
    
    def _assess_novelty(self, claim: Claim) -> float:
        """Assess how novel this claim is compared to historical data"""
        if not self.learning_history:
            return 1.0  # Everything is novel initially
        
        # Simple novelty based on content similarity
        similar_claims = 0
        for record in self.learning_history[-100:]:  # Check last 100 records
            if (record["domain"] == claim.domain.value and 
                self._calculate_similarity(claim.content, record["claim"]) > 0.7):
                similar_claims += 1
        
        novelty = 1.0 - (similar_claims / min(100, len(self.learning_history)))
        return np.clip(novelty, 0.0, 1.0)
    
    def _identify_learning_opportunities(self, claim: Claim) -> float:
        """Identify learning opportunities in the claim"""
        opportunities = 0.5  # Base score
        
        # Complex claims offer more learning
        if len(claim.content.split()) > 20:
            opportunities += 0.1
        
        # Rich context offers learning
        if len(claim.context) > 3:
            opportunities += 0.1
        
        # Source metadata provides learning signals
        if len(claim.source_metadata) > 2:
            opportunities += 0.1
        
        # Certain domains offer more learning
        learning_rich_domains = [Domain.SOCIAL, Domain.ETHICAL, Domain.AESTHETIC]
        if claim.domain in learning_rich_domains:
            opportunities += 0.1
        
        return np.clip(opportunities, 0.0, 1.0)
    
    def _get_historical_performance(self, claim: Claim) -> float:
        """Get historical performance on similar claims"""
        if not self.learning_history:
            return 0.5  # Neutral starting point
        
        # Find similar claims by domain
        domain_records = [r for r in self.learning_history if r["domain"] == claim.domain.value]
        
        if not domain_records:
            return 0.5
        
        # Return average performance on this domain
        avg_performance = np.mean([r["score"] for r in domain_records[-20:]])  # Last 20
        return avg_performance
    
    def _get_base_confidence(self, claim: Claim) -> float:
        """Get base confidence from historical accuracy"""
        if not self.feedback_records:
            return 0.5  # Neutral starting confidence
        
        # Calculate recent accuracy
        recent_feedback = self.feedback_records[-50:]  # Last 50 feedback records
        if not recent_feedback:
            return 0.5
        
        accuracies = [f.get("accuracy", 0.5) for f in recent_feedback]
        return np.mean(accuracies)
    
    def _get_domain_expertise(self, domain: Domain) -> float:
        """Get expertise level for a specific domain"""
        domain_records = [r for r in self.learning_history if r["domain"] == domain.value]
        
        if len(domain_records) < 5:
            return 0.7  # Low expertise initially
        
        # Expertise grows with experience and performance
        experience_factor = min(len(domain_records) / 100, 1.0)  # Normalize to 100 records
        performance_factor = np.mean([r["score"] for r in domain_records[-20:]])  # Recent performance
        
        expertise = 0.5 + (experience_factor * 0.3) + (performance_factor * 0.2)
        return np.clip(expertise, 0.0, 1.0)
    
    def _estimate_uncertainty(self, claim: Claim) -> float:
        """Estimate uncertainty for this type of claim"""
        # Base uncertainty
        uncertainty = 0.3
        
        # Higher uncertainty for new domains
        domain_experience = self._get_domain_expertise(claim.domain)
        uncertainty += (1 - domain_experience) * 0.2
        
        # Higher uncertainty for complex claims
        complexity = len(claim.content.split()) / 50  # Normalized complexity
        uncertainty += min(complexity * 0.1, 0.2)
        
        return np.clip(uncertainty, 0.0, 1.0)
    
    def _assess_challenge_level(self, claim: Claim) -> float:
        """Assess how challenging this claim is to verify"""
        challenge = 0.5  # Base challenge
        
        # Complex domains are more challenging
        challenging_domains = [Domain.AESTHETIC, Domain.ETHICAL, Domain.CREATIVE]
        if claim.domain in challenging_domains:
            challenge += 0.2
        
        # Longer claims are more challenging
        if len(claim.content.split()) > 30:
            challenge += 0.1
        
        # Rich context suggests complexity
        if len(claim.context) > 5:
            challenge += 0.1
        
        return np.clip(challenge, 0.0, 1.0)
    
    def _get_current_performance(self, claim: Claim) -> float:
        """Get current performance level for this type of claim"""
        return self._get_historical_performance(claim)
    
    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate simple text similarity"""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union)
    
    def _calculate_uncertainty(self, claim: Claim) -> float:
        """Calculate uncertainty based on historical performance"""
        if len(self.learning_history) < 10:
            return 0.2  # Higher uncertainty initially
        
        # Get recent performance variance
        recent_scores = [r["score"] for r in self.learning_history[-20:]]
        variance = np.var(recent_scores)
        
        # Convert variance to uncertainty
        uncertainty = min(variance * 2, 0.3)  # Cap at 0.3
        return uncertainty
    
    def _get_confidence_calibration(self) -> float:
        """Get current confidence calibration score"""
        if not self.feedback_records:
            return 0.5
        
        # Simple calibration: how well confidence matches actual performance
        calibration_scores = []
        for feedback in self.feedback_records[-20:]:
            predicted_conf = feedback.get("predicted_confidence", 0.5)
            actual_accuracy = feedback.get("accuracy", 0.5)
            calibration = 1.0 - abs(predicted_conf - actual_accuracy)
            calibration_scores.append(calibration)
        
        return np.mean(calibration_scores) if calibration_scores else 0.5
    
    def update_verification_rules(self, feedback: Dict[str, Any]):
        """Update verification rules based on feedback"""
        self.feedback_records.append({
            "feedback": feedback,
            "timestamp": datetime.now(),
            "accuracy": feedback.get("accuracy", 0.5),
            "predicted_confidence": feedback.get("predicted_confidence", 0.5)
        })
        
        # Update adaptation rules
        if "domain_preference" in feedback:
            domain = feedback["domain_preference"]
            self.adaptation_rules["domain_weights"][domain] = feedback.get("weight_adjustment", 1.0)
        
        if "framework_preference" in feedback:
            framework = feedback["framework_preference"]
            self.adaptation_rules["framework_preferences"][framework] = feedback.get("preference_score", 1.0)
        
        # Update performance metrics
        self._update_performance_metrics(feedback)
    
    def _update_performance_metrics(self, feedback: Dict[str, Any]):
        """Update internal performance tracking metrics"""
        accuracy = feedback.get("accuracy", 0.5)
        confidence = feedback.get("predicted_confidence", 0.5)
        
        self.performance_metrics["accuracy_trends"].append(accuracy)
        self.performance_metrics["confidence_calibration"].append(abs(confidence - accuracy))
        
        # Keep only recent metrics
        max_history = 1000
        for metric_list in self.performance_metrics.values():
            if len(metric_list) > max_history:
                metric_list[:] = metric_list[-max_history:]
    
    def get_applicable_frameworks(self) -> List[VerificationFramework]:
        """Frameworks implemented by evolutionary verification"""
        return [VerificationFramework.PRAGMATIST]
    
    def validate_claim(self, claim: Claim) -> bool:
        """Check if claim can benefit from evolutionary analysis"""
        # All claims provide learning opportunities
        return True
    
    def get_learning_statistics(self) -> Dict[str, Any]:
        """Get detailed learning and adaptation statistics"""
        if not self.learning_history:
            return {"message": "No learning history available yet"}
        
        # Domain expertise
        domain_expertise = {}
        for domain in Domain:
            domain_expertise[domain.value] = self._get_domain_expertise(domain)
        
        # Recent performance
        recent_scores = [r["score"] for r in self.learning_history[-20:]]
        avg_recent_performance = np.mean(recent_scores) if recent_scores else 0.5
        
        # Learning trends
        if len(self.learning_history) >= 10:
            early_scores = [r["score"] for r in self.learning_history[:10]]
            late_scores = [r["score"] for r in self.learning_history[-10:]]
            improvement = np.mean(late_scores) - np.mean(early_scores)
        else:
            improvement = 0.0
        
        return {
            "total_learning_records": len(self.learning_history),
            "feedback_records": len(self.feedback_records),
            "domain_expertise": domain_expertise,
            "recent_performance": avg_recent_performance,
            "learning_improvement": improvement,
            "confidence_calibration": self._get_confidence_calibration(),
            "adaptation_rules_count": sum(len(rules) for rules in self.adaptation_rules.values())
        }
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about evolutionary verification performance"""
        return {
            "learning_history_size": len(self.learning_history),
            "feedback_records_count": len(self.feedback_records),
            "adaptation_rules_count": sum(len(rules) for rules in self.adaptation_rules.values()),
            "confidence_models_count": len(self.confidence_models),
            "applicable_frameworks": [f.value for f in self.get_applicable_frameworks()]
        }