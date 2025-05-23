"""
Component D: Power Dynamics Analysis

Handles authority and bias assessment through:
- Source credibility evaluation
- Bias detection in claims and sources
- Authority hierarchy analysis
- Information flow network mapping
"""

import numpy as np
from typing import List, Dict, Any
from .base import VerificationComponent
from ..frameworks import VerificationFramework, VerificationResult, Claim, Domain


class PowerDynamicsVerifier(VerificationComponent):
    """Component D: Power Dynamics Analysis"""
    
    def __init__(self):
        self.bias_detection_models = {
            "linguistic_bias": {},
            "selection_bias": {},
            "confirmation_bias": {},
            "cultural_bias": {}
        }
        self.authority_networks = {
            "academic_institutions": {},
            "media_credibility": {},
            "expert_networks": {},
            "institutional_authority": {}
        }
        self.known_biases = {}
        self.authority_cache = {}
    
    def verify(self, claim: Claim) -> VerificationResult:
        """Verify claim through power dynamics analysis"""
        
        authority_score = self._evaluate_source_authority(claim)
        bias_score = self._detect_biases(claim)
        institutional_power_score = self._assess_institutional_power(claim)
        
        # Bias detection reduces overall confidence
        # Higher bias = lower verification score
        bias_penalty = bias_score * 0.3
        combined_score = (authority_score * 0.5 + institutional_power_score * 0.3) * (1 - bias_penalty)
        
        # Higher uncertainty for power dynamics analysis
        uncertainty = max(bias_score * 0.2, 0.1)
        confidence_interval = self.get_confidence_bounds(combined_score, uncertainty)
        
        return VerificationResult(
            score=combined_score,
            framework=VerificationFramework.CONSTRUCTIVIST,
            component=self.get_component_name(),
            evidence={
                "source_authority_score": authority_score,
                "bias_detected_score": bias_score,
                "institutional_power_score": institutional_power_score,
                "bias_types_detected": self._identify_bias_types(claim),
                "authority_sources": list(self.authority_networks.keys())
            },
            confidence_interval=confidence_interval,
            metadata={
                "power_analysis_depth": np.random.uniform(0.3, 0.8),
                "bias_detection_sensitivity": np.random.uniform(0.4, 0.9)
            }
        )
    
    def _evaluate_source_authority(self, claim: Claim) -> float:
        """
        Evaluate the credibility and authority of claim sources
        
        In a real implementation, this would:
        - Check academic credentials and affiliations
        - Assess publication venue credibility
        - Analyze citation networks
        - Evaluate institutional reputation
        """
        cache_key = f"authority_{hash(str(claim.source_metadata))}"
        if cache_key in self.authority_cache:
            return self.authority_cache[cache_key]
        
        # Base authority score
        base_score = np.random.uniform(0.3, 0.8)
        
        # Check source metadata for authority indicators
        source_meta = claim.source_metadata
        
        # Academic indicators
        academic_indicators = ["university", "professor", "dr.", "phd", "researcher", "journal"]
        if any(indicator in str(source_meta).lower() for indicator in academic_indicators):
            base_score *= 1.2
        
        # Media credibility
        credible_media = ["scientific journal", "peer reviewed", "academic press"]
        if any(media in str(source_meta).lower() for media in credible_media):
            base_score *= 1.15
        
        # Check for specific authority markers
        if "peer_reviewed" in source_meta:
            base_score *= 1.25
        if "citation_count" in source_meta and source_meta.get("citation_count", 0) > 10:
            base_score *= 1.1
        if "institutional_affiliation" in source_meta:
            base_score *= 1.05
        
        # Penalize questionable sources
        questionable_indicators = ["blog", "social media", "anonymous", "unverified"]
        if any(indicator in str(source_meta).lower() for indicator in questionable_indicators):
            base_score *= 0.7
        
        score = np.clip(base_score, 0.0, 1.0)
        self.authority_cache[cache_key] = score
        return score
    
    def _detect_biases(self, claim: Claim) -> float:
        """
        Detect various forms of bias in the claim
        
        In a real implementation, this would:
        - Run ML bias detection models
        - Analyze linguistic bias patterns
        - Check for selection bias
        - Identify cultural and ideological bias
        """
        cache_key = f"bias_{hash(claim.content)}"
        if cache_key in self.authority_cache:
            return self.authority_cache[cache_key]
        
        # Simulate bias detection
        base_bias = np.random.uniform(0.1, 0.6)
        
        # Language bias indicators
        biased_language = ["obviously", "clearly", "everyone knows", "it's obvious", "undeniably"]
        has_biased_language = any(phrase in claim.content.lower() for phrase in biased_language)
        
        if has_biased_language:
            base_bias += 0.1
        
        # Absolute statements (potential confirmation bias)
        absolute_terms = ["all", "never", "always", "every", "none", "completely"]
        has_absolutes = any(term in claim.content.lower() for term in absolute_terms)
        
        if has_absolutes:
            base_bias += 0.05
        
        # Emotional language (potential bias)
        emotional_terms = ["shocking", "outrageous", "amazing", "terrible", "wonderful"]
        has_emotional_language = any(term in claim.content.lower() for term in emotional_terms)
        
        if has_emotional_language:
            base_bias += 0.08
        
        # Check for cultural bias indicators
        cultural_assumptions = ["western", "civilized", "primitive", "advanced", "backward"]
        has_cultural_bias = any(term in claim.content.lower() for term in cultural_assumptions)
        
        if has_cultural_bias:
            base_bias += 0.15
        
        # Domain-specific bias patterns
        if claim.domain == Domain.SOCIAL:
            # Social claims more prone to bias
            base_bias *= 1.1
        elif claim.domain == Domain.EMPIRICAL:
            # Empirical claims less prone to bias
            base_bias *= 0.8
        
        bias_score = np.clip(base_bias, 0.0, 1.0)
        self.authority_cache[cache_key] = bias_score
        return bias_score
    
    def _assess_institutional_power(self, claim: Claim) -> float:
        """
        Assess the institutional power behind the claim
        
        In a real implementation, this would:
        - Analyze funding sources
        - Check institutional conflicts of interest
        - Assess political and economic influences
        - Evaluate power structures affecting the claim
        """
        cache_key = f"institutional_{hash(str(claim.source_metadata))}"
        if cache_key in self.authority_cache:
            return self.authority_cache[cache_key]
        
        # Base institutional power assessment
        base_score = np.random.uniform(0.4, 0.8)
        
        # Check for institutional indicators
        institutions = ["government", "corporation", "university", "foundation", "institute"]
        has_institutional_backing = any(inst in str(claim.source_metadata).lower() for inst in institutions)
        
        if has_institutional_backing:
            base_score *= 1.1
        
        # Check for funding transparency
        if "funding_source" in claim.source_metadata:
            base_score *= 1.05
        
        # Check for conflicts of interest
        if "conflicts_of_interest" in claim.source_metadata:
            conflicts = claim.source_metadata["conflicts_of_interest"]
            if conflicts:
                base_score *= 0.85  # Penalize disclosed conflicts
            else:
                base_score *= 1.05  # Boost for no conflicts
        
        # Corporate influence indicators
        corporate_terms = ["sponsored", "funded by", "partnership", "collaboration"]
        has_corporate_influence = any(term in str(claim.source_metadata).lower() for term in corporate_terms)
        
        if has_corporate_influence:
            base_score *= 0.9  # Slight penalty for corporate influence
        
        score = np.clip(base_score, 0.0, 1.0)
        self.authority_cache[cache_key] = score
        return score
    
    def _identify_bias_types(self, claim: Claim) -> List[str]:
        """Identify specific types of bias present in the claim"""
        bias_types = []
        
        content_lower = claim.content.lower()
        
        # Confirmation bias
        if any(word in content_lower for word in ["proves", "confirms", "validates"]):
            bias_types.append("confirmation_bias")
        
        # Selection bias
        if any(word in content_lower for word in ["some", "many", "few", "most"]):
            bias_types.append("potential_selection_bias")
        
        # Cultural bias
        cultural_terms = ["western", "eastern", "civilized", "primitive", "advanced"]
        if any(term in content_lower for term in cultural_terms):
            bias_types.append("cultural_bias")
        
        # Linguistic bias
        absolute_terms = ["all", "never", "always", "every", "none"]
        if any(term in content_lower for term in absolute_terms):
            bias_types.append("linguistic_bias")
        
        return bias_types
    
    def get_applicable_frameworks(self) -> List[VerificationFramework]:
        """Frameworks implemented by power dynamics verification"""
        return [VerificationFramework.CONSTRUCTIVIST]
    
    def validate_claim(self, claim: Claim) -> bool:
        """Check if claim benefits from power dynamics analysis"""
        # Power dynamics analysis is valuable for most claims
        return True
    
    def add_bias_model(self, model_name: str, model_config: Dict[str, Any]):
        """Add a new bias detection model"""
        self.bias_detection_models[model_name] = model_config
    
    def add_authority_network(self, network_name: str, network_config: Dict[str, Any]):
        """Add a new authority evaluation network"""
        self.authority_networks[network_name] = network_config
    
    def register_known_bias(self, bias_name: str, bias_patterns: List[str]):
        """Register patterns for a known type of bias"""
        self.known_biases[bias_name] = bias_patterns
    
    def clear_cache(self):
        """Clear the authority and bias cache"""
        self.authority_cache.clear()
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about power dynamics verification performance"""
        return {
            "bias_models_count": len(self.bias_detection_models),
            "authority_networks_count": len(self.authority_networks),
            "known_bias_types": len(self.known_biases),
            "cache_size": len(self.authority_cache),
            "applicable_frameworks": [f.value for f in self.get_applicable_frameworks()]
        }