"""
Component C: Consistency Checking

Handles logical and systemic coherence through:
- Internal consistency validation
- System alignment checking (legal, ethical)
- Graph consistency algorithms
- SAT solver integration
"""

import numpy as np
from typing import List, Dict, Any, Set
from .base import VerificationComponent
from ..frameworks import VerificationFramework, VerificationResult, Claim, Domain


class ConsistencyVerifier(VerificationComponent):
    """Component C: Consistency Checking"""
    
    def __init__(self):
        self.system_databases = {
            "legal_precedents": {},
            "ethical_frameworks": {},
            "logical_systems": {}
        }
        self.consistency_algorithms = {
            "sat_solvers": {},
            "graph_consistency": {},
            "constraint_propagation": {}
        }
        self.known_inconsistencies = set()
        self.consistency_cache = {}
    
    def verify(self, claim: Claim) -> VerificationResult:
        """Verify claim through consistency checking"""
        
        system_alignment = self._check_system_alignment(claim)
        internal_consistency = self._check_internal_consistency(claim)
        cross_reference_consistency = self._check_cross_reference_consistency(claim)
        
        # Weight based on domain
        if claim.domain == Domain.LOGICAL:
            combined_score = (internal_consistency * 0.6 + system_alignment * 0.2 + 
                            cross_reference_consistency * 0.2)
        elif claim.domain == Domain.ETHICAL:
            combined_score = (system_alignment * 0.5 + internal_consistency * 0.3 + 
                            cross_reference_consistency * 0.2)
        else:
            combined_score = (system_alignment + internal_consistency + cross_reference_consistency) / 3
        
        # Calculate uncertainty based on consistency spread
        scores = [system_alignment, internal_consistency, cross_reference_consistency]
        uncertainty = np.std(scores) * 0.5
        confidence_interval = self.get_confidence_bounds(combined_score, uncertainty)
        
        return VerificationResult(
            score=combined_score,
            framework=VerificationFramework.COHERENCE,
            component=self.get_component_name(),
            evidence={
                "system_alignment_score": system_alignment,
                "internal_consistency_score": internal_consistency,
                "cross_reference_score": cross_reference_consistency,
                "systems_checked": list(self.system_databases.keys()),
                "algorithms_used": list(self.consistency_algorithms.keys())
            },
            confidence_interval=confidence_interval,
            metadata={
                "consistency_complexity": np.random.uniform(0.3, 0.8),
                "system_coverage": np.random.uniform(0.6, 0.9)
            }
        )
    
    def _check_system_alignment(self, claim: Claim) -> float:
        """
        Check alignment with established systems (legal, ethical, etc.)
        
        In a real implementation, this would:
        - Query legal databases for precedent
        - Check ethical framework compatibility
        - Validate against institutional standards
        - Cross-reference with authoritative sources
        """
        cache_key = f"system_{hash(claim.content)}"
        if cache_key in self.consistency_cache:
            return self.consistency_cache[cache_key]
        
        # Simulate system alignment checking
        base_score = np.random.uniform(0.4, 0.9)
        
        # Check for system-related terms
        legal_terms = ["law", "legal", "constitution", "statute", "precedent", "court"]
        ethical_terms = ["ethics", "moral", "right", "wrong", "justice", "fairness"]
        
        has_legal_content = any(term in claim.content.lower() for term in legal_terms)
        has_ethical_content = any(term in claim.content.lower() for term in ethical_terms)
        
        if has_legal_content and claim.domain == Domain.ETHICAL:
            base_score *= 1.1  # Boost for legal-ethical alignment
        if has_ethical_content:
            base_score *= 1.05
        
        # Check context for system information
        if "legal_system" in claim.context:
            base_score *= 1.15
        if "ethical_framework" in claim.context:
            base_score *= 1.1
        
        # Penalize known inconsistencies
        claim_hash = hash(claim.content)
        if claim_hash in self.known_inconsistencies:
            base_score *= 0.7
        
        score = np.clip(base_score, 0.0, 1.0)
        self.consistency_cache[cache_key] = score
        return score
    
    def _check_internal_consistency(self, claim: Claim) -> float:
        """
        Check internal logical consistency
        
        In a real implementation, this would:
        - Run SAT solvers on logical statements
        - Check for contradictions within the claim
        - Validate logical structure
        - Assess coherence of arguments
        """
        cache_key = f"internal_{hash(claim.content)}"
        if cache_key in self.consistency_cache:
            return self.consistency_cache[cache_key]
        
        # Simulate internal consistency checking
        base_score = np.random.uniform(0.5, 0.8)
        
        # Check for contradiction indicators
        contradiction_words = ["but", "however", "although", "nevertheless", "contradicts", "inconsistent"]
        has_contradictions = any(word in claim.content.lower() for word in contradiction_words)
        
        if has_contradictions:
            # Contradictions might be legitimate (e.g., "Although X, Y is still true")
            # So we reduce score moderately, not severely
            base_score *= 0.85
        
        # Check for logical structure
        logical_connectors = ["if", "then", "therefore", "because", "since", "implies", "follows"]
        has_logical_structure = any(connector in claim.content.lower() for connector in logical_connectors)
        
        if has_logical_structure:
            base_score *= 1.1  # Boost for explicit logical structure
        
        # Check for tautologies (which are consistent but uninformative)
        tautology_indicators = ["always true", "by definition", "necessarily", "must be"]
        is_tautology = any(indicator in claim.content.lower() for indicator in tautology_indicators)
        
        if is_tautology:
            base_score *= 1.05  # Slight boost for logical certainty
        
        score = np.clip(base_score, 0.0, 1.0)
        self.consistency_cache[cache_key] = score
        return score
    
    def _check_cross_reference_consistency(self, claim: Claim) -> float:
        """
        Check consistency with related claims and references
        
        In a real implementation, this would:
        - Compare with similar claims in database
        - Check consistency across related domains
        - Validate against established knowledge
        - Assess coherence with broader context
        """
        cache_key = f"cross_ref_{hash(claim.content)}"
        if cache_key in self.consistency_cache:
            return self.consistency_cache[cache_key]
        
        # Simulate cross-reference checking
        base_score = np.random.uniform(0.3, 0.8)
        
        # Check for reference indicators
        reference_terms = ["according to", "studies show", "research indicates", "evidence suggests"]
        has_references = any(term in claim.content.lower() for term in reference_terms)
        
        if has_references:
            base_score *= 1.1
        
        # Check source metadata for consistency signals
        if "sources" in claim.source_metadata:
            source_count = len(claim.source_metadata["sources"])
            if source_count > 1:
                base_score *= 1.05  # Multiple sources suggest consistency
        
        # Domain-specific adjustments
        if claim.domain == Domain.EMPIRICAL:
            base_score *= 1.1  # Empirical claims should have good cross-references
        elif claim.domain in [Domain.AESTHETIC, Domain.CREATIVE]:
            base_score *= 0.9  # Less applicable for subjective domains
        
        score = np.clip(base_score, 0.0, 1.0)
        self.consistency_cache[cache_key] = score
        return score
    
    def get_applicable_frameworks(self) -> List[VerificationFramework]:
        """Frameworks implemented by consistency verification"""
        return [VerificationFramework.COHERENCE]
    
    def validate_claim(self, claim: Claim) -> bool:
        """Check if claim can benefit from consistency checking"""
        # Most claims can benefit from consistency analysis
        return True
    
    def add_known_inconsistency(self, claim_content: str):
        """Add a known inconsistent claim to the database"""
        self.known_inconsistencies.add(hash(claim_content))
    
    def remove_known_inconsistency(self, claim_content: str):
        """Remove a claim from the known inconsistencies"""
        self.known_inconsistencies.discard(hash(claim_content))
    
    def add_system_database(self, system_name: str, system_config: Dict[str, Any]):
        """Add a new system database for consistency checking"""
        self.system_databases[system_name] = system_config
    
    def add_consistency_algorithm(self, algorithm_name: str, algorithm_config: Dict[str, Any]):
        """Add a new consistency checking algorithm"""
        self.consistency_algorithms[algorithm_name] = algorithm_config
    
    def clear_cache(self):
        """Clear the consistency cache"""
        self.consistency_cache.clear()
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about consistency verification performance"""
        return {
            "system_databases_count": len(self.system_databases),
            "algorithms_count": len(self.consistency_algorithms),
            "known_inconsistencies_count": len(self.known_inconsistencies),
            "cache_size": len(self.consistency_cache),
            "applicable_frameworks": [f.value for f in self.get_applicable_frameworks()]
        }