"""
Mode Bridge: Translation between Individual and Consensus modes

This module provides translation capabilities between individual claims
and consensus proposals, enabling seamless operation across both modes.
"""

import json
from typing import Dict, Any, Optional, List, Union
from datetime import datetime
import logging

from ..frameworks import Claim, Domain, VerificationResult
from ..consensus_types import (
    ConsensusProposal, ProposalType, ConsensusVerificationResult,
    UnifiedResult, MergingStrategy, VerificationMode
)


class ModeBridge:
    """
    Bridges individual and consensus verification modes through translation
    and cross-mode analysis capabilities.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.translation_history = []
        self.cross_analysis_cache = {}
        
        # Translation templates for different content types
        self.proposal_templates = {
            ProposalType.CLAIM_VALIDATION: self._create_claim_validation_template,
            ProposalType.MODEL_UPDATE: self._create_model_update_template,
            ProposalType.POLICY_CHANGE: self._create_policy_change_template,
            ProposalType.ETHICAL_GUIDELINE: self._create_ethical_guideline_template
        }
    
    # ==================== CLAIM → PROPOSAL TRANSLATION ====================
    
    def translate_claim_to_proposal(self, claim: Claim, 
                                  proposal_type: ProposalType = ProposalType.CLAIM_VALIDATION,
                                  proposer_id: str = "unknown",
                                  priority_level: int = 3,
                                  timeout: int = 30) -> ConsensusProposal:
        """
        Convert individual claim to consensus proposal
        
        Args:
            claim: The claim to convert
            proposal_type: Type of consensus proposal to create
            proposer_id: ID of the proposer
            priority_level: Priority level (1=critical, 5=routine)
            timeout: Consensus timeout in seconds
            
        Returns:
            ConsensusProposal for consensus verification
        """
        try:
            # Use appropriate template based on proposal type
            if proposal_type in self.proposal_templates:
                content = self.proposal_templates[proposal_type](claim)
            else:
                content = self._create_default_template(claim)
            
            # Create the proposal
            proposal = ConsensusProposal(
                proposal_type=proposal_type,
                content=content,
                metadata=self._enhance_metadata_for_consensus(claim),
                domain=claim.domain,
                priority_level=priority_level,
                timeout=timeout,
                proposer_id=proposer_id,
                required_verifiers=self._determine_required_verifiers(claim, proposal_type)
            )
            
            # Record translation
            self._record_translation("claim_to_proposal", claim, proposal)
            
            return proposal
            
        except Exception as e:
            self.logger.error(f"Failed to translate claim to proposal: {e}")
            raise
    
    def _create_claim_validation_template(self, claim: Claim) -> Dict[str, Any]:
        """Create content template for claim validation proposals"""
        return {
            "claim_text": claim.content,
            "validation_request": "Verify the truth and validity of this claim",
            "original_context": claim.context,
            "evidence_requirements": self._extract_evidence_requirements(claim),
            "validation_scope": self._determine_validation_scope(claim),
            "expected_frameworks": self._suggest_applicable_frameworks(claim)
        }
    
    def _create_model_update_template(self, claim: Claim) -> Dict[str, Any]:
        """Create content template for model update proposals"""
        return {
            "update_description": claim.content,
            "justification": f"Model update based on: {claim.content}",
            "affected_components": self._identify_affected_components(claim),
            "performance_expectations": self._extract_performance_claims(claim),
            "risk_assessment": self._assess_update_risks(claim),
            "rollback_plan": "Standard rollback procedures apply",
            "validation_metrics": self._define_validation_metrics(claim)
        }
    
    def _create_policy_change_template(self, claim: Claim) -> Dict[str, Any]:
        """Create content template for policy change proposals"""
        return {
            "policy_statement": claim.content,
            "rationale": f"Policy change justified by: {claim.content}",
            "scope": self._determine_policy_scope(claim),
            "stakeholders": self._identify_stakeholders(claim),
            "implementation_plan": self._create_implementation_outline(claim),
            "impact_analysis": self._analyze_policy_impact(claim),
            "compliance_requirements": self._identify_compliance_needs(claim)
        }
    
    def _create_ethical_guideline_template(self, claim: Claim) -> Dict[str, Any]:
        """Create content template for ethical guideline proposals"""
        return {
            "guideline_text": claim.content,
            "ethical_basis": self._identify_ethical_foundations(claim),
            "scope_of_application": self._determine_ethical_scope(claim),
            "enforcement_mechanism": "Standard ethical review processes",
            "training_requirements": self._identify_training_needs(claim),
            "monitoring_plan": self._create_monitoring_plan(claim),
            "cultural_considerations": self._assess_cultural_factors(claim)
        }
    
    def _create_default_template(self, claim: Claim) -> Dict[str, Any]:
        """Create default content template"""
        return {
            "description": claim.content,
            "context": claim.context,
            "domain": claim.domain.value,
            "requires_verification": True,
            "source_metadata": claim.source_metadata
        }
    
    # ==================== PROPOSAL → CLAIM TRANSLATION ====================
    
    def translate_proposal_to_claim(self, proposal: ConsensusProposal,
                                   extract_mode: str = "primary") -> Claim:
        """
        Convert consensus proposal to individual claim
        
        Args:
            proposal: The consensus proposal to convert
            extract_mode: How to extract claim ("primary", "detailed", "summary")
            
        Returns:
            Claim for individual verification
        """
        try:
            if extract_mode == "primary":
                content = self._extract_primary_claim(proposal)
            elif extract_mode == "detailed":
                content = self._extract_detailed_claim(proposal)
            else:  # summary
                content = self._extract_summary_claim(proposal)
            
            # Create the claim
            claim = Claim(
                content=content,
                domain=proposal.domain or Domain.EMPIRICAL,  # Default fallback
                context=self._extract_context_from_proposal(proposal),
                source_metadata=self._extract_source_metadata(proposal)
            )
            
            # Record translation
            self._record_translation("proposal_to_claim", proposal, claim)
            
            return claim
            
        except Exception as e:
            self.logger.error(f"Failed to translate proposal to claim: {e}")
            raise
    
    def _extract_primary_claim(self, proposal: ConsensusProposal) -> str:
        """Extract the primary claim from proposal content"""
        content = proposal.content
        
        # Try different content fields based on proposal type
        claim_fields = [
            "claim_text", "policy_statement", "guideline_text", 
            "update_description", "description"
        ]
        
        for field in claim_fields:
            if field in content and content[field]:
                return content[field]
        
        # Fallback: create summary from available content
        return f"Consensus proposal: {json.dumps(content, indent=None)[:200]}..."
    
    def _extract_detailed_claim(self, proposal: ConsensusProposal) -> str:
        """Extract detailed claim with context"""
        primary = self._extract_primary_claim(proposal)
        
        additional_info = []
        content = proposal.content
        
        # Add justification if available
        if "justification" in content:
            additional_info.append(f"Justification: {content['justification']}")
        
        # Add rationale if available
        if "rationale" in content:
            additional_info.append(f"Rationale: {content['rationale']}")
        
        # Add scope information
        if "scope" in content:
            additional_info.append(f"Scope: {content['scope']}")
        
        if additional_info:
            return f"{primary} [{'; '.join(additional_info)}]"
        
        return primary
    
    def _extract_summary_claim(self, proposal: ConsensusProposal) -> str:
        """Extract summarized claim"""
        primary = self._extract_primary_claim(proposal)
        
        # Truncate if too long
        if len(primary) > 150:
            return primary[:147] + "..."
        
        return primary
    
    # ==================== CROSS-MODE ANALYSIS ====================
    
    def cross_mode_analysis(self, item: Union[Claim, ConsensusProposal],
                           individual_verifier: Any = None,
                           consensus_verifier: Any = None) -> Dict[str, Any]:
        """
        Analyze using both individual and consensus perspectives
        
        Args:
            item: Either a Claim or ConsensusProposal
            individual_verifier: Verifier for individual mode
            consensus_verifier: Verifier for consensus mode
            
        Returns:
            Cross-mode analysis results
        """
        analysis_id = f"cross_analysis_{hash(str(item))}"
        
        # Check cache
        if analysis_id in self.cross_analysis_cache:
            cached_result = self.cross_analysis_cache[analysis_id]
            if not self._is_cache_expired(cached_result):
                return cached_result["analysis"]
        
        analysis = {
            "analysis_id": analysis_id,
            "timestamp": datetime.now().isoformat(),
            "original_item_type": type(item).__name__,
            "individual_analysis": None,
            "consensus_analysis": None,
            "comparison": None,
            "recommendations": []
        }
        
        try:
            if isinstance(item, Claim):
                # Primary: individual analysis
                if individual_verifier:
                    analysis["individual_analysis"] = individual_verifier.verify(item)
                
                # Secondary: convert to proposal and analyze
                if consensus_verifier:
                    proposal = self.translate_claim_to_proposal(item)
                    # Note: This would need node context in real implementation
                    # analysis["consensus_analysis"] = consensus_verifier.verify_consensus(proposal, node_context)
                    analysis["consensus_analysis"] = "Requires node context for consensus analysis"
            
            elif isinstance(item, ConsensusProposal):
                # Primary: consensus analysis
                if consensus_verifier:
                    # Note: This would need node context in real implementation
                    analysis["consensus_analysis"] = "Requires node context for consensus analysis"
                
                # Secondary: convert to claim and analyze
                if individual_verifier:
                    claim = self.translate_proposal_to_claim(item)
                    analysis["individual_analysis"] = individual_verifier.verify(claim)
            
            # Generate comparison and recommendations
            analysis["comparison"] = self._compare_analyses(
                analysis["individual_analysis"], 
                analysis["consensus_analysis"]
            )
            analysis["recommendations"] = self._generate_cross_mode_recommendations(analysis)
            
            # Cache result
            self._cache_cross_analysis(analysis_id, analysis)
            
        except Exception as e:
            self.logger.error(f"Cross-mode analysis failed: {e}")
            analysis["error"] = str(e)
        
        return analysis
    
    def _compare_analyses(self, individual_result: Any, consensus_result: Any) -> Dict[str, Any]:
        """Compare individual and consensus analysis results"""
        comparison = {
            "both_available": individual_result is not None and consensus_result is not None,
            "agreement_level": 0.0,
            "discrepancies": [],
            "confidence_comparison": None,
            "framework_alignment": None
        }
        
        if comparison["both_available"] and hasattr(individual_result, 'score'):
            # Compare scores if both are available
            ind_score = getattr(individual_result, 'score', 0.0)
            cons_confidence = getattr(consensus_result, 'confidence', 0.0)
            
            # Calculate agreement level
            score_diff = abs(ind_score - cons_confidence)
            comparison["agreement_level"] = 1.0 - min(score_diff, 1.0)
            
            # Identify discrepancies
            if score_diff > 0.3:
                comparison["discrepancies"].append("Significant score difference")
            
            comparison["confidence_comparison"] = {
                "individual_score": ind_score,
                "consensus_confidence": cons_confidence,
                "difference": score_diff
            }
        
        return comparison
    
    def _generate_cross_mode_recommendations(self, analysis: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on cross-mode analysis"""
        recommendations = []
        
        comparison = analysis.get("comparison", {})
        
        if comparison.get("both_available"):
            agreement_level = comparison.get("agreement_level", 0.0)
            
            if agreement_level > 0.8:
                recommendations.append("High agreement between modes - result is reliable")
            elif agreement_level > 0.6:
                recommendations.append("Moderate agreement - consider additional validation")
            else:
                recommendations.append("Low agreement - investigate discrepancies")
                recommendations.append("Consider domain-specific validation approaches")
        
        else:
            if analysis.get("individual_analysis"):
                recommendations.append("Individual analysis available - consider consensus validation")
            if analysis.get("consensus_analysis"):
                recommendations.append("Consensus analysis available - consider individual validation")
        
        return recommendations
    
    # ==================== HELPER METHODS ====================
    
    def _enhance_metadata_for_consensus(self, claim: Claim) -> Dict[str, Any]:
        """Enhance claim metadata for consensus use"""
        enhanced_metadata = claim.context.copy()
        
        enhanced_metadata.update({
            "translation_timestamp": datetime.now().isoformat(),
            "original_claim_length": len(claim.content),
            "domain": claim.domain.value,
            "consensus_ready": True,
            "requires_verification": True
        })
        
        return enhanced_metadata
    
    def _determine_required_verifiers(self, claim: Claim, proposal_type: ProposalType) -> List[str]:
        """Determine which verifiers are required based on claim and proposal type"""
        verifiers = []
        
        # Always include empirical for most claims
        if claim.domain in [Domain.EMPIRICAL, Domain.LOGICAL, Domain.SOCIAL]:
            verifiers.append("empirical")
        
        # Add contextual for interpretation-heavy domains
        if claim.domain in [Domain.AESTHETIC, Domain.CREATIVE, Domain.SOCIAL]:
            verifiers.append("contextual")
        
        # Add consistency for logical and ethical claims
        if claim.domain in [Domain.LOGICAL, Domain.ETHICAL]:
            verifiers.append("consistency")
        
        # Add power dynamics for social and ethical claims
        if claim.domain in [Domain.SOCIAL, Domain.ETHICAL]:
            verifiers.append("power_dynamics")
        
        # Add utility for policy and practical claims
        if proposal_type in [ProposalType.POLICY_CHANGE, ProposalType.MODEL_UPDATE]:
            verifiers.append("utility")
        
        # Always include evolutionary for learning
        verifiers.append("evolutionary")
        
        return verifiers
    
    def _extract_evidence_requirements(self, claim: Claim) -> List[str]:
        """Extract evidence requirements from claim"""
        requirements = ["logical_consistency"]
        
        if claim.domain == Domain.EMPIRICAL:
            requirements.extend(["empirical_data", "statistical_validation"])
        elif claim.domain == Domain.ETHICAL:
            requirements.extend(["ethical_framework_alignment", "stakeholder_analysis"])
        elif claim.domain == Domain.AESTHETIC:
            requirements.extend(["cultural_context", "expert_opinion"])
        
        return requirements
    
    def _determine_validation_scope(self, claim: Claim) -> str:
        """Determine validation scope based on claim"""
        if "global" in claim.content.lower() or "universal" in claim.content.lower():
            return "global"
        elif "local" in claim.content.lower() or "specific" in claim.content.lower():
            return "local"
        else:
            return "standard"
    
    def _suggest_applicable_frameworks(self, claim: Claim) -> List[str]:
        """Suggest applicable verification frameworks"""
        frameworks = []
        
        if claim.domain == Domain.EMPIRICAL:
            frameworks.extend(["positivist", "correspondence"])
        elif claim.domain == Domain.AESTHETIC:
            frameworks.extend(["interpretivist", "constructivist"])
        elif claim.domain == Domain.ETHICAL:
            frameworks.extend(["coherence", "pragmatist"])
        elif claim.domain == Domain.SOCIAL:
            frameworks.extend(["interpretivist", "constructivist", "pragmatist"])
        
        return frameworks
    
    # Methods for different template types (simplified implementations)
    def _identify_affected_components(self, claim: Claim) -> List[str]:
        return ["core_model", "validation_pipeline"]
    
    def _extract_performance_claims(self, claim: Claim) -> Dict[str, Any]:
        return {"expected_improvement": "TBD", "baseline_metrics": "current_performance"}
    
    def _assess_update_risks(self, claim: Claim) -> Dict[str, str]:
        return {"risk_level": "medium", "mitigation_strategy": "gradual_rollout"}
    
    def _define_validation_metrics(self, claim: Claim) -> List[str]:
        return ["accuracy", "precision", "recall", "f1_score"]
    
    def _determine_policy_scope(self, claim: Claim) -> str:
        return "organizational"
    
    def _identify_stakeholders(self, claim: Claim) -> List[str]:
        return ["users", "developers", "administrators"]
    
    def _create_implementation_outline(self, claim: Claim) -> Dict[str, str]:
        return {"phase1": "planning", "phase2": "implementation", "phase3": "evaluation"}
    
    def _analyze_policy_impact(self, claim: Claim) -> Dict[str, str]:
        return {"expected_impact": "positive", "monitoring_required": True}
    
    def _identify_compliance_needs(self, claim: Claim) -> List[str]:
        return ["privacy_compliance", "ethical_guidelines"]
    
    def _identify_ethical_foundations(self, claim: Claim) -> List[str]:
        return ["beneficence", "autonomy", "justice", "non_maleficence"]
    
    def _determine_ethical_scope(self, claim: Claim) -> str:
        return "system_wide"
    
    def _identify_training_needs(self, claim: Claim) -> List[str]:
        return ["ethics_training", "implementation_guidelines"]
    
    def _create_monitoring_plan(self, claim: Claim) -> Dict[str, str]:
        return {"frequency": "quarterly", "metrics": "compliance_rate"}
    
    def _assess_cultural_factors(self, claim: Claim) -> Dict[str, str]:
        return {"cultural_sensitivity": "required", "localization": "recommended"}
    
    def _extract_context_from_proposal(self, proposal: ConsensusProposal) -> Dict[str, Any]:
        """Extract context for claim from proposal"""
        context = proposal.metadata.copy()
        
        # Add proposal-specific context
        context.update({
            "from_consensus_proposal": True,
            "proposal_type": proposal.proposal_type.value,
            "priority_level": proposal.priority_level,
            "consensus_timeout": proposal.timeout
        })
        
        return context
    
    def _extract_source_metadata(self, proposal: ConsensusProposal) -> Dict[str, Any]:
        """Extract source metadata from proposal"""
        return {
            "proposer_id": proposal.proposer_id,
            "proposal_id": proposal.proposal_id,
            "timestamp": proposal.timestamp.isoformat(),
            "required_verifiers": proposal.required_verifiers
        }
    
    def _record_translation(self, translation_type: str, source: Any, target: Any):
        """Record translation for audit and learning"""
        record = {
            "type": translation_type,
            "timestamp": datetime.now().isoformat(),
            "source_type": type(source).__name__,
            "target_type": type(target).__name__,
            "source_id": getattr(source, 'proposal_id', hash(str(source))),
            "target_id": getattr(target, 'proposal_id', hash(str(target)))
        }
        
        self.translation_history.append(record)
        
        # Keep history manageable
        if len(self.translation_history) > 1000:
            self.translation_history = self.translation_history[-1000:]
    
    def _cache_cross_analysis(self, analysis_id: str, analysis: Dict[str, Any]):
        """Cache cross-mode analysis result"""
        self.cross_analysis_cache[analysis_id] = {
            "analysis": analysis,
            "timestamp": datetime.now(),
            "expires_at": datetime.now().timestamp() + 3600  # 1 hour expiry
        }
    
    def _is_cache_expired(self, cache_entry: Dict[str, Any]) -> bool:
        """Check if cache entry is expired"""
        return datetime.now().timestamp() > cache_entry["expires_at"]
    
    def get_translation_statistics(self) -> Dict[str, Any]:
        """Get translation statistics"""
        if not self.translation_history:
            return {"total_translations": 0}
        
        claim_to_proposal = len([r for r in self.translation_history if r["type"] == "claim_to_proposal"])
        proposal_to_claim = len([r for r in self.translation_history if r["type"] == "proposal_to_claim"])
        
        return {
            "total_translations": len(self.translation_history),
            "claim_to_proposal": claim_to_proposal,
            "proposal_to_claim": proposal_to_claim,
            "cross_analyses_cached": len(self.cross_analysis_cache)
        }
    
    def clear_caches(self):
        """Clear all caches and history"""
        self.translation_history.clear()
        self.cross_analysis_cache.clear()