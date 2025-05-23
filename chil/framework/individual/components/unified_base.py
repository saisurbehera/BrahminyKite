"""
Unified base class for verification components supporting both individual and consensus modes.

This module provides the enhanced base interface that all verification components
must implement to support the unified framework.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union
import logging
import numpy as np
from datetime import datetime, timedelta

from ..frameworks import VerificationFramework, VerificationResult, Claim, Domain
from ..consensus_types import (
    ConsensusProposal, NodeContext, ConsensusCriteria, 
    ConsensusVerificationResult, ApprovalStatus, VerificationMode
)


class UnifiedVerificationComponent(ABC):
    """
    Enhanced base class for verification components supporting both modes:
    - Individual verification (existing functionality)
    - Consensus verification (new functionality)
    """
    
    def __init__(self, component_name: str = None):
        self.component_name = component_name or self.__class__.__name__.replace("Verifier", "").lower()
        self.logger = logging.getLogger(f"{__name__}.{self.component_name}")
        self.mode = VerificationMode.INDIVIDUAL
        
        # Performance tracking
        self.verification_count = 0
        self.consensus_count = 0
        self.last_verification_time = None
        
        # Consensus-specific state
        self.consensus_cache = {}
        self.node_context: Optional[NodeContext] = None
        self.consensus_criteria_templates = {}
        
        # Learning and adaptation
        self.performance_history = []
        self.adaptation_rate = 0.1
        
    # ==================== ABSTRACT METHODS ====================
    
    @abstractmethod
    def verify_individual(self, claim: Claim) -> VerificationResult:
        """
        Individual claim verification (existing functionality)
        
        Args:
            claim: The claim to verify
            
        Returns:
            VerificationResult with score, evidence, and metadata
        """
        pass
    
    @abstractmethod
    def verify_consensus(self, proposal: ConsensusProposal, node_context: NodeContext) -> ConsensusVerificationResult:
        """
        Consensus proposal verification (new functionality)
        
        Args:
            proposal: The consensus proposal to verify
            node_context: Current node's context in the network
            
        Returns:
            ConsensusVerificationResult with approval status and evidence
        """
        pass
    
    @abstractmethod
    def prepare_consensus_criteria(self, proposal: ConsensusProposal) -> ConsensusCriteria:
        """
        Define criteria for consensus evaluation
        
        Args:
            proposal: The proposal to create criteria for
            
        Returns:
            ConsensusCriteria defining validation requirements
        """
        pass
    
    @abstractmethod
    def get_applicable_frameworks(self) -> List[VerificationFramework]:
        """
        Get the philosophical frameworks this component implements
        
        Returns:
            List of applicable verification frameworks
        """
        pass
    
    # ==================== UNIFIED INTERFACE METHODS ====================
    
    def verify(self, item: Union[Claim, ConsensusProposal], 
               context: Optional[NodeContext] = None) -> Union[VerificationResult, ConsensusVerificationResult]:
        """
        Unified verification interface that routes to appropriate method
        
        Args:
            item: Either a Claim (individual) or ConsensusProposal (consensus)
            context: Node context for consensus mode
            
        Returns:
            Appropriate verification result based on input type
        """
        self.verification_count += 1
        self.last_verification_time = datetime.now()
        
        try:
            if isinstance(item, Claim):
                return self._verify_individual_with_tracking(item)
            elif isinstance(item, ConsensusProposal):
                if context is None:
                    raise ValueError("NodeContext required for consensus verification")
                return self._verify_consensus_with_tracking(item, context)
            else:
                raise TypeError(f"Unsupported verification item type: {type(item)}")
                
        except Exception as e:
            self.logger.error(f"Verification failed: {e}")
            raise
    
    def _verify_individual_with_tracking(self, claim: Claim) -> VerificationResult:
        """Individual verification with performance tracking"""
        start_time = datetime.now()
        result = self.verify_individual(claim)
        end_time = datetime.now()
        
        # Track performance
        duration = (end_time - start_time).total_seconds()
        self.performance_history.append({
            "type": "individual",
            "duration": duration,
            "score": result.score,
            "timestamp": start_time
        })
        
        return result
    
    def _verify_consensus_with_tracking(self, proposal: ConsensusProposal, 
                                      node_context: NodeContext) -> ConsensusVerificationResult:
        """Consensus verification with performance tracking"""
        start_time = datetime.now()
        self.consensus_count += 1
        
        result = self.verify_consensus(proposal, node_context)
        end_time = datetime.now()
        
        # Track performance
        duration = (end_time - start_time).total_seconds()
        self.performance_history.append({
            "type": "consensus",
            "duration": duration,
            "confidence": result.confidence,
            "approval": result.approval_status.value,
            "timestamp": start_time
        })
        
        return result
    
    # ==================== CONSENSUS SUPPORT METHODS ====================
    
    def validate_consensus_promise(self, promise_criteria: ConsensusCriteria, 
                                 proposal: ConsensusProposal) -> bool:
        """
        Validate promises from other nodes in consensus
        
        Args:
            promise_criteria: Criteria promised by another node
            proposal: The proposal being evaluated
            
        Returns:
            True if promise is valid and acceptable
        """
        try:
            # Check if criteria is compatible with our requirements
            our_criteria = self.prepare_consensus_criteria(proposal)
            
            # Validate threshold compatibility
            if promise_criteria.acceptance_threshold > our_criteria.acceptance_threshold:
                self.logger.warning(f"Promise threshold too high: {promise_criteria.acceptance_threshold}")
                return False
            
            # Check required evidence compatibility
            missing_evidence = set(our_criteria.required_evidence) - set(promise_criteria.required_evidence)
            if missing_evidence:
                self.logger.warning(f"Missing required evidence: {missing_evidence}")
                return False
            
            # Validate conditions
            incompatible_conditions = set(promise_criteria.exclusions) & set(our_criteria.conditions)
            if incompatible_conditions:
                self.logger.warning(f"Incompatible conditions: {incompatible_conditions}")
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Promise validation failed: {e}")
            return False
    
    def can_validate_criteria(self, criteria: ConsensusCriteria) -> bool:
        """
        Check if this component can validate the given criteria
        
        Args:
            criteria: Criteria to check
            
        Returns:
            True if component can handle these criteria
        """
        # Check if this is our verifier type
        if criteria.verifier_type != self.component_name:
            return False
        
        # Check if we have templates for this type of validation
        return self._has_validation_capability(criteria)
    
    def _has_validation_capability(self, criteria: ConsensusCriteria) -> bool:
        """Check if we have the capability to validate these criteria"""
        # Default implementation - can be overridden by specific components
        required_capabilities = criteria.validation_rules.get("required_capabilities", [])
        our_capabilities = self.get_validation_capabilities()
        
        return all(cap in our_capabilities for cap in required_capabilities)
    
    def get_validation_capabilities(self) -> List[str]:
        """
        Get list of validation capabilities this component provides
        Should be overridden by specific components
        """
        return ["basic_verification"]
    
    def get_validation_conditions(self, criteria: ConsensusCriteria) -> List[str]:
        """
        Get conditions this component requires for validation
        
        Args:
            criteria: Input criteria
            
        Returns:
            List of conditions required for validation
        """
        conditions = []
        
        # Add framework-specific conditions
        frameworks = self.get_applicable_frameworks()
        for framework in frameworks:
            conditions.append(f"framework_{framework.value}_applicable")
        
        # Add component-specific conditions
        conditions.extend(self._get_component_specific_conditions(criteria))
        
        return conditions
    
    def _get_component_specific_conditions(self, criteria: ConsensusCriteria) -> List[str]:
        """Get component-specific validation conditions - override in subclasses"""
        return []
    
    # ==================== MODE MANAGEMENT ====================
    
    def set_mode(self, mode: VerificationMode, node_context: Optional[NodeContext] = None):
        """
        Set the operational mode for this component
        
        Args:
            mode: The verification mode to use
            node_context: Node context for consensus mode
        """
        self.mode = mode
        self.node_context = node_context
        
        if mode == VerificationMode.CONSENSUS and node_context is None:
            self.logger.warning("Consensus mode set without node context")
    
    def get_mode(self) -> VerificationMode:
        """Get current operational mode"""
        return self.mode
    
    # ==================== PERFORMANCE AND MONITORING ====================
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics for this component"""
        if not self.performance_history:
            return {
                "total_verifications": 0,
                "individual_count": 0,
                "consensus_count": 0,
                "average_duration": 0.0
            }
        
        individual_history = [h for h in self.performance_history if h["type"] == "individual"]
        consensus_history = [h for h in self.performance_history if h["type"] == "consensus"]
        
        stats = {
            "total_verifications": self.verification_count,
            "individual_count": len(individual_history),
            "consensus_count": len(consensus_history),
            "average_duration": np.mean([h["duration"] for h in self.performance_history]),
            "last_verification": self.last_verification_time.isoformat() if self.last_verification_time else None
        }
        
        if individual_history:
            stats["individual_avg_score"] = np.mean([h["score"] for h in individual_history])
            stats["individual_avg_duration"] = np.mean([h["duration"] for h in individual_history])
        
        if consensus_history:
            stats["consensus_avg_confidence"] = np.mean([h["confidence"] for h in consensus_history])
            stats["consensus_avg_duration"] = np.mean([h["duration"] for h in consensus_history])
            stats["consensus_approval_rate"] = len([h for h in consensus_history if h["approval"] == "approve"]) / len(consensus_history)
        
        return stats
    
    def clear_performance_history(self):
        """Clear performance tracking history"""
        self.performance_history = []
        self.verification_count = 0
        self.consensus_count = 0
    
    # ==================== UTILITY METHODS ====================
    
    def get_component_name(self) -> str:
        """Get the name of this component"""
        return self.component_name
    
    def validate_claim(self, claim: Claim) -> bool:
        """
        Validate that a claim is suitable for this component
        Default implementation - can be overridden
        """
        return True
    
    def validate_proposal(self, proposal: ConsensusProposal) -> bool:
        """
        Validate that a proposal is suitable for this component
        Default implementation - can be overridden
        """
        return True
    
    def get_confidence_bounds(self, base_score: float, uncertainty: float = 0.1) -> tuple[float, float]:
        """
        Calculate confidence interval for a verification score
        
        Args:
            base_score: The base verification score
            uncertainty: The uncertainty level (default 0.1)
            
        Returns:
            Tuple of (lower_bound, upper_bound)
        """
        lower = max(0.0, base_score - uncertainty)
        upper = min(1.0, base_score + uncertainty)
        return (lower, upper)
    
    def create_consensus_result(self, proposal_id: str, approval_status: ApprovalStatus,
                              confidence: float, evidence: Dict[str, Any] = None,
                              reasoning: str = "", conditions: List[str] = None) -> ConsensusVerificationResult:
        """
        Helper method to create standardized consensus verification results
        
        Args:
            proposal_id: ID of the proposal being verified
            approval_status: The approval decision
            confidence: Confidence in the decision (0-1)
            evidence: Supporting evidence dictionary
            reasoning: Human-readable reasoning
            conditions: List of conditions for approval
            
        Returns:
            ConsensusVerificationResult instance
        """
        frameworks = self.get_applicable_frameworks()
        primary_framework = frameworks[0] if frameworks else None
        
        return ConsensusVerificationResult(
            verifier_type=self.component_name,
            proposal_id=proposal_id,
            approval_status=approval_status,
            confidence=confidence,
            evidence=evidence or {},
            conditions=conditions or [],
            expiry_time=datetime.now() + timedelta(minutes=30),  # 30 min default expiry
            node_id=self.node_context.node_id if self.node_context else "unknown",
            framework=primary_framework,
            reasoning=reasoning
        )
    
    # ==================== CONSENSUS CACHING ====================
    
    def cache_consensus_result(self, proposal_id: str, result: ConsensusVerificationResult):
        """Cache consensus result for reuse"""
        self.consensus_cache[proposal_id] = {
            "result": result,
            "timestamp": datetime.now(),
            "expires": result.expiry_time
        }
    
    def get_cached_consensus_result(self, proposal_id: str) -> Optional[ConsensusVerificationResult]:
        """Get cached consensus result if valid"""
        if proposal_id not in self.consensus_cache:
            return None
        
        cache_entry = self.consensus_cache[proposal_id]
        if cache_entry["expires"] and datetime.now() > cache_entry["expires"]:
            del self.consensus_cache[proposal_id]
            return None
        
        return cache_entry["result"]
    
    def clear_consensus_cache(self):
        """Clear consensus result cache"""
        self.consensus_cache = {}
    
    # ==================== LEARNING AND ADAPTATION ====================
    
    def learn_from_consensus_feedback(self, proposal_id: str, actual_outcome: Dict[str, Any]):
        """
        Learn from consensus outcomes to improve future verification
        
        Args:
            proposal_id: ID of the proposal
            actual_outcome: The actual outcome of the consensus
        """
        # Find our previous result
        cached_result = self.get_cached_consensus_result(proposal_id)
        if not cached_result:
            return
        
        # Calculate learning update
        predicted_confidence = cached_result.confidence
        actual_success = actual_outcome.get("success", False)
        
        # Simple adaptation - adjust confidence calibration
        error = abs(predicted_confidence - (1.0 if actual_success else 0.0))
        learning_signal = error * self.adaptation_rate
        
        # Store learning record
        learning_record = {
            "proposal_id": proposal_id,
            "predicted_confidence": predicted_confidence,
            "actual_success": actual_success,
            "error": error,
            "learning_signal": learning_signal,
            "timestamp": datetime.now()
        }
        
        self.performance_history.append(learning_record)
        self.logger.info(f"Learned from consensus feedback: error={error:.3f}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive statistics about this component
        Backwards compatible with existing components
        """
        base_stats = {
            "component_name": self.component_name,
            "applicable_frameworks": [f.value for f in self.get_applicable_frameworks()],
            "current_mode": self.mode.value
        }
        
        # Add performance stats
        base_stats.update(self.get_performance_stats())
        
        # Add consensus-specific stats if applicable
        if self.consensus_count > 0:
            base_stats["consensus_cache_size"] = len(self.consensus_cache)
            base_stats["node_context_available"] = self.node_context is not None
        
        return base_stats