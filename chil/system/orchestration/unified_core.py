"""
Unified Core Verifier System

Main orchestrator that integrates individual verification, consensus protocols,
and mode bridging for the complete unified verification framework.
"""

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Any, Optional, Union
import json
from datetime import datetime

from .frameworks import VerificationFramework, VerificationResult, Claim, Domain
from .consensus_types import (
    ConsensusProposal, NodeContext, ConsensusVerificationResult, ConsensusConfig,
    VerificationMode, UnifiedResult, MergingStrategy, ProposalType, NodeRole
)
from .components.unified_base import UnifiedVerificationComponent
from .components.empirical_unified import UnifiedEmpiricalVerifier
from .bridge.mode_bridge import ModeBridge
from .consensus.paxos import PhilosophicalPaxos
from .meta import MetaVerificationSystem
from .systems import DebateSystem

# Import original components for backward compatibility
from .components import (
    EmpiricalVerifier, ContextualVerifier, ConsistencyVerifier,
    PowerDynamicsVerifier, UtilityVerifier, EvolutionaryVerifier
)


class UnifiedIdealVerifier:
    """
    Unified verifier supporting both individual and consensus modes
    
    This system provides:
    - Individual claim verification (backward compatible)
    - Distributed consensus verification 
    - Seamless mode switching and bridging
    - Multi-framework philosophical integration
    - Adaptive learning and self-improvement
    """
    
    def __init__(self, 
                 mode: VerificationMode = VerificationMode.INDIVIDUAL,
                 consensus_config: Optional[ConsensusConfig] = None,
                 enable_unified_components: bool = True):
        
        self.mode = mode
        self.consensus_config = consensus_config or ConsensusConfig()
        self.logger = logging.getLogger(__name__)
        
        # Core systems
        self.mode_bridge = ModeBridge()
        self.meta_verifier = MetaVerificationSystem()
        self.debate_system = DebateSystem()
        
        # Component management
        self.enable_unified_components = enable_unified_components
        self.individual_components = {}
        self.unified_components = {}
        self.active_components = {}
        
        # Consensus system (initialized when needed)
        self.consensus_system: Optional[PhilosophicalPaxos] = None
        self.node_context: Optional[NodeContext] = None
        
        # Performance and monitoring
        self.verification_history = []
        self.consensus_history = []
        self.mode_switch_history = []
        
        # Configuration
        self.max_workers = 6
        self.verification_timeout = 30
        self.enable_parallel = True
        self.enable_debate = True
        
        # Initialize components
        self._initialize_components()
        
        # Initialize consensus if needed
        if mode in [VerificationMode.CONSENSUS, VerificationMode.HYBRID]:
            self._initialize_consensus_mode()
    
    def _initialize_components(self):
        """Initialize both individual and unified verification components"""
        
        # Initialize individual components (for backward compatibility)
        self.individual_components = {
            "empirical": EmpiricalVerifier(),
            "contextual": ContextualVerifier(),
            "consistency": ConsistencyVerifier(),
            "power_dynamics": PowerDynamicsVerifier(),
            "utility": UtilityVerifier(),
            "evolutionary": EvolutionaryVerifier()
        }
        
        # Initialize unified components (enhanced with consensus support)
        if self.enable_unified_components:
            self.unified_components = {
                "empirical": UnifiedEmpiricalVerifier(),
                # TODO: Create unified versions of other components
                # "contextual": UnifiedContextualVerifier(),
                # "consistency": UnifiedConsistencyVerifier(),
                # "power_dynamics": UnifiedPowerDynamicsVerifier(),
                # "utility": UnifiedUtilityVerifier(),
                # "evolutionary": UnifiedEvolutionaryVerifier()
            }
            
            # For now, use individual components for others
            self.unified_components.update({
                "contextual": ContextualVerifier(),
                "consistency": ConsistencyVerifier(),
                "power_dynamics": PowerDynamicsVerifier(),
                "utility": UtilityVerifier(),
                "evolutionary": EvolutionaryVerifier()
            })
        
        # Set active components based on configuration
        self.active_components = (self.unified_components if self.enable_unified_components 
                                else self.individual_components)
        
        self.logger.info(f"Initialized {len(self.active_components)} verification components")
    
    def _initialize_consensus_mode(self):
        """Initialize consensus mode components"""
        if not self.node_context:
            self.node_context = NodeContext(
                node_id=self.consensus_config.node_id,
                node_role=NodeRole.PROPOSER,  # Default role
                peer_nodes=self.consensus_config.peer_nodes.copy(),
                philosophical_weights={f.value: 1.0 for f in VerificationFramework}
            )
        
        # Set mode for unified components
        for component in self.unified_components.values():
            if isinstance(component, UnifiedVerificationComponent):
                component.set_mode(self.mode, self.node_context)
        
        # Initialize consensus system
        if not self.consensus_system:
            self.consensus_system = PhilosophicalPaxos(
                self.node_context,
                self.consensus_config,
                self.active_components
            )
        
        self.logger.info(f"Initialized consensus mode with {len(self.node_context.peer_nodes)} peers")
    
    # ==================== INDIVIDUAL VERIFICATION (BACKWARD COMPATIBLE) ====================
    
    def verify(self, claim: Claim) -> Dict[str, Any]:
        """
        Individual claim verification (backward compatible API)
        
        Args:
            claim: The claim to verify
            
        Returns:
            Verification result dictionary
        """
        if self.mode == VerificationMode.CONSENSUS:
            self.logger.warning("Individual verification called in consensus mode - switching temporarily")
            return self._verify_individual_in_consensus_mode(claim)
        
        return self._verify_individual(claim)
    
    def _verify_individual(self, claim: Claim) -> Dict[str, Any]:
        """Core individual verification logic"""
        self.logger.info(f"Starting individual verification for: {claim.content[:100]}...")
        
        start_time = datetime.now()
        
        try:
            # Step 1: Run verification components
            component_results = self._run_individual_components(claim)
            
            if not component_results:
                return self._create_error_response("No verification components completed successfully")
            
            # Step 2: Get applicable frameworks
            applicable_frameworks = self._get_applicable_frameworks()
            
            # Step 3: Conduct debate if enabled
            debate_result = None
            if self.enable_debate and len(applicable_frameworks) > 1:
                debate_result = self.debate_system.conduct_debate(claim, applicable_frameworks)
            
            # Step 4: Meta-verification
            meta_result = self.meta_verifier.resolve_conflicts(component_results, claim)
            
            # Step 5: Calculate final score
            final_score = self._calculate_individual_final_score(meta_result, debate_result)
            
            # Step 6: Generate explanation
            explanation = self._generate_individual_explanation(claim, component_results, meta_result, debate_result)
            
            # Step 7: Create response
            response = {
                "claim": claim.content,
                "domain": claim.domain.value,
                "final_score": final_score,
                "confidence_interval": meta_result.confidence_interval,
                "dominant_framework": meta_result.framework.value,
                "meta_result": self._serialize_verification_result(meta_result),
                "component_results": {
                    name: self._serialize_verification_result(result) 
                    for name, result in zip(self.active_components.keys(), component_results)
                },
                "debate_result": debate_result,
                "explanation": explanation,
                "verification_mode": "individual",
                "metadata": {
                    "verification_timestamp": datetime.now().isoformat(),
                    "components_used": len(component_results),
                    "frameworks_considered": len(applicable_frameworks),
                    "processing_time": (datetime.now() - start_time).total_seconds()
                }
            }
            
            # Record verification
            self._record_individual_verification(claim, response)
            
            return response
            
        except Exception as e:
            self.logger.error(f"Individual verification failed: {e}")
            return self._create_error_response(f"Verification error: {str(e)}")
    
    def _verify_individual_in_consensus_mode(self, claim: Claim) -> Dict[str, Any]:
        """Handle individual verification when in consensus mode"""
        # Temporarily switch to hybrid mode
        original_mode = self.mode
        self.mode = VerificationMode.HYBRID
        
        try:
            result = self._verify_individual(claim)
            result["metadata"]["consensus_mode_context"] = True
            return result
        finally:
            self.mode = original_mode
    
    # ==================== CONSENSUS VERIFICATION ====================
    
    async def propose_consensus(self, proposal: ConsensusProposal) -> Dict[str, Any]:
        """
        Initiate consensus process as proposer
        
        Args:
            proposal: The consensus proposal
            
        Returns:
            Consensus result
        """
        if self.mode == VerificationMode.INDIVIDUAL:
            raise ValueError("Consensus operations not available in individual mode")
        
        if not self.consensus_system:
            self._initialize_consensus_mode()
        
        self.logger.info(f"Proposing consensus for: {proposal.proposal_id}")
        
        try:
            consensus_result = await self.consensus_system.propose_consensus(proposal)
            
            # Record consensus attempt
            self._record_consensus_attempt(proposal, consensus_result)
            
            return consensus_result
            
        except Exception as e:
            self.logger.error(f"Consensus proposal failed: {e}")
            return {
                "proposal_id": proposal.proposal_id,
                "success": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    async def participate_consensus(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Participate in consensus as acceptor
        
        Args:
            request: Consensus request from proposer
            
        Returns:
            Participation result
        """
        if self.mode == VerificationMode.INDIVIDUAL:
            raise ValueError("Consensus operations not available in individual mode")
        
        if not self.consensus_system:
            self._initialize_consensus_mode()
        
        try:
            # Handle different types of consensus messages
            message_type = request.get("message_type", "unknown")
            message_data = request.get("data", {})
            
            response = await self.consensus_system.handle_message(message_type, message_data)
            
            return {
                "success": True,
                "response": response,
                "node_id": self.node_context.node_id,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Consensus participation failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "node_id": self.node_context.node_id if self.node_context else "unknown",
                "timestamp": datetime.now().isoformat()
            }
    
    # ==================== UNIFIED OPERATIONS ====================
    
    def switch_mode(self, new_mode: VerificationMode, config: Optional[Dict[str, Any]] = None):
        """
        Switch between verification modes
        
        Args:
            new_mode: The new verification mode
            config: Additional configuration for the new mode
        """
        old_mode = self.mode
        
        self.logger.info(f"Switching mode from {old_mode.value} to {new_mode.value}")
        
        try:
            # Record mode switch
            switch_record = {
                "from_mode": old_mode.value,
                "to_mode": new_mode.value,
                "timestamp": datetime.now().isoformat(),
                "config": config
            }
            
            # Handle mode-specific initialization
            if new_mode in [VerificationMode.CONSENSUS, VerificationMode.HYBRID]:
                if config and "consensus_config" in config:
                    self.consensus_config = ConsensusConfig(**config["consensus_config"])
                self._initialize_consensus_mode()
            
            # Update component modes
            for component in self.unified_components.values():
                if isinstance(component, UnifiedVerificationComponent):
                    component.set_mode(new_mode, self.node_context)
            
            self.mode = new_mode
            switch_record["success"] = True
            
            self.mode_switch_history.append(switch_record)
            self.logger.info(f"Successfully switched to {new_mode.value} mode")
            
        except Exception as e:
            self.logger.error(f"Mode switch failed: {e}")
            switch_record["success"] = False
            switch_record["error"] = str(e)
            self.mode_switch_history.append(switch_record)
            raise
    
    def cross_mode_analysis(self, item: Union[Claim, ConsensusProposal]) -> Dict[str, Any]:
        """
        Analyze using both individual and consensus perspectives
        
        Args:
            item: Either a Claim or ConsensusProposal
            
        Returns:
            Cross-mode analysis results
        """
        if self.mode == VerificationMode.INDIVIDUAL:
            self.logger.warning("Cross-mode analysis requested in individual mode - limited functionality")
        
        return self.mode_bridge.cross_mode_analysis(
            item,
            individual_verifier=self,
            consensus_verifier=self.consensus_system
        )
    
    def unified_verify(self, item: Union[Claim, ConsensusProposal], 
                      force_mode: Optional[VerificationMode] = None) -> UnifiedResult:
        """
        Unified verification that handles both claims and proposals
        
        Args:
            item: Either a Claim or ConsensusProposal
            force_mode: Force a specific verification mode
            
        Returns:
            UnifiedResult with merged analysis
        """
        effective_mode = force_mode or self.mode
        
        unified_result = UnifiedResult()
        
        try:
            if isinstance(item, Claim):
                # Primary: individual verification
                unified_result.individual_result = self._verify_individual(item)
                
                # Secondary: consensus verification if available
                if effective_mode in [VerificationMode.CONSENSUS, VerificationMode.HYBRID]:
                    proposal = self.mode_bridge.translate_claim_to_proposal(item)
                    # Note: Would need async handling in full implementation
                    unified_result.consensus_result = {"note": "Consensus verification would be performed here"}
            
            elif isinstance(item, ConsensusProposal):
                # Primary: consensus verification
                if effective_mode in [VerificationMode.CONSENSUS, VerificationMode.HYBRID]:
                    # Note: Would need async handling in full implementation
                    unified_result.consensus_result = {"note": "Consensus verification would be performed here"}
                
                # Secondary: individual verification
                claim = self.mode_bridge.translate_proposal_to_claim(item)
                unified_result.individual_result = self._verify_individual(claim)
            
            # Merge results
            unified_result = self._merge_verification_results(unified_result)
            
            return unified_result
            
        except Exception as e:
            self.logger.error(f"Unified verification failed: {e}")
            unified_result.discrepancies.append(f"Verification error: {str(e)}")
            return unified_result
    
    # ==================== COMPONENT MANAGEMENT ====================
    
    def _run_individual_components(self, claim: Claim) -> List[VerificationResult]:
        """Run individual verification components"""
        if self.enable_parallel:
            return self._run_parallel_individual_components(claim)
        else:
            return self._run_sequential_individual_components(claim)
    
    def _run_parallel_individual_components(self, claim: Claim) -> List[VerificationResult]:
        """Run components in parallel for individual verification"""
        results = []
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {}
            
            for name, component in self.active_components.items():
                if self._component_validates_claim(component, claim):
                    if isinstance(component, UnifiedVerificationComponent):
                        future = executor.submit(component.verify_individual, claim)
                    else:
                        future = executor.submit(component.verify, claim)
                    futures[future] = name
            
            for future in futures:
                component_name = futures[future]
                try:
                    result = future.result(timeout=self.verification_timeout)
                    results.append(result)
                    self.logger.debug(f"Component {component_name} completed: score={result.score:.3f}")
                except Exception as e:
                    self.logger.error(f"Component {component_name} failed: {e}")
        
        return results
    
    def _run_sequential_individual_components(self, claim: Claim) -> List[VerificationResult]:
        """Run components sequentially for individual verification"""
        results = []
        
        for name, component in self.active_components.items():
            if not self._component_validates_claim(component, claim):
                continue
            
            try:
                if isinstance(component, UnifiedVerificationComponent):
                    result = component.verify_individual(claim)
                else:
                    result = component.verify(claim)
                results.append(result)
                self.logger.debug(f"Component {name} completed: score={result.score:.3f}")
            except Exception as e:
                self.logger.error(f"Component {name} failed: {e}")
        
        return results
    
    def _component_validates_claim(self, component: Any, claim: Claim) -> bool:
        """Check if component can validate the claim"""
        if hasattr(component, 'validate_claim'):
            return component.validate_claim(claim)
        return True  # Default: assume all components can validate
    
    # ==================== HELPER METHODS ====================
    
    def _get_applicable_frameworks(self) -> List[VerificationFramework]:
        """Get applicable frameworks from active components"""
        frameworks = set()
        for component in self.active_components.values():
            if hasattr(component, 'get_applicable_frameworks'):
                frameworks.update(component.get_applicable_frameworks())
        return list(frameworks)
    
    def _calculate_individual_final_score(self, meta_result: VerificationResult, 
                                        debate_result: Optional[Dict[str, Any]]) -> float:
        """Calculate final score for individual verification"""
        base_score = meta_result.score
        
        if debate_result and "final_consensus" in debate_result:
            consensus_adjustment = debate_result["final_consensus"] * 0.1
            return min(1.0, base_score + consensus_adjustment)
        
        return base_score
    
    def _generate_individual_explanation(self, claim: Claim, component_results: List[VerificationResult],
                                       meta_result: VerificationResult, debate_result: Optional[Dict[str, Any]]) -> str:
        """Generate explanation for individual verification"""
        explanation_parts = [
            f"Individual verification for {claim.domain.value} claim:",
            f"Final score: {meta_result.score:.3f} using {meta_result.framework.value} framework"
        ]
        
        # Component breakdown
        explanation_parts.append("\nComponent Analysis:")
        component_names = list(self.active_components.keys())
        for i, result in enumerate(component_results):
            comp_name = component_names[i] if i < len(component_names) else f"component_{i}"
            explanation_parts.append(f"- {comp_name}: {result.score:.3f}")
        
        # Debate analysis
        if debate_result:
            explanation_parts.append(f"\nDebate consensus: {debate_result.get('final_consensus', 0.0):.3f}")
        
        return "\n".join(explanation_parts)
    
    def _merge_verification_results(self, unified_result: UnifiedResult) -> UnifiedResult:
        """Merge individual and consensus verification results"""
        # Placeholder implementation - would be more sophisticated in practice
        if unified_result.individual_result and unified_result.consensus_result:
            # Both results available
            individual_score = unified_result.individual_result.get("final_score", 0.0)
            # consensus_score would be extracted from consensus_result
            
            unified_result.merged_score = individual_score  # Simplified
            unified_result.agreement_level = 0.8  # Placeholder
            unified_result.merging_strategy = MergingStrategy.WEIGHTED_AVERAGE
            unified_result.recommendation = "Both individual and consensus analysis completed"
        
        elif unified_result.individual_result:
            # Only individual result
            unified_result.merged_score = unified_result.individual_result.get("final_score", 0.0)
            unified_result.recommendation = "Individual analysis only - consensus verification recommended"
        
        elif unified_result.consensus_result:
            # Only consensus result
            unified_result.merged_score = 0.5  # Placeholder
            unified_result.recommendation = "Consensus analysis only - individual verification recommended"
        
        return unified_result
    
    def _serialize_verification_result(self, result: VerificationResult) -> Dict[str, Any]:
        """Serialize verification result for JSON output"""
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
            "verification_mode": self.mode.value,
            "timestamp": datetime.now().isoformat()
        }
    
    def _record_individual_verification(self, claim: Claim, response: Dict[str, Any]):
        """Record individual verification for tracking"""
        record = {
            "type": "individual",
            "claim_content": claim.content,
            "domain": claim.domain.value,
            "final_score": response["final_score"],
            "components_used": response["metadata"]["components_used"],
            "timestamp": response["metadata"]["verification_timestamp"],
            "mode": self.mode.value
        }
        
        self.verification_history.append(record)
        
        # Keep history manageable
        if len(self.verification_history) > 1000:
            self.verification_history = self.verification_history[-1000:]
    
    def _record_consensus_attempt(self, proposal: ConsensusProposal, result: Dict[str, Any]):
        """Record consensus attempt for tracking"""
        record = {
            "type": "consensus",
            "proposal_id": proposal.proposal_id,
            "proposal_type": proposal.proposal_type.value,
            "success": result.get("success", False),
            "final_decision": result.get("final_decision", "unknown"),
            "consensus_time": result.get("consensus_time", 0.0),
            "timestamp": result.get("timestamp", datetime.now().isoformat()),
            "mode": self.mode.value
        }
        
        self.consensus_history.append(record)
        
        # Keep history manageable
        if len(self.consensus_history) > 1000:
            self.consensus_history = self.consensus_history[-1000:]
    
    # ==================== LEARNING AND ADAPTATION ====================
    
    def learn_from_feedback(self, item: Union[Claim, ConsensusProposal], 
                           verification_result: Dict[str, Any], 
                           human_feedback: Dict[str, Any]):
        """Learn and adapt from human feedback"""
        try:
            # Update meta-verifier
            if isinstance(item, Claim) and "framework_preference" in human_feedback:
                framework = VerificationFramework(human_feedback["framework_preference"])
                framework_feedback = {framework: human_feedback.get("weight_adjustment", 1.1)}
                self.meta_verifier.update_framework_weights(framework_feedback)
            
            # Update components
            for component in self.active_components.values():
                if isinstance(component, UnifiedVerificationComponent):
                    component.learn_from_consensus_feedback(
                        getattr(item, 'proposal_id', str(hash(str(item)))),
                        human_feedback
                    )
            
            # Update consensus system if available
            if self.consensus_system and "philosophical_adjustments" in human_feedback:
                self.consensus_system.update_philosophical_commitments(
                    human_feedback["philosophical_adjustments"]
                )
            
            self.logger.info("Successfully processed feedback for learning")
            
        except Exception as e:
            self.logger.error(f"Failed to process feedback: {e}")
    
    # ==================== STATISTICS AND MONITORING ====================
    
    def get_verification_statistics(self) -> Dict[str, Any]:
        """Get comprehensive verification statistics"""
        total_verifications = len(self.verification_history)
        total_consensus = len(self.consensus_history)
        
        stats = {
            "system_info": {
                "current_mode": self.mode.value,
                "unified_components_enabled": self.enable_unified_components,
                "consensus_available": self.consensus_system is not None,
                "node_id": self.node_context.node_id if self.node_context else None
            },
            "verification_counts": {
                "total_individual": total_verifications,
                "total_consensus": total_consensus,
                "mode_switches": len(self.mode_switch_history)
            },
            "component_statistics": {
                name: component.get_statistics() 
                for name, component in self.active_components.items()
                if hasattr(component, 'get_statistics')
            }
        }
        
        # Add individual verification stats
        if self.verification_history:
            recent_individual = self.verification_history[-50:]
            stats["individual_performance"] = {
                "recent_average_score": sum(v["final_score"] for v in recent_individual) / len(recent_individual),
                "domain_distribution": self._get_domain_distribution(recent_individual),
                "component_usage": self._get_component_usage_stats(recent_individual)
            }
        
        # Add consensus stats
        if self.consensus_history:
            recent_consensus = self.consensus_history[-50:]
            successful_consensus = [c for c in recent_consensus if c["success"]]
            stats["consensus_performance"] = {
                "success_rate": len(successful_consensus) / len(recent_consensus),
                "average_consensus_time": sum(c["consensus_time"] for c in successful_consensus) / len(successful_consensus) if successful_consensus else 0.0,
                "proposal_type_distribution": self._get_proposal_type_distribution(recent_consensus)
            }
        
        # Add consensus system stats if available
        if self.consensus_system:
            stats["consensus_system"] = self.consensus_system.get_consensus_statistics()
        
        # Add meta-verifier stats
        stats["meta_verifier"] = self.meta_verifier.get_statistics()
        
        # Add debate system stats
        stats["debate_system"] = self.debate_system.get_statistics()
        
        return stats
    
    def _get_domain_distribution(self, records: List[Dict[str, Any]]) -> Dict[str, int]:
        """Get distribution of verifications by domain"""
        distribution = {}
        for record in records:
            domain = record.get("domain", "unknown")
            distribution[domain] = distribution.get(domain, 0) + 1
        return distribution
    
    def _get_component_usage_stats(self, records: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Get component usage statistics"""
        total_components = sum(record.get("components_used", 0) for record in records)
        avg_components = total_components / len(records) if records else 0
        
        return {
            "average_components_per_verification": avg_components,
            "total_component_invocations": total_components
        }
    
    def _get_proposal_type_distribution(self, records: List[Dict[str, Any]]) -> Dict[str, int]:
        """Get distribution of consensus proposals by type"""
        distribution = {}
        for record in records:
            proposal_type = record.get("proposal_type", "unknown")
            distribution[proposal_type] = distribution.get(proposal_type, 0) + 1
        return distribution
    
    # ==================== CONFIGURATION MANAGEMENT ====================
    
    def export_configuration(self) -> Dict[str, Any]:
        """Export current system configuration"""
        config = {
            "system_settings": {
                "mode": self.mode.value,
                "enable_unified_components": self.enable_unified_components,
                "enable_parallel": self.enable_parallel,
                "enable_debate": self.enable_debate,
                "max_workers": self.max_workers,
                "verification_timeout": self.verification_timeout
            },
            "meta_verifier_config": self.meta_verifier.get_statistics(),
            "component_configurations": {}
        }
        
        # Add component configurations
        for name, component in self.active_components.items():
            if hasattr(component, 'get_statistics'):
                config["component_configurations"][name] = component.get_statistics()
        
        # Add consensus configuration if available
        if self.consensus_config:
            config["consensus_config"] = {
                "node_id": self.consensus_config.node_id,
                "peer_nodes": self.consensus_config.peer_nodes,
                "required_quorum": self.consensus_config.required_quorum,
                "consensus_timeout": self.consensus_config.consensus_timeout,
                "enable_debate": self.consensus_config.enable_debate,
                "max_debate_rounds": self.consensus_config.max_debate_rounds
            }
        
        return config
    
    def import_configuration(self, config: Dict[str, Any]):
        """Import system configuration"""
        try:
            # Update system settings
            if "system_settings" in config:
                settings = config["system_settings"]
                self.enable_parallel = settings.get("enable_parallel", self.enable_parallel)
                self.enable_debate = settings.get("enable_debate", self.enable_debate)
                self.max_workers = settings.get("max_workers", self.max_workers)
                self.verification_timeout = settings.get("verification_timeout", self.verification_timeout)
            
            # Update consensus configuration
            if "consensus_config" in config:
                consensus_settings = config["consensus_config"]
                self.consensus_config.required_quorum = consensus_settings.get("required_quorum", self.consensus_config.required_quorum)
                self.consensus_config.consensus_timeout = consensus_settings.get("consensus_timeout", self.consensus_config.consensus_timeout)
                self.consensus_config.enable_debate = consensus_settings.get("enable_debate", self.consensus_config.enable_debate)
            
            self.logger.info("Configuration imported successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to import configuration: {e}")
            raise