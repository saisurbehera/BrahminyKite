"""
Philosophical Paxos: Modified Paxos Protocol with Multi-Framework Verification

This module implements a modified Paxos consensus protocol that integrates
philosophical verification frameworks for distributed decision-making.
"""

import asyncio
import json
import time
from typing import Dict, List, Set, Optional, Any, Callable
from datetime import datetime, timedelta
import logging
from dataclasses import asdict

from ..consensus_types import (
    ConsensusProposal, NodeContext, ConsensusCriteria, ConsensusVerificationResult,
    PaxosMessage, PaxosPrepareRequest, PaxosPromise, PaxosAcceptRequest, PaxosAcceptResponse,
    ApprovalStatus, NodeRole, NetworkTopology, ConsensusConfig
)
from ..frameworks import VerificationFramework


class PhilosophicalPaxos:
    """
    Modified Paxos protocol integrating philosophical verification frameworks
    
    Key enhancements over standard Paxos:
    - Multi-criteria promises with philosophical validation
    - Framework-specific acceptance thresholds
    - Integrated debate resolution for conflicts
    - Dynamic quorum adjustment based on philosophical alignment
    """
    
    def __init__(self, node_context: NodeContext, config: ConsensusConfig,
                 verifier_components: Dict[str, Any]):
        self.node_context = node_context
        self.config = config
        self.verifier_components = verifier_components
        self.logger = logging.getLogger(f"{__name__}.{node_context.node_id}")
        
        # Paxos state
        self.current_proposal_number = 0
        self.highest_promised_number = 0
        self.accepted_proposal_number: Optional[int] = None
        self.accepted_value: Optional[ConsensusProposal] = None
        
        # Philosophical state
        self.philosophical_commitments: Dict[VerificationFramework, float] = {}
        self.framework_trust_scores: Dict[str, Dict[VerificationFramework, float]] = {}
        
        # Network state
        self.network_topology = NetworkTopology()
        self.active_consensus_sessions: Dict[str, Dict[str, Any]] = {}
        self.message_handlers: Dict[str, Callable] = {}
        
        # Performance tracking
        self.consensus_history: List[Dict[str, Any]] = []
        self.performance_metrics = {
            "successful_consensus": 0,
            "failed_consensus": 0,
            "average_consensus_time": 0.0,
            "philosophical_conflicts_resolved": 0
        }
        
        self._initialize_message_handlers()
        self._initialize_philosophical_state()
    
    def _initialize_message_handlers(self):
        """Initialize message handlers for different Paxos phases"""
        self.message_handlers = {
            "prepare_request": self.handle_prepare_request,
            "promise": self.handle_promise,
            "accept_request": self.handle_accept_request,
            "accept_response": self.handle_accept_response,
            "heartbeat": self.handle_heartbeat,
            "philosophical_debate": self.handle_philosophical_debate
        }
    
    def _initialize_philosophical_state(self):
        """Initialize philosophical commitments and trust scores"""
        # Default equal weighting for all frameworks
        for framework in VerificationFramework:
            self.philosophical_commitments[framework] = 1.0
        
        # Initialize trust scores for peer nodes
        for peer_id in self.node_context.peer_nodes:
            self.framework_trust_scores[peer_id] = {
                framework: 0.5 for framework in VerificationFramework
            }
    
    # ==================== PROPOSER ROLE ====================
    
    async def propose_consensus(self, proposal: ConsensusProposal) -> Dict[str, Any]:
        """
        Initiate consensus process as proposer (Phase 1 + 2)
        
        Args:
            proposal: The proposal to achieve consensus on
            
        Returns:
            Consensus result with philosophical validation details
        """
        session_id = f"consensus_{proposal.proposal_id}_{self.node_context.node_id}"
        start_time = datetime.now()
        
        self.logger.info(f"Starting consensus for proposal {proposal.proposal_id}")
        
        try:
            # Phase 1: Prepare with philosophical pre-validation
            prepare_result = await self._phase1_prepare(proposal)
            
            if not prepare_result["success"]:
                return self._create_failed_consensus_result(
                    proposal, "Phase 1 failed", prepare_result.get("reason", "Unknown")
                )
            
            # Phase 2: Accept with multi-criteria validation
            accept_result = await self._phase2_accept(proposal, prepare_result["promises"])
            
            # Record consensus attempt
            consensus_result = self._create_consensus_result(
                proposal, accept_result, start_time, datetime.now()
            )
            
            self._record_consensus_attempt(session_id, consensus_result)
            
            return consensus_result
            
        except Exception as e:
            self.logger.error(f"Consensus failed for proposal {proposal.proposal_id}: {e}")
            return self._create_failed_consensus_result(proposal, "Exception", str(e))
    
    async def _phase1_prepare(self, proposal: ConsensusProposal) -> Dict[str, Any]:
        """
        Phase 1: Prepare with philosophical criteria validation
        """
        self.current_proposal_number += 1
        proposal_number = self.current_proposal_number
        
        # Pre-validate proposal with local philosophical verifiers
        local_validations = await self._pre_validate_proposal(proposal)
        
        # Create prepare request with philosophical criteria
        prepare_request = PaxosPrepareRequest(
            proposal_number=proposal_number,
            sender_id=self.node_context.node_id,
            philosophical_criteria=local_validations,
            proposal=proposal
        )
        
        # Send prepare requests to all acceptors
        promises = await self._broadcast_prepare_request(prepare_request)
        
        # Analyze promises for philosophical alignment
        promise_analysis = self._analyze_philosophical_promises(promises, proposal)
        
        return {
            "success": promise_analysis["quorum_achieved"],
            "promises": promises,
            "philosophical_analysis": promise_analysis,
            "proposal_number": proposal_number,
            "reason": promise_analysis.get("failure_reason")
        }
    
    async def _phase2_accept(self, proposal: ConsensusProposal, promises: List[PaxosPromise]) -> Dict[str, Any]:
        """
        Phase 2: Accept with harmonized multi-criteria validation
        """
        proposal_number = self.current_proposal_number
        
        # Perform comprehensive philosophical validation
        philosophical_validations = await self._perform_philosophical_validations(proposal)
        
        # Create accept request
        accept_request = PaxosAcceptRequest(
            proposal_number=proposal_number,
            sender_id=self.node_context.node_id,
            proposal=proposal,
            philosophical_validation=philosophical_validations
        )
        
        # Send accept requests to all acceptors
        accept_responses = await self._broadcast_accept_request(accept_request)
        
        # Analyze accept responses
        acceptance_analysis = self._analyze_philosophical_acceptance(accept_responses, proposal)
        
        return {
            "success": acceptance_analysis["consensus_achieved"],
            "accept_responses": accept_responses,
            "philosophical_analysis": acceptance_analysis,
            "final_decision": acceptance_analysis.get("final_decision"),
            "reason": acceptance_analysis.get("failure_reason")
        }
    
    # ==================== ACCEPTOR ROLE ====================
    
    async def handle_prepare_request(self, request: PaxosPrepareRequest) -> PaxosPromise:
        """
        Handle Phase 1a: Prepare request with philosophical evaluation
        """
        self.logger.debug(f"Handling prepare request {request.proposal_number} from {request.sender_id}")
        
        # Standard Paxos promise logic
        if request.proposal_number <= self.highest_promised_number:
            return PaxosPromise(
                proposal_number=request.proposal_number,
                sender_id=self.node_context.node_id,
                promised=False
            )
        
        # Update promise state
        self.highest_promised_number = request.proposal_number
        
        # Philosophical validation of criteria
        verifier_approvals = {}
        for verifier_name, criteria in request.philosophical_criteria.items():
            if verifier_name in self.verifier_components:
                verifier = self.verifier_components[verifier_name]
                
                # Check if verifier can validate these criteria
                can_validate = verifier.can_validate_criteria(criteria)
                if can_validate:
                    validation_conditions = verifier.get_validation_conditions(criteria)
                    verifier_approvals[verifier_name] = ConsensusCriteria(
                        verifier_type=verifier_name,
                        acceptance_threshold=criteria.acceptance_threshold,
                        conditions=validation_conditions,
                        weight=criteria.weight
                    )
        
        # Create philosophical promise
        promise = PaxosPromise(
            proposal_number=request.proposal_number,
            sender_id=self.node_context.node_id,
            promised=True,
            previous_proposal_number=self.accepted_proposal_number,
            previous_value=self.accepted_value,
            verifier_approvals=verifier_approvals,
            philosophical_stance=self._get_philosophical_stance(),
            conditions=self._generate_promise_conditions(request)
        )
        
        return promise
    
    async def handle_accept_request(self, request: PaxosAcceptRequest) -> PaxosAcceptResponse:
        """
        Handle Phase 2a: Accept request with multi-criteria validation
        """
        self.logger.debug(f"Handling accept request {request.proposal_number} from {request.sender_id}")
        
        # Standard Paxos acceptance logic
        if request.proposal_number < self.highest_promised_number:
            return PaxosAcceptResponse(
                proposal_number=request.proposal_number,
                sender_id=self.node_context.node_id,
                accepted=False
            )
        
        # Perform philosophical validation
        verifier_validations = {}
        for verifier_name, component in self.verifier_components.items():
            try:
                validation_result = await self._validate_with_component(
                    component, request.proposal, verifier_name
                )
                verifier_validations[verifier_name] = validation_result
            except Exception as e:
                self.logger.error(f"Validation failed for {verifier_name}: {e}")
        
        # Meta-validation: determine if proposal should be accepted
        acceptance_decision = self._make_philosophical_acceptance_decision(
            verifier_validations, request.proposal
        )
        
        # Update acceptor state if accepting
        if acceptance_decision:
            self.accepted_proposal_number = request.proposal_number
            self.accepted_value = request.proposal
        
        return PaxosAcceptResponse(
            proposal_number=request.proposal_number,
            sender_id=self.node_context.node_id,
            accepted=acceptance_decision,
            proposal=request.proposal if acceptance_decision else None,
            verifier_validations=verifier_validations,
            node_philosophical_state=self._get_philosophical_state()
        )
    
    # ==================== PHILOSOPHICAL VALIDATION ====================
    
    async def _pre_validate_proposal(self, proposal: ConsensusProposal) -> Dict[str, ConsensusCriteria]:
        """Pre-validate proposal with local philosophical verifiers"""
        validations = {}
        
        for verifier_name, component in self.verifier_components.items():
            if component.validate_proposal(proposal):
                criteria = component.prepare_consensus_criteria(proposal)
                validations[verifier_name] = criteria
        
        return validations
    
    async def _perform_philosophical_validations(self, proposal: ConsensusProposal) -> Dict[str, ConsensusVerificationResult]:
        """Perform comprehensive philosophical validation"""
        validations = {}
        
        for verifier_name, component in self.verifier_components.items():
            if component.validate_proposal(proposal):
                try:
                    result = component.verify_consensus(proposal, self.node_context)
                    validations[verifier_name] = result
                except Exception as e:
                    self.logger.error(f"Philosophical validation failed for {verifier_name}: {e}")
        
        return validations
    
    async def _validate_with_component(self, component: Any, proposal: ConsensusProposal, 
                                     verifier_name: str) -> ConsensusVerificationResult:
        """Validate proposal with a specific component"""
        if hasattr(component, 'verify_consensus'):
            return component.verify_consensus(proposal, self.node_context)
        else:
            # Fallback for components without consensus support
            return ConsensusVerificationResult(
                verifier_type=verifier_name,
                proposal_id=proposal.proposal_id,
                approval_status=ApprovalStatus.ABSTAIN,
                confidence=0.5,
                reasoning="Component does not support consensus verification"
            )
    
    def _make_philosophical_acceptance_decision(self, 
                                              validations: Dict[str, ConsensusVerificationResult],
                                              proposal: ConsensusProposal) -> bool:
        """Make final acceptance decision based on philosophical validations"""
        
        # Check if required verifiers all approve
        required_verifiers = proposal.required_verifiers
        if required_verifiers:
            for verifier_name in required_verifiers:
                if verifier_name not in validations:
                    self.logger.warning(f"Required verifier {verifier_name} not available")
                    return False
                
                result = validations[verifier_name]
                if result.approval_status == ApprovalStatus.REJECT:
                    self.logger.info(f"Required verifier {verifier_name} rejected proposal")
                    return False
        
        # Calculate weighted approval score
        total_weight = 0.0
        approval_score = 0.0
        
        for verifier_name, result in validations.items():
            weight = self._get_verifier_weight(verifier_name, proposal)
            total_weight += weight
            
            if result.approval_status == ApprovalStatus.APPROVE:
                approval_score += weight * result.confidence
            elif result.approval_status == ApprovalStatus.CONDITIONAL:
                approval_score += weight * result.confidence * 0.7  # Reduced weight for conditional
        
        # Decision threshold
        if total_weight > 0:
            final_score = approval_score / total_weight
            threshold = self._get_acceptance_threshold(proposal)
            return final_score >= threshold
        
        return False
    
    # ==================== NETWORK COMMUNICATION ====================
    
    async def _broadcast_prepare_request(self, request: PaxosPrepareRequest) -> List[PaxosPromise]:
        """Broadcast prepare request to all acceptor nodes"""
        promises = []
        
        # Include self if we're also an acceptor
        if self.node_context.node_role in [NodeRole.ACCEPTOR, NodeRole.PROPOSER]:
            self_promise = await self.handle_prepare_request(request)
            promises.append(self_promise)
        
        # Send to peer nodes (simulated for now)
        for peer_id in self.node_context.peer_nodes:
            if self.node_context.get_trust(peer_id) > 0.3:  # Only send to trusted peers
                # In real implementation, this would be actual network communication
                simulated_promise = self._simulate_peer_promise(peer_id, request)
                promises.append(simulated_promise)
        
        return promises
    
    async def _broadcast_accept_request(self, request: PaxosAcceptRequest) -> List[PaxosAcceptResponse]:
        """Broadcast accept request to all acceptor nodes"""
        responses = []
        
        # Include self if we're also an acceptor
        if self.node_context.node_role in [NodeRole.ACCEPTOR, NodeRole.PROPOSER]:
            self_response = await self.handle_accept_request(request)
            responses.append(self_response)
        
        # Send to peer nodes (simulated for now)
        for peer_id in self.node_context.peer_nodes:
            if self.node_context.get_trust(peer_id) > 0.3:
                # In real implementation, this would be actual network communication
                simulated_response = self._simulate_peer_accept_response(peer_id, request)
                responses.append(simulated_response)
        
        return responses
    
    def _simulate_peer_promise(self, peer_id: str, request: PaxosPrepareRequest) -> PaxosPromise:
        """Simulate peer promise response (for testing/demo)"""
        import random
        
        # Simulate philosophical alignment with peer
        peer_alignment = self.framework_trust_scores.get(peer_id, {})
        alignment_score = sum(peer_alignment.values()) / len(peer_alignment) if peer_alignment else 0.5
        
        promised = random.random() < (0.7 + alignment_score * 0.2)  # Higher chance if aligned
        
        return PaxosPromise(
            proposal_number=request.proposal_number,
            sender_id=peer_id,
            promised=promised,
            philosophical_stance={f.value: random.uniform(0.3, 0.9) for f in VerificationFramework},
            conditions=[f"peer_{peer_id}_condition"] if promised else []
        )
    
    def _simulate_peer_accept_response(self, peer_id: str, request: PaxosAcceptRequest) -> PaxosAcceptResponse:
        """Simulate peer accept response (for testing/demo)"""
        import random
        
        # Simulate acceptance based on trust and alignment
        trust_score = self.node_context.get_trust(peer_id)
        accepted = random.random() < (0.6 + trust_score * 0.3)
        
        return PaxosAcceptResponse(
            proposal_number=request.proposal_number,
            sender_id=peer_id,
            accepted=accepted,
            proposal=request.proposal if accepted else None,
            node_philosophical_state={"simulated": True, "peer_id": peer_id}
        )
    
    # ==================== ANALYSIS AND DECISION MAKING ====================
    
    def _analyze_philosophical_promises(self, promises: List[PaxosPromise], 
                                      proposal: ConsensusProposal) -> Dict[str, Any]:
        """Analyze promises for philosophical alignment and quorum"""
        analysis = {
            "total_promises": len(promises),
            "positive_promises": 0,
            "quorum_achieved": False,
            "philosophical_alignment": {},
            "framework_consensus": {},
            "conditions_summary": [],
            "failure_reason": None
        }
        
        positive_promises = [p for p in promises if p.promised]
        analysis["positive_promises"] = len(positive_promises)
        
        # Check quorum
        required_quorum = max(1, int(len(self.node_context.peer_nodes) * self.config.required_quorum))
        analysis["quorum_achieved"] = len(positive_promises) >= required_quorum
        
        if not analysis["quorum_achieved"]:
            analysis["failure_reason"] = f"Insufficient promises: {len(positive_promises)}/{required_quorum}"
            return analysis
        
        # Analyze philosophical alignment
        framework_scores = {f.value: [] for f in VerificationFramework}
        all_conditions = []
        
        for promise in positive_promises:
            # Collect philosophical stance data
            for framework, score in promise.philosophical_stance.items():
                if framework in framework_scores:
                    framework_scores[framework].append(score)
            
            # Collect conditions
            all_conditions.extend(promise.conditions)
        
        # Calculate framework consensus
        for framework, scores in framework_scores.items():
            if scores:
                analysis["framework_consensus"][framework] = {
                    "average_score": sum(scores) / len(scores),
                    "std_deviation": self._calculate_std(scores),
                    "consensus_level": 1.0 - min(self._calculate_std(scores), 1.0)
                }
        
        # Summarize conditions
        from collections import Counter
        condition_counts = Counter(all_conditions)
        analysis["conditions_summary"] = [
            {"condition": cond, "count": count} 
            for cond, count in condition_counts.most_common()
        ]
        
        return analysis
    
    def _analyze_philosophical_acceptance(self, responses: List[PaxosAcceptResponse],
                                        proposal: ConsensusProposal) -> Dict[str, Any]:
        """Analyze accept responses for consensus achievement"""
        analysis = {
            "total_responses": len(responses),
            "acceptances": 0,
            "rejections": 0,
            "consensus_achieved": False,
            "verifier_consensus": {},
            "philosophical_summary": {},
            "final_decision": None,
            "failure_reason": None
        }
        
        accepted_responses = [r for r in responses if r.accepted]
        analysis["acceptances"] = len(accepted_responses)
        analysis["rejections"] = len(responses) - len(accepted_responses)
        
        # Check consensus achievement
        required_acceptances = max(1, int(len(self.node_context.peer_nodes) * self.config.required_quorum))
        analysis["consensus_achieved"] = len(accepted_responses) >= required_acceptances
        
        if not analysis["consensus_achieved"]:
            analysis["failure_reason"] = f"Insufficient acceptances: {len(accepted_responses)}/{required_acceptances}"
            return analysis
        
        # Analyze verifier consensus
        verifier_results = {}
        for response in accepted_responses:
            for verifier_name, result in response.verifier_validations.items():
                if verifier_name not in verifier_results:
                    verifier_results[verifier_name] = []
                verifier_results[verifier_name].append(result)
        
        for verifier_name, results in verifier_results.items():
            approval_count = sum(1 for r in results if r.approval_status == ApprovalStatus.APPROVE)
            avg_confidence = sum(r.confidence for r in results) / len(results)
            
            analysis["verifier_consensus"][verifier_name] = {
                "approval_rate": approval_count / len(results),
                "average_confidence": avg_confidence,
                "total_validations": len(results)
            }
        
        # Determine final decision
        analysis["final_decision"] = self._determine_final_consensus_decision(analysis, proposal)
        
        return analysis
    
    def _determine_final_consensus_decision(self, analysis: Dict[str, Any], 
                                          proposal: ConsensusProposal) -> str:
        """Determine final consensus decision"""
        if not analysis["consensus_achieved"]:
            return "REJECTED"
        
        # Check verifier consensus quality
        verifier_consensus = analysis["verifier_consensus"]
        
        # All required verifiers must have good approval rates
        for verifier_name in proposal.required_verifiers:
            if verifier_name in verifier_consensus:
                consensus_data = verifier_consensus[verifier_name]
                if (consensus_data["approval_rate"] < 0.6 or 
                    consensus_data["average_confidence"] < 0.5):
                    return "CONDITIONAL_APPROVAL"
        
        # Check overall consensus quality
        if verifier_consensus:
            avg_approval_rate = sum(v["approval_rate"] for v in verifier_consensus.values()) / len(verifier_consensus)
            avg_confidence = sum(v["average_confidence"] for v in verifier_consensus.values()) / len(verifier_consensus)
            
            if avg_approval_rate >= 0.8 and avg_confidence >= 0.7:
                return "APPROVED"
            elif avg_approval_rate >= 0.6 and avg_confidence >= 0.5:
                return "CONDITIONAL_APPROVAL"
        
        return "REJECTED"
    
    # ==================== HELPER METHODS ====================
    
    def _get_philosophical_stance(self) -> Dict[str, float]:
        """Get current philosophical stance of this node"""
        return {f.value: score for f, score in self.philosophical_commitments.items()}
    
    def _get_philosophical_state(self) -> Dict[str, Any]:
        """Get comprehensive philosophical state"""
        return {
            "philosophical_commitments": self._get_philosophical_stance(),
            "node_id": self.node_context.node_id,
            "domain_expertise": self.node_context.domain_expertise,
            "timestamp": datetime.now().isoformat()
        }
    
    def _generate_promise_conditions(self, request: PaxosPrepareRequest) -> List[str]:
        """Generate conditions for promise based on request"""
        conditions = []
        
        if request.proposal:
            # Add domain-specific conditions
            if hasattr(request.proposal, 'domain') and request.proposal.domain:
                conditions.append(f"domain_{request.proposal.domain.value}_validation_required")
            
            # Add priority-based conditions
            if request.proposal.priority_level <= 2:  # High priority
                conditions.append("expedited_review_required")
            
            # Add timeout-based conditions
            if request.proposal.timeout < 60:  # Short timeout
                conditions.append("rapid_consensus_mode")
        
        return conditions
    
    def _get_verifier_weight(self, verifier_name: str, proposal: ConsensusProposal) -> float:
        """Get weight for a verifier based on proposal and context"""
        base_weight = 1.0
        
        # Adjust based on proposal domain
        if hasattr(proposal, 'domain') and proposal.domain:
            domain_weights = {
                "empirical": {"empirical": 1.2, "contextual": 0.8},
                "aesthetic": {"contextual": 1.2, "empirical": 0.7},
                "ethical": {"consistency": 1.1, "power_dynamics": 1.1}
            }
            
            domain_name = proposal.domain.value
            if domain_name in domain_weights and verifier_name in domain_weights[domain_name]:
                base_weight *= domain_weights[domain_name][verifier_name]
        
        return base_weight
    
    def _get_acceptance_threshold(self, proposal: ConsensusProposal) -> float:
        """Get acceptance threshold based on proposal characteristics"""
        base_threshold = 0.6
        
        # Higher threshold for critical proposals
        if proposal.priority_level <= 2:
            base_threshold = 0.8
        
        # Higher threshold for ethical guidelines
        if proposal.proposal_type.value == "ethical_guideline":
            base_threshold = 0.9
        
        return base_threshold
    
    def _calculate_std(self, values: List[float]) -> float:
        """Calculate standard deviation"""
        if len(values) <= 1:
            return 0.0
        
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / len(values)
        return variance ** 0.5
    
    # ==================== RESULT CREATION ====================
    
    def _create_consensus_result(self, proposal: ConsensusProposal, accept_result: Dict[str, Any],
                               start_time: datetime, end_time: datetime) -> Dict[str, Any]:
        """Create comprehensive consensus result"""
        duration = (end_time - start_time).total_seconds()
        
        result = {
            "proposal_id": proposal.proposal_id,
            "success": accept_result["success"],
            "final_decision": accept_result.get("final_decision", "UNKNOWN"),
            "consensus_time": duration,
            "timestamp": end_time.isoformat(),
            "proposer_node": self.node_context.node_id,
            "participating_nodes": len(self.node_context.peer_nodes) + 1,
            "philosophical_analysis": accept_result.get("philosophical_analysis", {}),
            "verifier_results": self._extract_verifier_summary(accept_result),
            "network_health": self._assess_network_health(),
            "performance_metrics": self._calculate_performance_metrics(duration, accept_result["success"])
        }
        
        # Update performance tracking
        if accept_result["success"]:
            self.performance_metrics["successful_consensus"] += 1
        else:
            self.performance_metrics["failed_consensus"] += 1
        
        self._update_average_consensus_time(duration)
        
        return result
    
    def _create_failed_consensus_result(self, proposal: ConsensusProposal, 
                                      failure_stage: str, reason: str) -> Dict[str, Any]:
        """Create result for failed consensus"""
        return {
            "proposal_id": proposal.proposal_id,
            "success": False,
            "failure_stage": failure_stage,
            "failure_reason": reason,
            "timestamp": datetime.now().isoformat(),
            "proposer_node": self.node_context.node_id
        }
    
    def _extract_verifier_summary(self, accept_result: Dict[str, Any]) -> Dict[str, Any]:
        """Extract summary of verifier results from accept result"""
        summary = {}
        
        philosophical_analysis = accept_result.get("philosophical_analysis", {})
        verifier_consensus = philosophical_analysis.get("verifier_consensus", {})
        
        for verifier_name, consensus_data in verifier_consensus.items():
            summary[verifier_name] = {
                "approval_rate": consensus_data.get("approval_rate", 0.0),
                "confidence": consensus_data.get("average_confidence", 0.0),
                "validations_count": consensus_data.get("total_validations", 0)
            }
        
        return summary
    
    def _assess_network_health(self) -> Dict[str, Any]:
        """Assess current network health"""
        active_peers = len([peer for peer in self.node_context.peer_nodes 
                           if self.node_context.get_trust(peer) > 0.3])
        
        return {
            "total_peers": len(self.node_context.peer_nodes),
            "active_peers": active_peers,
            "network_partition": self.node_context.network_partition,
            "average_trust": sum(self.node_context.trust_scores.values()) / len(self.node_context.trust_scores) if self.node_context.trust_scores else 0.5
        }
    
    def _calculate_performance_metrics(self, duration: float, success: bool) -> Dict[str, Any]:
        """Calculate performance metrics for this consensus round"""
        return {
            "consensus_duration": duration,
            "success": success,
            "efficiency_score": max(0.0, 1.0 - duration / 60.0),  # Efficiency decreases with time
            "network_utilization": len(self.node_context.peer_nodes) / 10.0  # Normalize by expected network size
        }
    
    def _update_average_consensus_time(self, duration: float):
        """Update rolling average consensus time"""
        total_consensus = self.performance_metrics["successful_consensus"] + self.performance_metrics["failed_consensus"]
        if total_consensus > 0:
            current_avg = self.performance_metrics["average_consensus_time"]
            self.performance_metrics["average_consensus_time"] = (
                (current_avg * (total_consensus - 1) + duration) / total_consensus
            )
    
    def _record_consensus_attempt(self, session_id: str, result: Dict[str, Any]):
        """Record consensus attempt for learning and analysis"""
        record = {
            "session_id": session_id,
            "timestamp": datetime.now().isoformat(),
            "result": result,
            "node_state": self._get_philosophical_state()
        }
        
        self.consensus_history.append(record)
        
        # Keep history manageable
        if len(self.consensus_history) > 1000:
            self.consensus_history = self.consensus_history[-1000:]
    
    # ==================== PUBLIC INTERFACE ====================
    
    def get_consensus_statistics(self) -> Dict[str, Any]:
        """Get comprehensive consensus statistics"""
        total_attempts = self.performance_metrics["successful_consensus"] + self.performance_metrics["failed_consensus"]
        
        stats = {
            "total_consensus_attempts": total_attempts,
            "success_rate": self.performance_metrics["successful_consensus"] / max(total_attempts, 1),
            "average_consensus_time": self.performance_metrics["average_consensus_time"],
            "philosophical_conflicts_resolved": self.performance_metrics["philosophical_conflicts_resolved"],
            "current_philosophical_commitments": self._get_philosophical_stance(),
            "network_health": self._assess_network_health(),
            "recent_consensus_history": len(self.consensus_history)
        }
        
        return stats
    
    def update_philosophical_commitments(self, framework_updates: Dict[VerificationFramework, float]):
        """Update philosophical commitments based on learning"""
        for framework, adjustment in framework_updates.items():
            if framework in self.philosophical_commitments:
                old_value = self.philosophical_commitments[framework]
                new_value = max(0.1, min(2.0, old_value * adjustment))  # Keep in reasonable bounds
                self.philosophical_commitments[framework] = new_value
                
                self.logger.info(f"Updated {framework.value} commitment: {old_value:.3f} -> {new_value:.3f}")
    
    def update_peer_trust(self, peer_id: str, framework: VerificationFramework, trust_delta: float):
        """Update trust score for a peer on a specific framework"""
        if peer_id not in self.framework_trust_scores:
            self.framework_trust_scores[peer_id] = {f: 0.5 for f in VerificationFramework}
        
        current_trust = self.framework_trust_scores[peer_id][framework]
        new_trust = max(0.0, min(1.0, current_trust + trust_delta))
        self.framework_trust_scores[peer_id][framework] = new_trust
        
        # Update overall node trust in node_context
        avg_trust = sum(self.framework_trust_scores[peer_id].values()) / len(VerificationFramework)
        self.node_context.update_trust(peer_id, avg_trust)
    
    async def handle_message(self, message_type: str, message_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Handle incoming consensus messages"""
        if message_type in self.message_handlers:
            handler = self.message_handlers[message_type]
            return await handler(message_data)
        else:
            self.logger.warning(f"Unknown message type: {message_type}")
            return None
    
    # Additional message handlers (placeholders for future implementation)
    async def handle_promise(self, message_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Handle promise messages (for proposer role)"""
        pass
    
    async def handle_accept_response(self, message_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Handle accept response messages (for proposer role)"""
        pass
    
    async def handle_heartbeat(self, message_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Handle heartbeat messages for network health"""
        pass
    
    async def handle_philosophical_debate(self, message_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Handle philosophical debate messages"""
        pass