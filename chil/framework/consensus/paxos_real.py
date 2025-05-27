"""
Real Philosophical Paxos Implementation

Integrates the real networking layer with philosophical verification for
distributed consensus.
"""

import asyncio
import uuid
import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from dataclasses import asdict

from ...networking import GRPCTransport, PeerDiscovery, ServiceRegistry, NodeInfo, NodeStatus, PaxosProtocol
from ..consensus_types import (
    ConsensusProposal, NodeContext, ConsensusVerificationResult,
    ApprovalStatus, NodeRole
)


class RealPhilosophicalPaxos:
    """
    Real implementation of Philosophical Paxos using actual network communication
    """
    
    def __init__(self, node_context: NodeContext, verifier_components: Dict[str, Any],
                 registry_url: Optional[str] = None):
        self.node_context = node_context
        self.verifier_components = verifier_components
        self.logger = logging.getLogger(f"{__name__}.{node_context.node_id}")
        
        # Network components
        self.transport = GRPCTransport(node_context.node_id)
        self.service_registry = ServiceRegistry(registry_url)
        self.peer_discovery = PeerDiscovery(node_context.node_id, self.service_registry)
        self.paxos_protocol = PaxosProtocol(node_context.node_id, self.transport, self.peer_discovery)
        
        # Server configuration
        self.host = "0.0.0.0"
        self.port = self._get_node_port()
        
        # Consensus state
        self.active_proposals: Dict[str, ConsensusProposal] = {}
        self.verification_results: Dict[str, Dict[str, ConsensusVerificationResult]] = {}
        
        # Metrics
        self.metrics = {
            "proposals_submitted": 0,
            "proposals_accepted": 0,
            "proposals_rejected": 0,
            "philosophical_validations": 0,
            "network_messages": 0
        }
        
        self._running = False
    
    def _get_node_port(self) -> int:
        """Get port based on node ID"""
        # Simple port assignment: base_port + hash(node_id) % 1000
        base_port = 9000
        node_hash = hash(self.node_context.node_id)
        return base_port + (abs(node_hash) % 1000)
    
    async def start(self):
        """Start the Philosophical Paxos node"""
        if self._running:
            return
        
        self.logger.info(f"Starting Philosophical Paxos node on {self.host}:{self.port}")
        
        # Start network transport
        await self.transport.start(self.host, self.port)
        
        # Register with service registry
        node_info = NodeInfo(
            node_id=self.node_context.node_id,
            address=self.host,
            port=self.port,
            status=NodeStatus.ONLINE,
            role=self.node_context.node_role.value,
            last_seen=datetime.utcnow(),
            metadata={
                "philosophical_weights": self.node_context.philosophical_weights,
                "trust_threshold": self.node_context.trust_threshold
            }
        )
        
        await self.service_registry.register_node(node_info)
        
        # Start heartbeat
        await self.service_registry.start_heartbeat(self.node_context.node_id)
        
        # Start peer discovery
        await self.peer_discovery.start_discovery()
        
        # Register message handlers
        self._register_handlers()
        
        self._running = True
        self.logger.info("Philosophical Paxos node started successfully")
    
    async def stop(self):
        """Stop the Philosophical Paxos node"""
        if not self._running:
            return
        
        self.logger.info("Stopping Philosophical Paxos node")
        
        # Stop peer discovery
        await self.peer_discovery.stop_discovery()
        
        # Stop heartbeat
        await self.service_registry.stop_heartbeat()
        
        # Deregister from service registry
        await self.service_registry.deregister_node(self.node_context.node_id)
        
        # Stop network transport
        await self.transport.stop()
        
        self._running = False
        self.logger.info("Philosophical Paxos node stopped")
    
    def _register_handlers(self):
        """Register philosophical message handlers"""
        # Paxos handlers are registered in the protocol
        
        # Philosophical handlers
        self.transport.register_handler("philosophical_verification_request", 
                                      self.handle_verification_request)
        self.transport.register_handler("philosophical_verification_response",
                                      self.handle_verification_response)
        self.transport.register_handler("philosophical_debate_request",
                                      self.handle_debate_request)
    
    async def propose_consensus(self, proposal: ConsensusProposal) -> Dict[str, Any]:
        """
        Propose a value for philosophical consensus
        
        This is the main entry point for consensus proposals.
        """
        self.metrics["proposals_submitted"] += 1
        
        try:
            # Store proposal
            self.active_proposals[proposal.proposal_id] = proposal
            self.verification_results[proposal.proposal_id] = {}
            
            # Phase 1: Philosophical Verification
            self.logger.info(f"Starting philosophical verification for proposal {proposal.proposal_id}")
            verification_passed = await self._philosophical_verification_phase(proposal)
            
            if not verification_passed:
                self.metrics["proposals_rejected"] += 1
                return {
                    "success": False,
                    "proposal_id": proposal.proposal_id,
                    "reason": "Failed philosophical verification",
                    "verification_results": self.verification_results.get(proposal.proposal_id, {})
                }
            
            # Phase 2: Paxos Consensus
            self.logger.info(f"Starting Paxos consensus for proposal {proposal.proposal_id}")
            
            # Prepare proposal value with verification results
            proposal_value = {
                "proposal": asdict(proposal),
                "verification_results": self._serialize_verification_results(
                    self.verification_results[proposal.proposal_id]
                ),
                "proposer_id": self.node_context.node_id,
                "timestamp": datetime.utcnow().isoformat()
            }
            
            # Run Paxos
            success, decided_value = await self.paxos_protocol.propose(proposal_value)
            
            if success:
                self.metrics["proposals_accepted"] += 1
                self.logger.info(f"Consensus achieved for proposal {proposal.proposal_id}")
                
                return {
                    "success": True,
                    "proposal_id": proposal.proposal_id,
                    "decided_value": decided_value,
                    "consensus_type": "philosophical_paxos",
                    "participants": len(self.peer_discovery.get_all_peers()) + 1
                }
            else:
                self.metrics["proposals_rejected"] += 1
                return {
                    "success": False,
                    "proposal_id": proposal.proposal_id,
                    "reason": "Paxos consensus failed",
                    "verification_results": self.verification_results.get(proposal.proposal_id, {})
                }
                
        except Exception as e:
            self.logger.error(f"Consensus proposal failed: {e}")
            self.metrics["proposals_rejected"] += 1
            return {
                "success": False,
                "proposal_id": proposal.proposal_id,
                "reason": f"Exception: {str(e)}",
                "error": str(e)
            }
        finally:
            # Cleanup
            asyncio.create_task(self._cleanup_proposal(proposal.proposal_id))
    
    async def _philosophical_verification_phase(self, proposal: ConsensusProposal) -> bool:
        """
        Execute philosophical verification phase
        
        This sends verification requests to all peers and collects their
        philosophical assessments.
        """
        # Get all peer acceptors
        peers = self.peer_discovery.get_all_peers()
        
        # Perform local verification
        self.logger.info("Performing local philosophical verification")
        local_results = await self._perform_local_verification(proposal)
        
        # Store local results
        for framework, result in local_results.items():
            self.verification_results[proposal.proposal_id][f"{self.node_context.node_id}_{framework}"] = result
        
        # Send verification requests to peers
        if peers:
            self.logger.info(f"Requesting philosophical verification from {len(peers)} peers")
            await self._request_peer_verifications(proposal, peers)
            
            # Wait for responses (with timeout)
            await asyncio.sleep(3.0)  # Give peers time to respond
        
        # Evaluate verification results
        return self._evaluate_verification_results(proposal)
    
    async def _perform_local_verification(self, proposal: ConsensusProposal) -> Dict[str, ConsensusVerificationResult]:
        """Perform local philosophical verification"""
        results = {}
        
        for framework_name, component in self.verifier_components.items():
            try:
                self.logger.debug(f"Running {framework_name} verification")
                result = await component.verify_consensus(proposal, self.node_context)
                results[framework_name] = result
                self.metrics["philosophical_validations"] += 1
            except Exception as e:
                self.logger.error(f"Verification failed for {framework_name}: {e}")
                
        return results
    
    async def _request_peer_verifications(self, proposal: ConsensusProposal, peers: List[NodeInfo]):
        """Request philosophical verification from peers"""
        from ...networking import Message
        
        request_msg = Message(
            message_id=str(uuid.uuid4()),
            message_type="philosophical_verification_request",
            sender_id=self.node_context.node_id,
            recipient_id="*",
            payload={
                "proposal": asdict(proposal),
                "requester_id": self.node_context.node_id
            },
            timestamp=datetime.utcnow()
        )
        
        # Send to all peers
        targets = [peer.endpoint for peer in peers]
        results = await self.transport.broadcast_message(request_msg, targets)
        
        self.metrics["network_messages"] += len(targets)
        
        successful_sends = sum(1 for success in results.values() if success)
        self.logger.info(f"Sent verification requests to {successful_sends}/{len(targets)} peers")
    
    async def handle_verification_request(self, message):
        """Handle incoming philosophical verification request"""
        from ...networking import Message
        
        self.logger.debug(f"Received verification request from {message.sender_id}")
        
        # Extract proposal
        proposal_data = message.payload["proposal"]
        proposal = ConsensusProposal(**proposal_data)
        
        # Perform verification
        results = await self._perform_local_verification(proposal)
        
        # Send response
        response_msg = Message(
            message_id=str(uuid.uuid4()),
            message_type="philosophical_verification_response",
            sender_id=self.node_context.node_id,
            recipient_id=message.sender_id,
            payload={
                "proposal_id": proposal.proposal_id,
                "verifier_id": self.node_context.node_id,
                "results": self._serialize_verification_results(results)
            },
            timestamp=datetime.utcnow()
        )
        
        # Get sender endpoint
        sender_info = self.peer_discovery.get_peer(message.sender_id)
        if sender_info:
            await self.transport.send_message(sender_info.endpoint, response_msg)
            self.metrics["network_messages"] += 1
    
    async def handle_verification_response(self, message):
        """Handle incoming philosophical verification response"""
        payload = message.payload
        proposal_id = payload["proposal_id"]
        verifier_id = payload["verifier_id"]
        results = payload["results"]
        
        self.logger.debug(f"Received verification response from {verifier_id} for {proposal_id}")
        
        # Store results
        if proposal_id in self.verification_results:
            for framework, result_data in results.items():
                key = f"{verifier_id}_{framework}"
                # Deserialize result
                result = ConsensusVerificationResult(**result_data)
                self.verification_results[proposal_id][key] = result
    
    async def handle_debate_request(self, message):
        """Handle philosophical debate request"""
        # TODO: Implement philosophical debate mechanism
        self.logger.info(f"Received debate request from {message.sender_id}")
    
    def _evaluate_verification_results(self, proposal: ConsensusProposal) -> bool:
        """
        Evaluate all verification results to determine if proposal should proceed
        """
        results = self.verification_results.get(proposal.proposal_id, {})
        
        if not results:
            self.logger.warning("No verification results available")
            return False
        
        # Count approvals by framework
        framework_approvals = defaultdict(int)
        framework_totals = defaultdict(int)
        
        for key, result in results.items():
            framework = key.split('_')[-1]
            framework_totals[framework] += 1
            
            if result.consensus_readiness:
                framework_approvals[framework] += 1
        
        # Check required verifiers
        if proposal.required_verifiers:
            for required in proposal.required_verifiers:
                if framework_approvals.get(required, 0) == 0:
                    self.logger.warning(f"Required framework {required} has no approvals")
                    return False
        
        # Calculate overall approval rate
        total_approvals = sum(framework_approvals.values())
        total_checks = sum(framework_totals.values())
        
        if total_checks == 0:
            return False
        
        approval_rate = total_approvals / total_checks
        threshold = proposal.criteria.min_confidence if proposal.criteria else 0.6
        
        self.logger.info(f"Verification approval rate: {approval_rate:.2%} (threshold: {threshold:.2%})")
        
        return approval_rate >= threshold
    
    def _serialize_verification_results(self, results: Dict[str, ConsensusVerificationResult]) -> Dict[str, Dict]:
        """Serialize verification results for network transmission"""
        serialized = {}
        for key, result in results.items():
            serialized[key] = asdict(result)
        return serialized
    
    async def _cleanup_proposal(self, proposal_id: str, delay: float = 300.0):
        """Clean up proposal data after delay"""
        await asyncio.sleep(delay)
        
        if proposal_id in self.active_proposals:
            del self.active_proposals[proposal_id]
        
        if proposal_id in self.verification_results:
            del self.verification_results[proposal_id]
        
        self.logger.debug(f"Cleaned up proposal {proposal_id}")
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get comprehensive metrics"""
        return {
            "consensus_metrics": self.metrics,
            "transport_metrics": self.transport.get_metrics(),
            "protocol_metrics": self.paxos_protocol.get_metrics(),
            "active_proposals": len(self.active_proposals),
            "discovered_peers": len(self.peer_discovery.get_all_peers())
        }
    
    def get_network_status(self) -> Dict[str, Any]:
        """Get network status"""
        peers = self.peer_discovery.get_all_peers()
        
        return {
            "node_id": self.node_context.node_id,
            "address": f"{self.host}:{self.port}",
            "status": "online" if self._running else "offline",
            "peers": [
                {
                    "node_id": peer.node_id,
                    "endpoint": peer.endpoint,
                    "status": peer.status.value,
                    "role": peer.role
                }
                for peer in peers
            ],
            "total_peers": len(peers)
        }