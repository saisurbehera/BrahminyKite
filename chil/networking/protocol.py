"""
Consensus Protocol Implementation

Implements the actual Paxos protocol using the networking layer.
"""

import asyncio
import uuid
import logging
from typing import Dict, List, Optional, Any, Set, Tuple
from datetime import datetime
from dataclasses import dataclass
from abc import ABC, abstractmethod
from enum import Enum

from .transport import NetworkTransport, Message
from .discovery import PeerDiscovery, NodeInfo


class ConsensusPhase(Enum):
    """Phases of consensus protocol"""
    IDLE = "idle"
    PREPARE = "prepare"
    ACCEPT = "accept"
    DECIDED = "decided"
    FAILED = "failed"


@dataclass
class ConsensusState:
    """State of a consensus instance"""
    proposal_id: str
    phase: ConsensusPhase
    proposal_number: int
    value: Any
    promises_received: Dict[str, Any]
    accepts_received: Dict[str, bool]
    quorum_size: int
    decided_value: Optional[Any] = None
    
    @property
    def has_promise_quorum(self) -> bool:
        """Check if we have enough promises"""
        return len(self.promises_received) >= self.quorum_size
    
    @property
    def has_accept_quorum(self) -> bool:
        """Check if we have enough accepts"""
        accepted_count = sum(1 for accepted in self.accepts_received.values() if accepted)
        return accepted_count >= self.quorum_size


class ConsensusProtocol(ABC):
    """Abstract base class for consensus protocols"""
    
    @abstractmethod
    async def propose(self, value: Any) -> Tuple[bool, Any]:
        """Propose a value for consensus"""
        pass
    
    @abstractmethod
    async def handle_message(self, message: Message):
        """Handle incoming consensus message"""
        pass


class PaxosProtocol(ConsensusProtocol):
    """Real Paxos protocol implementation"""
    
    def __init__(self, node_id: str, transport: NetworkTransport, 
                 peer_discovery: PeerDiscovery):
        self.node_id = node_id
        self.transport = transport
        self.peer_discovery = peer_discovery
        self.logger = logging.getLogger(f"{__name__}.{node_id}")
        
        # Paxos state
        self.proposal_number = 0
        self.highest_proposal_seen = 0
        self.accepted_proposal_number: Optional[int] = None
        self.accepted_value: Optional[Any] = None
        
        # Active consensus instances
        self.active_proposals: Dict[str, ConsensusState] = {}
        
        # Register message handlers
        self._register_handlers()
        
        # Metrics
        self.metrics = {
            "proposals_initiated": 0,
            "proposals_succeeded": 0,
            "proposals_failed": 0,
            "messages_sent": 0,
            "messages_received": 0
        }
    
    def _register_handlers(self):
        """Register message handlers"""
        self.transport.register_handler("paxos_prepare", self.handle_prepare)
        self.transport.register_handler("paxos_promise", self.handle_promise)
        self.transport.register_handler("paxos_accept", self.handle_accept)
        self.transport.register_handler("paxos_accepted", self.handle_accepted)
    
    async def propose(self, value: Any) -> Tuple[bool, Any]:
        """Propose a value for consensus"""
        self.metrics["proposals_initiated"] += 1
        
        # Generate unique proposal ID
        proposal_id = str(uuid.uuid4())
        
        # Get acceptors (all peers)
        acceptors = self.peer_discovery.get_all_peers()
        if not acceptors:
            self.logger.error("No acceptors available")
            self.metrics["proposals_failed"] += 1
            return False, None
        
        # Calculate quorum size
        total_nodes = len(acceptors) + 1  # Include self
        quorum_size = (total_nodes // 2) + 1
        
        # Initialize consensus state
        state = ConsensusState(
            proposal_id=proposal_id,
            phase=ConsensusPhase.PREPARE,
            proposal_number=self._next_proposal_number(),
            value=value,
            promises_received={},
            accepts_received={},
            quorum_size=quorum_size
        )
        self.active_proposals[proposal_id] = state
        
        try:
            # Phase 1: Prepare
            success = await self._phase_prepare(state, acceptors)
            if not success:
                state.phase = ConsensusPhase.FAILED
                self.metrics["proposals_failed"] += 1
                return False, None
            
            # Phase 2: Accept
            success = await self._phase_accept(state, acceptors)
            if not success:
                state.phase = ConsensusPhase.FAILED
                self.metrics["proposals_failed"] += 1
                return False, None
            
            # Success
            state.phase = ConsensusPhase.DECIDED
            self.metrics["proposals_succeeded"] += 1
            return True, state.decided_value
            
        except Exception as e:
            self.logger.error(f"Consensus proposal failed: {e}")
            state.phase = ConsensusPhase.FAILED
            self.metrics["proposals_failed"] += 1
            return False, None
        finally:
            # Clean up after timeout
            asyncio.create_task(self._cleanup_proposal(proposal_id))
    
    async def _phase_prepare(self, state: ConsensusState, acceptors: List[NodeInfo]) -> bool:
        """Execute Paxos Phase 1 (Prepare)"""
        self.logger.info(f"Starting prepare phase for proposal {state.proposal_id}")
        
        # Send prepare to all acceptors (including self)
        prepare_msg = Message(
            message_id=str(uuid.uuid4()),
            message_type="paxos_prepare",
            sender_id=self.node_id,
            recipient_id="*",  # Broadcast
            payload={
                "proposal_id": state.proposal_id,
                "proposal_number": state.proposal_number
            },
            timestamp=datetime.utcnow()
        )
        
        # Handle self
        self_promise = await self._handle_prepare_self(state.proposal_number)
        state.promises_received[self.node_id] = self_promise
        
        # Send to peers
        targets = [peer.endpoint for peer in acceptors]
        await self.transport.broadcast_message(prepare_msg, targets)
        self.metrics["messages_sent"] += len(targets)
        
        # Wait for promises (with timeout)
        try:
            await asyncio.wait_for(
                self._wait_for_promises(state),
                timeout=5.0  # 5 second timeout
            )
        except asyncio.TimeoutError:
            self.logger.warning(f"Prepare phase timeout for {state.proposal_id}")
        
        # Check if we have quorum
        if state.has_promise_quorum:
            self.logger.info(f"Prepare phase succeeded with {len(state.promises_received)} promises")
            
            # Check for previously accepted values
            highest_accepted = None
            highest_proposal = -1
            
            for promise in state.promises_received.values():
                if promise.get("accepted_proposal_number", -1) > highest_proposal:
                    highest_proposal = promise["accepted_proposal_number"]
                    highest_accepted = promise.get("accepted_value")
            
            # Use previously accepted value if any
            if highest_accepted is not None:
                state.value = highest_accepted
                self.logger.info(f"Using previously accepted value from proposal {highest_proposal}")
            
            return True
        else:
            self.logger.warning(f"Prepare phase failed - only {len(state.promises_received)} promises")
            return False
    
    async def _phase_accept(self, state: ConsensusState, acceptors: List[NodeInfo]) -> bool:
        """Execute Paxos Phase 2 (Accept)"""
        self.logger.info(f"Starting accept phase for proposal {state.proposal_id}")
        
        state.phase = ConsensusPhase.ACCEPT
        
        # Send accept to all acceptors who promised
        accept_msg = Message(
            message_id=str(uuid.uuid4()),
            message_type="paxos_accept",
            sender_id=self.node_id,
            recipient_id="*",
            payload={
                "proposal_id": state.proposal_id,
                "proposal_number": state.proposal_number,
                "value": state.value
            },
            timestamp=datetime.utcnow()
        )
        
        # Handle self
        self_accepted = await self._handle_accept_self(state.proposal_number, state.value)
        state.accepts_received[self.node_id] = self_accepted
        
        # Send to peers who promised
        promising_peers = []
        for peer in acceptors:
            if peer.node_id in state.promises_received:
                promising_peers.append(peer.endpoint)
        
        if promising_peers:
            await self.transport.broadcast_message(accept_msg, promising_peers)
            self.metrics["messages_sent"] += len(promising_peers)
        
        # Wait for accepts
        try:
            await asyncio.wait_for(
                self._wait_for_accepts(state),
                timeout=5.0
            )
        except asyncio.TimeoutError:
            self.logger.warning(f"Accept phase timeout for {state.proposal_id}")
        
        # Check if we have quorum
        if state.has_accept_quorum:
            state.decided_value = state.value
            self.logger.info(f"Accept phase succeeded - value decided: {state.decided_value}")
            return True
        else:
            accepted_count = sum(1 for a in state.accepts_received.values() if a)
            self.logger.warning(f"Accept phase failed - only {accepted_count} accepts")
            return False
    
    async def handle_prepare(self, message: Message):
        """Handle incoming prepare message"""
        self.metrics["messages_received"] += 1
        
        payload = message.payload
        proposal_number = payload["proposal_number"]
        proposal_id = payload["proposal_id"]
        
        self.logger.debug(f"Received prepare for proposal {proposal_id} with number {proposal_number}")
        
        # Check if we can promise
        can_promise = proposal_number > self.highest_proposal_seen
        
        if can_promise:
            self.highest_proposal_seen = proposal_number
            
        # Send promise
        promise_msg = Message(
            message_id=str(uuid.uuid4()),
            message_type="paxos_promise",
            sender_id=self.node_id,
            recipient_id=message.sender_id,
            payload={
                "proposal_id": proposal_id,
                "proposal_number": proposal_number,
                "promised": can_promise,
                "highest_proposal_seen": self.highest_proposal_seen,
                "accepted_proposal_number": self.accepted_proposal_number,
                "accepted_value": self.accepted_value
            },
            timestamp=datetime.utcnow()
        )
        
        # Get sender endpoint
        sender_info = self.peer_discovery.get_peer(message.sender_id)
        if sender_info:
            await self.transport.send_message(sender_info.endpoint, promise_msg)
            self.metrics["messages_sent"] += 1
    
    async def handle_promise(self, message: Message):
        """Handle incoming promise message"""
        self.metrics["messages_received"] += 1
        
        payload = message.payload
        proposal_id = payload["proposal_id"]
        
        # Find active proposal
        if proposal_id not in self.active_proposals:
            self.logger.warning(f"Received promise for unknown proposal {proposal_id}")
            return
        
        state = self.active_proposals[proposal_id]
        
        # Record promise
        if payload["promised"]:
            state.promises_received[message.sender_id] = payload
            self.logger.debug(f"Received promise from {message.sender_id} for {proposal_id}")
    
    async def handle_accept(self, message: Message):
        """Handle incoming accept message"""
        self.metrics["messages_received"] += 1
        
        payload = message.payload
        proposal_number = payload["proposal_number"]
        proposal_id = payload["proposal_id"]
        value = payload["value"]
        
        self.logger.debug(f"Received accept for proposal {proposal_id}")
        
        # Check if we can accept
        can_accept = proposal_number >= self.highest_proposal_seen
        
        if can_accept:
            self.accepted_proposal_number = proposal_number
            self.accepted_value = value
        
        # Send accepted
        accepted_msg = Message(
            message_id=str(uuid.uuid4()),
            message_type="paxos_accepted",
            sender_id=self.node_id,
            recipient_id=message.sender_id,
            payload={
                "proposal_id": proposal_id,
                "proposal_number": proposal_number,
                "accepted": can_accept
            },
            timestamp=datetime.utcnow()
        )
        
        # Get sender endpoint
        sender_info = self.peer_discovery.get_peer(message.sender_id)
        if sender_info:
            await self.transport.send_message(sender_info.endpoint, accepted_msg)
            self.metrics["messages_sent"] += 1
    
    async def handle_accepted(self, message: Message):
        """Handle incoming accepted message"""
        self.metrics["messages_received"] += 1
        
        payload = message.payload
        proposal_id = payload["proposal_id"]
        
        # Find active proposal
        if proposal_id not in self.active_proposals:
            self.logger.warning(f"Received accepted for unknown proposal {proposal_id}")
            return
        
        state = self.active_proposals[proposal_id]
        
        # Record accepted
        state.accepts_received[message.sender_id] = payload["accepted"]
        self.logger.debug(f"Received accepted from {message.sender_id} for {proposal_id}")
    
    async def handle_message(self, message: Message):
        """Route message to appropriate handler"""
        handler_map = {
            "paxos_prepare": self.handle_prepare,
            "paxos_promise": self.handle_promise,
            "paxos_accept": self.handle_accept,
            "paxos_accepted": self.handle_accepted
        }
        
        handler = handler_map.get(message.message_type)
        if handler:
            await handler(message)
        else:
            self.logger.warning(f"Unknown message type: {message.message_type}")
    
    def _next_proposal_number(self) -> int:
        """Generate next proposal number"""
        self.proposal_number += 1
        return self.proposal_number
    
    async def _handle_prepare_self(self, proposal_number: int) -> Dict[str, Any]:
        """Handle prepare for self as acceptor"""
        can_promise = proposal_number > self.highest_proposal_seen
        
        if can_promise:
            self.highest_proposal_seen = proposal_number
        
        return {
            "promised": can_promise,
            "highest_proposal_seen": self.highest_proposal_seen,
            "accepted_proposal_number": self.accepted_proposal_number,
            "accepted_value": self.accepted_value
        }
    
    async def _handle_accept_self(self, proposal_number: int, value: Any) -> bool:
        """Handle accept for self as acceptor"""
        can_accept = proposal_number >= self.highest_proposal_seen
        
        if can_accept:
            self.accepted_proposal_number = proposal_number
            self.accepted_value = value
        
        return can_accept
    
    async def _wait_for_promises(self, state: ConsensusState):
        """Wait for promise quorum"""
        while not state.has_promise_quorum and state.phase == ConsensusPhase.PREPARE:
            await asyncio.sleep(0.1)
    
    async def _wait_for_accepts(self, state: ConsensusState):
        """Wait for accept quorum"""
        while not state.has_accept_quorum and state.phase == ConsensusPhase.ACCEPT:
            await asyncio.sleep(0.1)
    
    async def _cleanup_proposal(self, proposal_id: str, delay: float = 60.0):
        """Clean up proposal state after delay"""
        await asyncio.sleep(delay)
        if proposal_id in self.active_proposals:
            del self.active_proposals[proposal_id]
            self.logger.debug(f"Cleaned up proposal {proposal_id}")
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get protocol metrics"""
        return {
            **self.metrics,
            "active_proposals": len(self.active_proposals),
            "highest_proposal_seen": self.highest_proposal_seen,
            "accepted_proposal_number": self.accepted_proposal_number
        }