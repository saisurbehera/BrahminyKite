"""
Peer Discovery and Membership Management

Handles peer discovery, health monitoring, and membership management
for the consensus network.
"""

import asyncio
import time
import logging
import random
from typing import Dict, List, Set, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
import socket
import dns.resolver

from ..protos import consensus_pb2
from ..network import NetworkTransport

logger = logging.getLogger(__name__)


class PeerStatus(Enum):
    """Peer health status."""
    HEALTHY = "healthy"
    SUSPECT = "suspect"
    FAILED = "failed"
    UNKNOWN = "unknown"


@dataclass
class PeerInfo:
    """Information about a peer node."""
    node_id: str
    address: str
    role: consensus_pb2.NodeRole
    status: PeerStatus = PeerStatus.UNKNOWN
    last_heartbeat: float = 0.0
    heartbeat_failures: int = 0
    join_time: float = field(default_factory=time.time)
    metadata: Dict[str, str] = field(default_factory=dict)
    
    def to_proto(self) -> consensus_pb2.NodeInfo:
        """Convert to protobuf message."""
        return consensus_pb2.NodeInfo(
            node_id=self.node_id,
            address=self.address,
            role=self.role,
            status=self._status_to_proto(),
            metadata=self.metadata
        )
    
    def _status_to_proto(self) -> consensus_pb2.NodeStatus:
        """Convert status enum to protobuf."""
        mapping = {
            PeerStatus.HEALTHY: consensus_pb2.NODE_STATUS_HEALTHY,
            PeerStatus.SUSPECT: consensus_pb2.NODE_STATUS_SUSPECT,
            PeerStatus.FAILED: consensus_pb2.NODE_STATUS_FAILED,
            PeerStatus.UNKNOWN: consensus_pb2.NODE_STATUS_UNSPECIFIED,
        }
        return mapping.get(self.status, consensus_pb2.NODE_STATUS_UNSPECIFIED)


class PeerDiscovery:
    """
    Handles peer discovery through various mechanisms.
    
    Supports:
    - Static peer list
    - DNS-based discovery
    - Kubernetes service discovery
    """
    
    def __init__(self, discovery_config: Dict[str, any]):
        self.config = discovery_config
        self.discovery_type = discovery_config.get('type', 'static')
        
    async def discover_peers(self) -> List[str]:
        """Discover peer addresses based on configuration."""
        if self.discovery_type == 'static':
            return self._discover_static()
        elif self.discovery_type == 'dns':
            return await self._discover_dns()
        elif self.discovery_type == 'kubernetes':
            return await self._discover_kubernetes()
        else:
            logger.warning(f"Unknown discovery type: {self.discovery_type}")
            return []
    
    def _discover_static(self) -> List[str]:
        """Static peer discovery from configuration."""
        peers = self.config.get('peers', [])
        logger.info(f"Static discovery found {len(peers)} peers")
        return peers
    
    async def _discover_dns(self) -> List[str]:
        """DNS-based peer discovery."""
        domain = self.config.get('domain')
        port = self.config.get('port', 7000)
        
        if not domain:
            logger.error("DNS discovery requires 'domain' configuration")
            return []
        
        try:
            # Resolve A records
            resolver = dns.resolver.Resolver()
            answers = resolver.resolve(domain, 'A')
            
            peers = [f"{rdata.address}:{port}" for rdata in answers]
            logger.info(f"DNS discovery found {len(peers)} peers for {domain}")
            return peers
            
        except Exception as e:
            logger.error(f"DNS discovery failed: {e}")
            return []
    
    async def _discover_kubernetes(self) -> List[str]:
        """Kubernetes service discovery."""
        # This would use the Kubernetes API to discover peer pods
        # For now, return empty list
        logger.warning("Kubernetes discovery not yet implemented")
        return []


class PeerManager:
    """
    Manages peer membership and health monitoring.
    
    Features:
    - Peer registration and tracking
    - Health monitoring via heartbeats
    - Failure detection (Phi Accrual)
    - Leader election support
    """
    
    def __init__(
        self,
        node_id: str,
        network: NetworkTransport,
        discovery: PeerDiscovery,
        heartbeat_interval: float = 1.0,
        suspect_threshold: float = 8.0,
        failure_threshold: float = 16.0,
        max_heartbeat_failures: int = 5
    ):
        self.node_id = node_id
        self.network = network
        self.discovery = discovery
        self.heartbeat_interval = heartbeat_interval
        self.suspect_threshold = suspect_threshold
        self.failure_threshold = failure_threshold
        self.max_heartbeat_failures = max_heartbeat_failures
        
        # Peer tracking
        self.peers: Dict[str, PeerInfo] = {}
        self._lock = asyncio.Lock()
        
        # Self info
        self.self_info = PeerInfo(
            node_id=node_id,
            address=network.bind_address,
            role=consensus_pb2.NODE_ROLE_FOLLOWER
        )
        
        # Leader tracking
        self.current_leader: Optional[str] = None
        self.current_term: int = 0
        
        # Callbacks
        self.on_peer_joined: Optional[Callable[[PeerInfo], None]] = None
        self.on_peer_left: Optional[Callable[[PeerInfo], None]] = None
        self.on_peer_failed: Optional[Callable[[PeerInfo], None]] = None
        self.on_leader_changed: Optional[Callable[[Optional[str]], None]] = None
        
        # Background tasks
        self._heartbeat_task: Optional[asyncio.Task] = None
        self._discovery_task: Optional[asyncio.Task] = None
        self._failure_detector_task: Optional[asyncio.Task] = None
        self._running = False
    
    async def start(self) -> None:
        """Start the peer manager."""
        if self._running:
            return
        
        self._running = True
        
        # Discover initial peers
        await self._discover_and_join_peers()
        
        # Start background tasks
        self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())
        self._discovery_task = asyncio.create_task(self._discovery_loop())
        self._failure_detector_task = asyncio.create_task(self._failure_detection_loop())
        
        logger.info(f"Peer manager started for node {self.node_id}")
    
    async def stop(self) -> None:
        """Stop the peer manager."""
        self._running = False
        
        # Cancel background tasks
        for task in [self._heartbeat_task, self._discovery_task, self._failure_detector_task]:
            if task:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        
        # Notify peers we're leaving
        await self._notify_leave()
        
        logger.info(f"Peer manager stopped for node {self.node_id}")
    
    async def add_peer(self, peer_info: PeerInfo) -> bool:
        """Add a peer to the membership."""
        async with self._lock:
            if peer_info.node_id == self.node_id:
                return False  # Don't add ourselves
            
            is_new = peer_info.node_id not in self.peers
            self.peers[peer_info.node_id] = peer_info
            
            if is_new:
                logger.info(f"Added new peer: {peer_info.node_id} at {peer_info.address}")
                if self.on_peer_joined:
                    asyncio.create_task(self.on_peer_joined(peer_info))
            
            return True
    
    async def remove_peer(self, node_id: str) -> bool:
        """Remove a peer from membership."""
        async with self._lock:
            if node_id in self.peers:
                peer_info = self.peers.pop(node_id)
                logger.info(f"Removed peer: {node_id}")
                
                if self.on_peer_left:
                    asyncio.create_task(self.on_peer_left(peer_info))
                
                # Check if this was the leader
                if self.current_leader == node_id:
                    self.current_leader = None
                    if self.on_leader_changed:
                        asyncio.create_task(self.on_leader_changed(None))
                
                return True
            return False
    
    async def get_peers(self, include_failed: bool = False) -> List[PeerInfo]:
        """Get list of known peers."""
        async with self._lock:
            if include_failed:
                return list(self.peers.values())
            else:
                return [
                    p for p in self.peers.values()
                    if p.status != PeerStatus.FAILED
                ]
    
    async def get_healthy_peers(self) -> List[PeerInfo]:
        """Get list of healthy peers."""
        async with self._lock:
            return [
                p for p in self.peers.values()
                if p.status == PeerStatus.HEALTHY
            ]
    
    def update_leader(self, leader_id: Optional[str], term: int) -> None:
        """Update the current leader."""
        if self.current_term < term:
            self.current_term = term
            self.current_leader = leader_id
            logger.info(f"Leader changed to {leader_id} for term {term}")
            
            if self.on_leader_changed:
                asyncio.create_task(self.on_leader_changed(leader_id))
    
    async def _discover_and_join_peers(self) -> None:
        """Discover peers and attempt to join them."""
        peer_addresses = await self.discovery.discover_peers()
        
        for address in peer_addresses:
            if address == self.network.bind_address:
                continue  # Skip self
            
            try:
                # Send join request
                request = consensus_pb2.JoinRequest(
                    node_info=self.self_info.to_proto(),
                    cluster_token=self.network.tls_config.get('cluster_token', '')
                )
                
                response = await self.network.send_to_peer(
                    address,
                    request,
                    'Join',
                    timeout=5.0
                )
                
                if response.accepted:
                    # Add peers from response
                    for peer_proto in response.peers:
                        peer_info = PeerInfo(
                            node_id=peer_proto.node_id,
                            address=peer_proto.address,
                            role=peer_proto.role,
                            metadata=dict(peer_proto.metadata)
                        )
                        await self.add_peer(peer_info)
                    
                    logger.info(f"Successfully joined cluster via {address}")
                    break
                else:
                    logger.warning(f"Join request rejected by {address}: {response.reason}")
                    
            except Exception as e:
                logger.error(f"Failed to join via {address}: {e}")
    
    async def _heartbeat_loop(self) -> None:
        """Send periodic heartbeats to peers."""
        while self._running:
            try:
                await asyncio.sleep(self.heartbeat_interval)
                
                # Get current peers
                peers = await self.get_peers()
                
                # Send heartbeat to each peer
                request = consensus_pb2.HeartbeatRequest(
                    node_id=self.node_id,
                    term=self.current_term,
                    status=consensus_pb2.NODE_STATUS_HEALTHY,
                    metrics={
                        "cpu_usage": str(self._get_cpu_usage()),
                        "memory_usage": str(self._get_memory_usage()),
                    }
                )
                
                # Broadcast heartbeat
                peer_addresses = [p.address for p in peers]
                responses = await self.network.broadcast(
                    request,
                    peer_addresses,
                    'Heartbeat',
                    timeout=2.0
                )
                
                # Process responses
                for address, response in responses.items():
                    if isinstance(response, Exception):
                        await self._handle_heartbeat_failure(address)
                    else:
                        await self._handle_heartbeat_response(address, response)
                        
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in heartbeat loop: {e}")
    
    async def _discovery_loop(self) -> None:
        """Periodically discover new peers."""
        while self._running:
            try:
                # Wait longer between discoveries
                await asyncio.sleep(30.0)
                
                # Discover peers
                peer_addresses = await self.discovery.discover_peers()
                
                # Check for new peers
                async with self._lock:
                    known_addresses = {p.address for p in self.peers.values()}
                
                for address in peer_addresses:
                    if address not in known_addresses and address != self.network.bind_address:
                        # Attempt to connect to new peer
                        logger.info(f"Discovered new peer at {address}")
                        # The new peer will be added when they send a heartbeat
                        
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in discovery loop: {e}")
    
    async def _failure_detection_loop(self) -> None:
        """Detect failed peers using Phi Accrual failure detector."""
        while self._running:
            try:
                await asyncio.sleep(1.0)  # Check every second
                
                current_time = time.time()
                
                async with self._lock:
                    for peer in list(self.peers.values()):
                        # Calculate phi value
                        time_since_heartbeat = current_time - peer.last_heartbeat
                        phi = self._calculate_phi(time_since_heartbeat)
                        
                        # Update peer status based on phi
                        if phi < self.suspect_threshold:
                            peer.status = PeerStatus.HEALTHY
                        elif phi < self.failure_threshold:
                            if peer.status != PeerStatus.SUSPECT:
                                peer.status = PeerStatus.SUSPECT
                                logger.warning(f"Peer {peer.node_id} is now suspect (phi={phi:.2f})")
                        else:
                            if peer.status != PeerStatus.FAILED:
                                peer.status = PeerStatus.FAILED
                                logger.error(f"Peer {peer.node_id} has failed (phi={phi:.2f})")
                                if self.on_peer_failed:
                                    asyncio.create_task(self.on_peer_failed(peer))
                                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in failure detection loop: {e}")
    
    def _calculate_phi(self, time_since_heartbeat: float) -> float:
        """
        Calculate Phi value for failure detection.
        
        Simplified version of Phi Accrual Failure Detector.
        """
        # Expected heartbeat interval with some variance
        expected_interval = self.heartbeat_interval * 1.5
        
        # Calculate phi based on how many expected intervals have passed
        phi = time_since_heartbeat / expected_interval
        
        return phi
    
    async def _handle_heartbeat_failure(self, address: str) -> None:
        """Handle heartbeat failure for a peer."""
        async with self._lock:
            # Find peer by address
            for peer in self.peers.values():
                if peer.address == address:
                    peer.heartbeat_failures += 1
                    if peer.heartbeat_failures >= self.max_heartbeat_failures:
                        peer.status = PeerStatus.FAILED
                    break
    
    async def _handle_heartbeat_response(
        self,
        address: str,
        response: consensus_pb2.HeartbeatResponse
    ) -> None:
        """Handle heartbeat response from a peer."""
        async with self._lock:
            # Find peer by address
            for peer in self.peers.values():
                if peer.address == address:
                    peer.last_heartbeat = time.time()
                    peer.heartbeat_failures = 0
                    peer.status = PeerStatus.HEALTHY
                    break
        
        # Check for leader updates
        if response.current_term > self.current_term:
            self.update_leader(response.leader_id, response.current_term)
    
    async def _notify_leave(self) -> None:
        """Notify peers that we're leaving."""
        peers = await self.get_peers()
        
        request = consensus_pb2.LeaveRequest(
            node_id=self.node_id,
            reason="Shutting down"
        )
        
        # Best effort notification
        peer_addresses = [p.address for p in peers]
        await self.network.broadcast(
            request,
            peer_addresses,
            'Leave',
            timeout=1.0
        )
    
    def _get_cpu_usage(self) -> float:
        """Get current CPU usage percentage."""
        # Simplified implementation
        return 0.0
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage percentage."""
        # Simplified implementation  
        return 0.0