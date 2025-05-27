"""
Peer Discovery and Service Registry

Handles node discovery and service registration for distributed consensus.
"""

import asyncio
import json
import logging
from typing import Dict, List, Set, Optional, Any
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
import aiohttp
import socket


class NodeStatus(Enum):
    """Node status in the network"""
    ONLINE = "online"
    OFFLINE = "offline"
    UNREACHABLE = "unreachable"
    UNHEALTHY = "unhealthy"


@dataclass
class NodeInfo:
    """Information about a node in the network"""
    node_id: str
    address: str
    port: int
    status: NodeStatus
    role: str  # proposer, acceptor, learner
    last_seen: datetime
    metadata: Dict[str, Any]
    
    @property
    def endpoint(self) -> str:
        """Get full endpoint address"""
        return f"{self.address}:{self.port}"


class ServiceRegistry:
    """Service registry for node discovery"""
    
    def __init__(self, registry_url: Optional[str] = None):
        self.registry_url = registry_url
        self.local_registry: Dict[str, NodeInfo] = {}
        self.logger = logging.getLogger(__name__)
        
        # For local testing without external registry
        self.use_local_registry = registry_url is None
        
        # Heartbeat settings
        self.heartbeat_interval = 30  # seconds
        self.node_timeout = 90  # seconds
        
        self._heartbeat_task: Optional[asyncio.Task] = None
    
    async def register_node(self, node_info: NodeInfo) -> bool:
        """Register a node with the registry"""
        try:
            if self.use_local_registry:
                # Local registry
                self.local_registry[node_info.node_id] = node_info
                self.logger.info(f"Registered node {node_info.node_id} locally")
                return True
            else:
                # External registry
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        f"{self.registry_url}/nodes",
                        json=asdict(node_info)
                    ) as response:
                        return response.status == 200
                        
        except Exception as e:
            self.logger.error(f"Failed to register node: {e}")
            return False
    
    async def deregister_node(self, node_id: str) -> bool:
        """Deregister a node from the registry"""
        try:
            if self.use_local_registry:
                # Local registry
                if node_id in self.local_registry:
                    del self.local_registry[node_id]
                    self.logger.info(f"Deregistered node {node_id} locally")
                return True
            else:
                # External registry
                async with aiohttp.ClientSession() as session:
                    async with session.delete(
                        f"{self.registry_url}/nodes/{node_id}"
                    ) as response:
                        return response.status == 200
                        
        except Exception as e:
            self.logger.error(f"Failed to deregister node: {e}")
            return False
    
    async def get_nodes(self, role: Optional[str] = None) -> List[NodeInfo]:
        """Get all registered nodes, optionally filtered by role"""
        try:
            if self.use_local_registry:
                # Local registry
                nodes = list(self.local_registry.values())
            else:
                # External registry
                async with aiohttp.ClientSession() as session:
                    url = f"{self.registry_url}/nodes"
                    if role:
                        url += f"?role={role}"
                        
                    async with session.get(url) as response:
                        if response.status == 200:
                            data = await response.json()
                            nodes = [NodeInfo(**node_data) for node_data in data]
                        else:
                            nodes = []
            
            # Filter by role if specified (for local registry)
            if role and self.use_local_registry:
                nodes = [n for n in nodes if n.role == role]
            
            # Filter out stale nodes
            now = datetime.utcnow()
            active_nodes = []
            for node in nodes:
                if isinstance(node.last_seen, str):
                    node.last_seen = datetime.fromisoformat(node.last_seen)
                    
                if (now - node.last_seen).total_seconds() < self.node_timeout:
                    active_nodes.append(node)
                else:
                    node.status = NodeStatus.OFFLINE
                    
            return active_nodes
            
        except Exception as e:
            self.logger.error(f"Failed to get nodes: {e}")
            return []
    
    async def update_node_status(self, node_id: str, status: NodeStatus) -> bool:
        """Update node status"""
        try:
            if self.use_local_registry:
                # Local registry
                if node_id in self.local_registry:
                    self.local_registry[node_id].status = status
                    self.local_registry[node_id].last_seen = datetime.utcnow()
                return True
            else:
                # External registry
                async with aiohttp.ClientSession() as session:
                    async with session.patch(
                        f"{self.registry_url}/nodes/{node_id}/status",
                        json={"status": status.value}
                    ) as response:
                        return response.status == 200
                        
        except Exception as e:
            self.logger.error(f"Failed to update node status: {e}")
            return False
    
    async def heartbeat(self, node_id: str) -> bool:
        """Send heartbeat for a node"""
        return await self.update_node_status(node_id, NodeStatus.ONLINE)
    
    async def start_heartbeat(self, node_id: str):
        """Start periodic heartbeat"""
        async def heartbeat_loop():
            while True:
                try:
                    await self.heartbeat(node_id)
                    await asyncio.sleep(self.heartbeat_interval)
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    self.logger.error(f"Heartbeat error: {e}")
                    await asyncio.sleep(5)  # Retry after 5 seconds
        
        self._heartbeat_task = asyncio.create_task(heartbeat_loop())
        self.logger.info(f"Started heartbeat for node {node_id}")
    
    async def stop_heartbeat(self):
        """Stop heartbeat"""
        if self._heartbeat_task:
            self._heartbeat_task.cancel()
            try:
                await self._heartbeat_task
            except asyncio.CancelledError:
                pass
            self._heartbeat_task = None
            self.logger.info("Stopped heartbeat")


class PeerDiscovery:
    """Peer discovery mechanism for finding other nodes"""
    
    def __init__(self, node_id: str, service_registry: ServiceRegistry):
        self.node_id = node_id
        self.service_registry = service_registry
        self.logger = logging.getLogger(__name__)
        
        # Discovered peers
        self.peers: Dict[str, NodeInfo] = {}
        
        # Discovery settings
        self.discovery_interval = 30  # seconds
        self._discovery_task: Optional[asyncio.Task] = None
        
        # Callbacks
        self.on_peer_discovered: Optional[callable] = None
        self.on_peer_lost: Optional[callable] = None
    
    async def start_discovery(self):
        """Start peer discovery"""
        async def discovery_loop():
            while True:
                try:
                    await self.discover_peers()
                    await asyncio.sleep(self.discovery_interval)
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    self.logger.error(f"Discovery error: {e}")
                    await asyncio.sleep(5)
        
        self._discovery_task = asyncio.create_task(discovery_loop())
        self.logger.info("Started peer discovery")
    
    async def stop_discovery(self):
        """Stop peer discovery"""
        if self._discovery_task:
            self._discovery_task.cancel()
            try:
                await self._discovery_task
            except asyncio.CancelledError:
                pass
            self._discovery_task = None
            self.logger.info("Stopped peer discovery")
    
    async def discover_peers(self) -> List[NodeInfo]:
        """Discover available peers"""
        try:
            # Get all nodes from registry
            all_nodes = await self.service_registry.get_nodes()
            
            # Filter out self
            peer_nodes = [n for n in all_nodes if n.node_id != self.node_id]
            
            # Update peer list
            current_peer_ids = set(self.peers.keys())
            discovered_peer_ids = set(n.node_id for n in peer_nodes)
            
            # Find new peers
            new_peers = discovered_peer_ids - current_peer_ids
            for node in peer_nodes:
                if node.node_id in new_peers:
                    self.peers[node.node_id] = node
                    self.logger.info(f"Discovered new peer: {node.node_id} at {node.endpoint}")
                    
                    if self.on_peer_discovered:
                        asyncio.create_task(self.on_peer_discovered(node))
            
            # Find lost peers
            lost_peers = current_peer_ids - discovered_peer_ids
            for peer_id in lost_peers:
                peer_info = self.peers.pop(peer_id, None)
                self.logger.info(f"Lost peer: {peer_id}")
                
                if peer_info and self.on_peer_lost:
                    asyncio.create_task(self.on_peer_lost(peer_info))
            
            # Update existing peers
            for node in peer_nodes:
                if node.node_id in self.peers:
                    self.peers[node.node_id] = node
            
            return list(self.peers.values())
            
        except Exception as e:
            self.logger.error(f"Peer discovery failed: {e}")
            return []
    
    def get_peers_by_role(self, role: str) -> List[NodeInfo]:
        """Get peers by role"""
        return [p for p in self.peers.values() if p.role == role]
    
    def get_peer(self, node_id: str) -> Optional[NodeInfo]:
        """Get specific peer info"""
        return self.peers.get(node_id)
    
    def get_all_peers(self) -> List[NodeInfo]:
        """Get all discovered peers"""
        return list(self.peers.values())
    
    async def ping_peer(self, node_id: str) -> bool:
        """Ping a peer to check connectivity"""
        peer = self.peers.get(node_id)
        if not peer:
            return False
        
        try:
            # Simple HTTP ping
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"http://{peer.endpoint}/health",
                    timeout=aiohttp.ClientTimeout(total=5)
                ) as response:
                    return response.status == 200
        except:
            return False
    
    def get_quorum_size(self, total_nodes: Optional[int] = None) -> int:
        """Calculate quorum size (majority)"""
        if total_nodes is None:
            total_nodes = len(self.peers) + 1  # Include self
        return (total_nodes // 2) + 1


def get_local_ip() -> str:
    """Get local IP address"""
    try:
        # Create a socket to external host to find local IP
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        local_ip = s.getsockname()[0]
        s.close()
        return local_ip
    except:
        return "127.0.0.1"