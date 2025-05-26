"""
gRPC Transport Layer for Consensus Network

This module provides the network transport layer for peer-to-peer communication
in the consensus network. It handles connection management, message routing,
and secure communication between nodes.
"""

import asyncio
import logging
import ssl
from typing import Dict, List, Optional, Callable, Any
from contextlib import asynccontextmanager
import grpc
from grpc import aio
from concurrent.futures import ThreadPoolExecutor
import time

from ..protos import consensus_pb2, consensus_pb2_grpc
from .connection_pool import ConnectionPool
from .metrics import NetworkMetrics

logger = logging.getLogger(__name__)


class NetworkTransport:
    """
    Network transport layer for consensus communication.
    
    Handles gRPC server/client setup, connection pooling, and message routing.
    """
    
    def __init__(
        self,
        node_id: str,
        bind_address: str,
        tls_config: Optional[Dict[str, str]] = None,
        max_workers: int = 10,
        connection_pool_size: int = 10
    ):
        self.node_id = node_id
        self.bind_address = bind_address
        self.tls_config = tls_config or {}
        self.max_workers = max_workers
        
        # Connection management
        self.connection_pool = ConnectionPool(
            max_size=connection_pool_size,
            node_id=node_id
        )
        
        # gRPC server
        self.server: Optional[aio.Server] = None
        self.service_impl: Optional[ConsensusNetworkServicer] = None
        
        # Metrics
        self.metrics = NetworkMetrics(node_id)
        
        # Running state
        self._running = False
        self._server_task: Optional[asyncio.Task] = None
    
    async def start(self, service_impl: 'ConsensusNetworkServicer') -> None:
        """Start the network transport layer."""
        if self._running:
            logger.warning(f"Network transport already running for node {self.node_id}")
            return
        
        self.service_impl = service_impl
        
        # Start gRPC server
        await self._start_server()
        
        # Initialize connection pool
        await self.connection_pool.start()
        
        self._running = True
        logger.info(f"Network transport started for node {self.node_id} on {self.bind_address}")
    
    async def stop(self) -> None:
        """Stop the network transport layer."""
        if not self._running:
            return
        
        self._running = False
        
        # Stop gRPC server
        if self.server:
            await self.server.stop(grace=5.0)
            self.server = None
        
        # Close connection pool
        await self.connection_pool.stop()
        
        logger.info(f"Network transport stopped for node {self.node_id}")
    
    async def _start_server(self) -> None:
        """Start the gRPC server."""
        self.server = aio.server(
            ThreadPoolExecutor(max_workers=self.max_workers)
        )
        
        # Add service to server
        consensus_pb2_grpc.add_ConsensusNetworkServicer_to_server(
            self.service_impl, self.server
        )
        
        # Configure TLS if enabled
        if self.tls_config.get('tls_enabled', False):
            server_credentials = self._create_server_credentials()
            self.server.add_secure_port(self.bind_address, server_credentials)
        else:
            self.server.add_insecure_port(self.bind_address)
        
        await self.server.start()
    
    def _create_server_credentials(self) -> grpc.ServerCredentials:
        """Create gRPC server credentials for TLS."""
        with open(self.tls_config['cert_file'], 'rb') as f:
            server_cert = f.read()
        with open(self.tls_config['key_file'], 'rb') as f:
            server_key = f.read()
        
        ca_cert = None
        if self.tls_config.get('ca_file'):
            with open(self.tls_config['ca_file'], 'rb') as f:
                ca_cert = f.read()
        
        return grpc.ssl_server_credentials(
            [(server_key, server_cert)],
            root_certificates=ca_cert,
            require_client_auth=bool(ca_cert)
        )
    
    def _create_channel_credentials(self) -> grpc.ChannelCredentials:
        """Create gRPC channel credentials for TLS client."""
        ca_cert = None
        if self.tls_config.get('ca_file'):
            with open(self.tls_config['ca_file'], 'rb') as f:
                ca_cert = f.read()
        
        client_cert = None
        client_key = None
        if self.tls_config.get('cert_file') and self.tls_config.get('key_file'):
            with open(self.tls_config['cert_file'], 'rb') as f:
                client_cert = f.read()
            with open(self.tls_config['key_file'], 'rb') as f:
                client_key = f.read()
        
        return grpc.ssl_channel_credentials(
            root_certificates=ca_cert,
            private_key=client_key,
            certificate_chain=client_cert
        )
    
    @asynccontextmanager
    async def get_client(self, peer_address: str):
        """Get a gRPC client connection to a peer."""
        start_time = time.time()
        
        try:
            # Get connection from pool
            channel = await self.connection_pool.get_connection(
                peer_address,
                self._create_channel_factory()
            )
            
            # Create stub
            stub = consensus_pb2_grpc.ConsensusNetworkStub(channel)
            
            # Record metrics
            self.metrics.record_connection_acquired(
                peer_address,
                time.time() - start_time
            )
            
            yield stub
            
        except Exception as e:
            self.metrics.record_connection_error(peer_address, str(e))
            raise
        finally:
            # Return connection to pool
            if 'channel' in locals():
                await self.connection_pool.release_connection(peer_address, channel)
    
    def _create_channel_factory(self) -> Callable[[str], aio.Channel]:
        """Create a factory function for creating gRPC channels."""
        def create_channel(address: str) -> aio.Channel:
            options = [
                ('grpc.keepalive_time_ms', 30000),
                ('grpc.keepalive_timeout_ms', 10000),
                ('grpc.keepalive_permit_without_calls', True),
                ('grpc.http2.max_pings_without_data', 0),
            ]
            
            if self.tls_config.get('tls_enabled', False):
                credentials = self._create_channel_credentials()
                return aio.secure_channel(address, credentials, options=options)
            else:
                return aio.insecure_channel(address, options=options)
        
        return create_channel
    
    async def broadcast(
        self,
        message: Any,
        peers: List[str],
        method_name: str,
        timeout: float = 5.0
    ) -> Dict[str, Any]:
        """
        Broadcast a message to multiple peers.
        
        Args:
            message: The protobuf message to send
            peers: List of peer addresses
            method_name: The RPC method name to call
            timeout: Timeout for each RPC call
            
        Returns:
            Dictionary mapping peer addresses to responses or errors
        """
        results = {}
        
        async def send_to_peer(peer_address: str) -> None:
            try:
                async with self.get_client(peer_address) as stub:
                    method = getattr(stub, method_name)
                    response = await method(message, timeout=timeout)
                    results[peer_address] = response
                    self.metrics.record_message_sent(peer_address, method_name)
            except Exception as e:
                results[peer_address] = e
                self.metrics.record_message_error(peer_address, method_name, str(e))
        
        # Send to all peers concurrently
        await asyncio.gather(
            *[send_to_peer(peer) for peer in peers],
            return_exceptions=True
        )
        
        return results
    
    async def send_to_peer(
        self,
        peer_address: str,
        message: Any,
        method_name: str,
        timeout: float = 5.0
    ) -> Any:
        """
        Send a message to a specific peer.
        
        Args:
            peer_address: The peer's address
            message: The protobuf message to send
            method_name: The RPC method name to call
            timeout: Timeout for the RPC call
            
        Returns:
            The response from the peer
        """
        async with self.get_client(peer_address) as stub:
            method = getattr(stub, method_name)
            response = await method(message, timeout=timeout)
            self.metrics.record_message_sent(peer_address, method_name)
            return response


class ConsensusNetworkServicer(consensus_pb2_grpc.ConsensusNetworkServicer):
    """
    Base implementation of the ConsensusNetwork gRPC service.
    
    This should be extended by the actual consensus implementation.
    """
    
    def __init__(self, node_id: str):
        self.node_id = node_id
        self.logger = logging.getLogger(f"{__name__}.{node_id}")
    
    async def Join(self, request, context):
        """Handle node join requests."""
        self.logger.info(f"Join request from {request.node_info.node_id}")
        # To be implemented by consensus layer
        return consensus_pb2.JoinResponse(accepted=False, reason="Not implemented")
    
    async def Leave(self, request, context):
        """Handle node leave requests."""
        self.logger.info(f"Leave request from {request.node_id}")
        # To be implemented by consensus layer
        return consensus_pb2.LeaveResponse(acknowledged=True)
    
    async def Heartbeat(self, request, context):
        """Handle heartbeat messages."""
        # To be implemented by consensus layer
        return consensus_pb2.HeartbeatResponse(acknowledged=True)
    
    async def GetPeers(self, request, context):
        """Handle peer list requests."""
        # To be implemented by consensus layer
        return consensus_pb2.GetPeersResponse(peers=[])
    
    async def Prepare(self, request, context):
        """Handle Paxos prepare messages."""
        # To be implemented by consensus layer
        return consensus_pb2.PrepareResponse(promised=False)
    
    async def Promise(self, request, context):
        """Handle Paxos promise messages."""
        # To be implemented by consensus layer
        return consensus_pb2.PromiseResponse(acknowledged=False)
    
    async def Accept(self, request, context):
        """Handle Paxos accept messages."""
        # To be implemented by consensus layer
        return consensus_pb2.AcceptResponse(accepted=False)
    
    async def Accepted(self, request, context):
        """Handle Paxos accepted messages."""
        # To be implemented by consensus layer
        return consensus_pb2.AcceptedResponse(acknowledged=False)
    
    async def GetState(self, request, context):
        """Handle state retrieval requests."""
        # To be implemented by consensus layer
        return consensus_pb2.GetStateResponse(state={}, version=0)
    
    async def SyncState(self, request_iterator, context):
        """Handle state synchronization streams."""
        # To be implemented by consensus layer
        async for request in request_iterator:
            yield consensus_pb2.StateSyncResponse(
                ack=consensus_pb2.StateSyncAck(chunk_number=0)
            )
    
    async def GetSnapshot(self, request, context):
        """Handle snapshot retrieval requests."""
        # To be implemented by consensus layer
        yield consensus_pb2.SnapshotChunk(
            chunk_number=0,
            total_chunks=1,
            data=b"",
            checksum=""
        )
    
    async def ApplySnapshot(self, request_iterator, context):
        """Handle snapshot application requests."""
        # To be implemented by consensus layer
        async for chunk in request_iterator:
            pass
        return consensus_pb2.ApplySnapshotResponse(success=False)