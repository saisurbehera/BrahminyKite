"""
Connection Pool for gRPC Channels

Manages a pool of gRPC channel connections to peers for efficient
resource utilization and connection reuse.
"""

import asyncio
import time
import logging
from typing import Dict, Optional, Callable, Set
from dataclasses import dataclass, field
from contextlib import asynccontextmanager
import grpc
from grpc import aio

logger = logging.getLogger(__name__)


@dataclass
class PooledConnection:
    """Represents a pooled gRPC connection."""
    channel: aio.Channel
    address: str
    created_at: float
    last_used: float
    use_count: int = 0
    in_use: bool = False
    health_check_failures: int = 0


class ConnectionPool:
    """
    Connection pool for managing gRPC channels to peer nodes.
    
    Features:
    - Connection reuse
    - Health checking
    - Automatic reconnection
    - Connection limits
    - Idle timeout
    """
    
    def __init__(
        self,
        max_size: int = 10,
        max_idle_time: float = 300.0,  # 5 minutes
        health_check_interval: float = 30.0,
        max_health_failures: int = 3,
        node_id: str = "unknown"
    ):
        self.max_size = max_size
        self.max_idle_time = max_idle_time
        self.health_check_interval = health_check_interval
        self.max_health_failures = max_health_failures
        self.node_id = node_id
        
        # Connection storage
        self._connections: Dict[str, PooledConnection] = {}
        self._lock = asyncio.Lock()
        
        # Health check task
        self._health_check_task: Optional[asyncio.Task] = None
        self._running = False
        
        # Metrics
        self.total_connections_created = 0
        self.total_connections_reused = 0
        self.total_connections_closed = 0
    
    async def start(self) -> None:
        """Start the connection pool."""
        if self._running:
            return
        
        self._running = True
        self._health_check_task = asyncio.create_task(self._health_check_loop())
        logger.info(f"Connection pool started for node {self.node_id}")
    
    async def stop(self) -> None:
        """Stop the connection pool and close all connections."""
        self._running = False
        
        # Cancel health check task
        if self._health_check_task:
            self._health_check_task.cancel()
            try:
                await self._health_check_task
            except asyncio.CancelledError:
                pass
        
        # Close all connections
        async with self._lock:
            for conn in self._connections.values():
                await self._close_connection(conn)
            self._connections.clear()
        
        logger.info(f"Connection pool stopped for node {self.node_id}")
    
    async def get_connection(
        self,
        address: str,
        channel_factory: Callable[[str], aio.Channel]
    ) -> aio.Channel:
        """
        Get a connection from the pool or create a new one.
        
        Args:
            address: The peer address
            channel_factory: Factory function to create new channels
            
        Returns:
            A gRPC channel
        """
        async with self._lock:
            # Check for existing connection
            if address in self._connections:
                conn = self._connections[address]
                if not conn.in_use and await self._is_connection_healthy(conn):
                    conn.in_use = True
                    conn.last_used = time.time()
                    conn.use_count += 1
                    self.total_connections_reused += 1
                    logger.debug(f"Reusing connection to {address}")
                    return conn.channel
                elif conn.in_use:
                    # Connection is in use, create a new one if pool not full
                    if len(self._connections) < self.max_size:
                        return await self._create_new_connection(address, channel_factory)
                    else:
                        # Wait for connection to be available
                        logger.warning(f"Connection pool full, waiting for connection to {address}")
                        # In a real implementation, we'd implement a wait queue
                        raise RuntimeError(f"Connection pool full for {address}")
            
            # Create new connection
            if len(self._connections) < self.max_size:
                return await self._create_new_connection(address, channel_factory)
            else:
                # Try to evict idle connections
                await self._evict_idle_connections()
                if len(self._connections) < self.max_size:
                    return await self._create_new_connection(address, channel_factory)
                else:
                    raise RuntimeError(f"Connection pool full, cannot connect to {address}")
    
    async def release_connection(self, address: str, channel: aio.Channel) -> None:
        """Release a connection back to the pool."""
        async with self._lock:
            if address in self._connections:
                conn = self._connections[address]
                if conn.channel == channel:
                    conn.in_use = False
                    conn.last_used = time.time()
                    logger.debug(f"Released connection to {address}")
    
    async def _create_new_connection(
        self,
        address: str,
        channel_factory: Callable[[str], aio.Channel]
    ) -> aio.Channel:
        """Create a new connection."""
        try:
            channel = channel_factory(address)
            
            # Wait for channel to be ready
            await channel.channel_ready()
            
            conn = PooledConnection(
                channel=channel,
                address=address,
                created_at=time.time(),
                last_used=time.time(),
                in_use=True,
                use_count=1
            )
            
            self._connections[address] = conn
            self.total_connections_created += 1
            
            logger.info(f"Created new connection to {address}")
            return channel
            
        except Exception as e:
            logger.error(f"Failed to create connection to {address}: {e}")
            raise
    
    async def _close_connection(self, conn: PooledConnection) -> None:
        """Close a connection."""
        try:
            await conn.channel.close()
            self.total_connections_closed += 1
            logger.debug(f"Closed connection to {conn.address}")
        except Exception as e:
            logger.error(f"Error closing connection to {conn.address}: {e}")
    
    async def _is_connection_healthy(self, conn: PooledConnection) -> bool:
        """Check if a connection is healthy."""
        try:
            # Check channel state
            state = conn.channel.get_state(try_to_connect=True)
            if state == grpc.ChannelConnectivity.READY:
                conn.health_check_failures = 0
                return True
            else:
                conn.health_check_failures += 1
                return conn.health_check_failures < self.max_health_failures
        except Exception as e:
            logger.error(f"Health check failed for {conn.address}: {e}")
            conn.health_check_failures += 1
            return False
    
    async def _evict_idle_connections(self) -> None:
        """Evict idle connections from the pool."""
        current_time = time.time()
        to_remove = []
        
        for address, conn in self._connections.items():
            if not conn.in_use and (current_time - conn.last_used) > self.max_idle_time:
                to_remove.append(address)
        
        for address in to_remove:
            conn = self._connections.pop(address)
            await self._close_connection(conn)
            logger.info(f"Evicted idle connection to {address}")
    
    async def _health_check_loop(self) -> None:
        """Periodic health check for all connections."""
        while self._running:
            try:
                await asyncio.sleep(self.health_check_interval)
                
                async with self._lock:
                    unhealthy = []
                    
                    for address, conn in self._connections.items():
                        if not conn.in_use and not await self._is_connection_healthy(conn):
                            if conn.health_check_failures >= self.max_health_failures:
                                unhealthy.append(address)
                    
                    # Remove unhealthy connections
                    for address in unhealthy:
                        conn = self._connections.pop(address)
                        await self._close_connection(conn)
                        logger.warning(f"Removed unhealthy connection to {address}")
                    
                    # Evict idle connections
                    await self._evict_idle_connections()
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in health check loop: {e}")
    
    def get_stats(self) -> Dict[str, int]:
        """Get connection pool statistics."""
        active_connections = sum(1 for conn in self._connections.values() if conn.in_use)
        idle_connections = len(self._connections) - active_connections
        
        return {
            "total_connections": len(self._connections),
            "active_connections": active_connections,
            "idle_connections": idle_connections,
            "total_created": self.total_connections_created,
            "total_reused": self.total_connections_reused,
            "total_closed": self.total_connections_closed,
            "pool_size": self.max_size
        }