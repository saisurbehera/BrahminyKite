"""
Network metrics collection for consensus layer.

Tracks connection health, message rates, and network performance.
"""

import time
import threading
from typing import Dict, DefaultDict, Optional
from collections import defaultdict
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)


@dataclass
class ConnectionMetrics:
    """Metrics for a single peer connection."""
    messages_sent: int = 0
    messages_received: int = 0
    bytes_sent: int = 0
    bytes_received: int = 0
    errors: int = 0
    last_error: Optional[str] = None
    last_error_time: Optional[float] = None
    connection_attempts: int = 0
    successful_connections: int = 0
    average_latency_ms: float = 0.0
    latency_samples: int = 0


class NetworkMetrics:
    """
    Collects and tracks network metrics for the consensus layer.
    
    Thread-safe implementation for concurrent access.
    """
    
    def __init__(self, node_id: str):
        self.node_id = node_id
        self._lock = threading.RLock()
        
        # Per-peer metrics
        self.peer_metrics: DefaultDict[str, ConnectionMetrics] = defaultdict(ConnectionMetrics)
        
        # Global metrics
        self.total_messages_sent = 0
        self.total_messages_received = 0
        self.total_bytes_sent = 0
        self.total_bytes_received = 0
        
        # Message type metrics
        self.message_counts: DefaultDict[str, int] = defaultdict(int)
        self.message_errors: DefaultDict[str, int] = defaultdict(int)
        
        # Timing metrics
        self.start_time = time.time()
    
    def record_message_sent(
        self,
        peer: str,
        message_type: str,
        size_bytes: int = 0
    ) -> None:
        """Record a sent message."""
        with self._lock:
            self.peer_metrics[peer].messages_sent += 1
            self.peer_metrics[peer].bytes_sent += size_bytes
            self.total_messages_sent += 1
            self.total_bytes_sent += size_bytes
            self.message_counts[f"sent_{message_type}"] += 1
    
    def record_message_received(
        self,
        peer: str,
        message_type: str,
        size_bytes: int = 0
    ) -> None:
        """Record a received message."""
        with self._lock:
            self.peer_metrics[peer].messages_received += 1
            self.peer_metrics[peer].bytes_received += size_bytes
            self.total_messages_received += 1
            self.total_bytes_received += size_bytes
            self.message_counts[f"received_{message_type}"] += 1
    
    def record_message_error(
        self,
        peer: str,
        message_type: str,
        error: str
    ) -> None:
        """Record a message error."""
        with self._lock:
            self.peer_metrics[peer].errors += 1
            self.peer_metrics[peer].last_error = error
            self.peer_metrics[peer].last_error_time = time.time()
            self.message_errors[message_type] += 1
    
    def record_connection_attempt(self, peer: str, success: bool) -> None:
        """Record a connection attempt."""
        with self._lock:
            self.peer_metrics[peer].connection_attempts += 1
            if success:
                self.peer_metrics[peer].successful_connections += 1
    
    def record_connection_acquired(self, peer: str, duration_ms: float) -> None:
        """Record connection acquisition time."""
        with self._lock:
            metrics = self.peer_metrics[peer]
            # Update running average
            total_latency = metrics.average_latency_ms * metrics.latency_samples
            metrics.latency_samples += 1
            metrics.average_latency_ms = (total_latency + duration_ms) / metrics.latency_samples
    
    def record_connection_error(self, peer: str, error: str) -> None:
        """Record a connection error."""
        with self._lock:
            self.peer_metrics[peer].errors += 1
            self.peer_metrics[peer].last_error = f"Connection error: {error}"
            self.peer_metrics[peer].last_error_time = time.time()
    
    def get_peer_metrics(self, peer: str) -> ConnectionMetrics:
        """Get metrics for a specific peer."""
        with self._lock:
            return ConnectionMetrics(**vars(self.peer_metrics[peer]))
    
    def get_all_metrics(self) -> Dict[str, any]:
        """Get all collected metrics."""
        with self._lock:
            uptime = time.time() - self.start_time
            
            # Calculate rates
            message_rate = self.total_messages_sent / uptime if uptime > 0 else 0
            bytes_rate = self.total_bytes_sent / uptime if uptime > 0 else 0
            
            # Peer summary
            peer_count = len(self.peer_metrics)
            healthy_peers = sum(
                1 for m in self.peer_metrics.values()
                if m.errors == 0 or (m.last_error_time and 
                    time.time() - m.last_error_time > 300)  # 5 minutes
            )
            
            return {
                "node_id": self.node_id,
                "uptime_seconds": uptime,
                "global": {
                    "total_messages_sent": self.total_messages_sent,
                    "total_messages_received": self.total_messages_received,
                    "total_bytes_sent": self.total_bytes_sent,
                    "total_bytes_received": self.total_bytes_received,
                    "message_rate_per_sec": message_rate,
                    "bytes_rate_per_sec": bytes_rate,
                },
                "peers": {
                    "total": peer_count,
                    "healthy": healthy_peers,
                    "unhealthy": peer_count - healthy_peers,
                },
                "message_types": dict(self.message_counts),
                "message_errors": dict(self.message_errors),
                "peer_details": {
                    peer: vars(metrics) for peer, metrics in self.peer_metrics.items()
                }
            }
    
    def get_summary(self) -> Dict[str, any]:
        """Get a summary of key metrics."""
        with self._lock:
            uptime = time.time() - self.start_time
            
            return {
                "node_id": self.node_id,
                "uptime_seconds": uptime,
                "total_peers": len(self.peer_metrics),
                "messages_sent": self.total_messages_sent,
                "messages_received": self.total_messages_received,
                "total_errors": sum(m.errors for m in self.peer_metrics.values()),
                "average_latency_ms": sum(
                    m.average_latency_ms for m in self.peer_metrics.values()
                ) / len(self.peer_metrics) if self.peer_metrics else 0
            }
    
    def reset_peer_metrics(self, peer: str) -> None:
        """Reset metrics for a specific peer."""
        with self._lock:
            if peer in self.peer_metrics:
                del self.peer_metrics[peer]
    
    def export_prometheus_format(self) -> str:
        """Export metrics in Prometheus format."""
        with self._lock:
            lines = []
            
            # Add metric descriptions
            lines.append("# HELP consensus_messages_sent_total Total messages sent")
            lines.append("# TYPE consensus_messages_sent_total counter")
            lines.append(f'consensus_messages_sent_total{{node_id="{self.node_id}"}} {self.total_messages_sent}')
            
            lines.append("# HELP consensus_messages_received_total Total messages received")
            lines.append("# TYPE consensus_messages_received_total counter")
            lines.append(f'consensus_messages_received_total{{node_id="{self.node_id}"}} {self.total_messages_received}')
            
            lines.append("# HELP consensus_bytes_sent_total Total bytes sent")
            lines.append("# TYPE consensus_bytes_sent_total counter")
            lines.append(f'consensus_bytes_sent_total{{node_id="{self.node_id}"}} {self.total_bytes_sent}')
            
            lines.append("# HELP consensus_peer_errors_total Total errors per peer")
            lines.append("# TYPE consensus_peer_errors_total counter")
            for peer, metrics in self.peer_metrics.items():
                lines.append(f'consensus_peer_errors_total{{node_id="{self.node_id}",peer="{peer}"}} {metrics.errors}')
            
            lines.append("# HELP consensus_peer_latency_ms Average latency per peer")
            lines.append("# TYPE consensus_peer_latency_ms gauge")
            for peer, metrics in self.peer_metrics.items():
                if metrics.average_latency_ms > 0:
                    lines.append(f'consensus_peer_latency_ms{{node_id="{self.node_id}",peer="{peer}"}} {metrics.average_latency_ms}')
            
            return "\n".join(lines)