"""
Distributed Networking Layer for BrahminyKite

Provides real network communication for distributed consensus.
"""

from .transport import NetworkTransport, GRPCTransport, Message
from .discovery import PeerDiscovery, ServiceRegistry
from .protocol import PaxosProtocol, ConsensusProtocol

__all__ = [
    'NetworkTransport',
    'GRPCTransport', 
    'Message',
    'PeerDiscovery',
    'ServiceRegistry',
    'PaxosProtocol',
    'ConsensusProtocol'
]