"""
Consensus Module

Implements distributed consensus protocols with philosophical verification
integration, including modified Paxos and distributed debate systems.
"""

from .paxos import PhilosophicalPaxos
from .paxos_real import RealPhilosophicalPaxos
from .consensus_types import *  # Re-export consensus types
from .network import NetworkManager
from .conflict_resolver import ConsensusConflictResolver

__all__ = [
    "PhilosophicalPaxos",
    "RealPhilosophicalPaxos",
    "NetworkManager", 
    "ConsensusConflictResolver"
]