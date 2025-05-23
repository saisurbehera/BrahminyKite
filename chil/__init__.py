"""
Chil - Unified Philosophical Verification and Distributed Consensus Framework

Named after the Chilika Chil (Brahminy Kite), this framework combines individual 
claim verification with distributed consensus protocols using philosophical foundations.

The framework supports three modes:
- Individual: Traditional claim verification 
- Consensus: Distributed verification using philosophical Paxos
- Hybrid: Both capabilities simultaneously
"""

__version__ = "0.1.0"
__author__ = "BrahminyKite Project"

# Core exports
from .framework import (
    VerificationMode,
    Claim,
    ConsensusProposal, 
    ProposalType,
    VerificationResult,
    ConsensusVerificationResult
)

from .system import UnifiedIdealVerifier, CompatibilityLayer

# Configuration
from .config.default_config import DEFAULT_CONFIG, SystemConfig

__all__ = [
    # Core types
    "VerificationMode",
    "Claim", 
    "ConsensusProposal",
    "ProposalType", 
    "VerificationResult",
    "ConsensusVerificationResult",
    
    # Main verifier
    "UnifiedIdealVerifier",
    "CompatibilityLayer",
    
    # Configuration
    "DEFAULT_CONFIG",
    "SystemConfig",
    
    # Package info
    "__version__",
    "__author__"
]


def create_verifier(mode: VerificationMode = VerificationMode.INDIVIDUAL) -> UnifiedIdealVerifier:
    """Create a new verifier instance with the specified mode.
    
    Args:
        mode: Verification mode (INDIVIDUAL, CONSENSUS, or HYBRID)
        
    Returns:
        UnifiedIdealVerifier instance configured for the specified mode
    """
    return UnifiedIdealVerifier(mode=mode)


def get_version() -> str:
    """Get the current version of Chil."""
    return __version__