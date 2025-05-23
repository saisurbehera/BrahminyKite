"""
BrahminyKite Unified Verification Framework

A modular, multi-framework verification system that integrates philosophical approaches
to truth and verification across both objective and subjective domains, supporting
both individual claim verification and distributed consensus protocols.

## Quick Start

### Individual Verification (Backward Compatible)
```python
from verifier import IdealVerifier, Claim, Domain

# Original API - no changes needed
verifier = IdealVerifier()
claim = Claim("The Earth orbits the Sun", Domain.EMPIRICAL)
result = verifier.verify(claim)
```

### Unified Framework (New Capabilities)
```python
from verifier import UnifiedIdealVerifier, ConsensusConfig, VerificationMode

# Individual mode
verifier = UnifiedIdealVerifier(mode=VerificationMode.INDIVIDUAL)

# Consensus mode
consensus_config = ConsensusConfig(
    node_id="node_1",
    peer_nodes=["node_2", "node_3"],
    required_quorum=0.6
)
verifier = UnifiedIdealVerifier(
    mode=VerificationMode.CONSENSUS,
    consensus_config=consensus_config
)
```

### Gradual Migration
```python
from verifier.compatibility import create_compatible_verifier

# Backward compatible with optional consensus
verifier = create_compatible_verifier()
verifier.enable_consensus_extensions()  # Opt-in to new features
```
"""

# Core framework exports
from .frameworks import (
    VerificationFramework, Domain, Claim, VerificationResult,
    get_framework_descriptions, get_domain_characteristics
)

from .consensus_types import (
    VerificationMode, ConsensusProposal, ProposalType, ConsensusConfig,
    NodeContext, NodeRole, ApprovalStatus, UnifiedResult
)

# Main verifier classes
from .unified_core import UnifiedIdealVerifier
from .compatibility import IdealVerifier, create_compatible_verifier, create_unified_verifier

# Components (for advanced usage)
from .components import (
    EmpiricalVerifier, ContextualVerifier, ConsistencyVerifier,
    PowerDynamicsVerifier, UtilityVerifier, EvolutionaryVerifier
)

# Enhanced components
from .components.unified_base import UnifiedVerificationComponent
from .components.empirical_unified import UnifiedEmpiricalVerifier

# Systems
from .meta import MetaVerificationSystem
from .systems import DebateSystem
from .bridge import ModeBridge
from .consensus import PhilosophicalPaxos

# Version and metadata
__version__ = "2.0.0"
__author__ = "BrahminyKite Project"
__description__ = "Unified philosophical verification framework with consensus support"

# Main exports (what users typically need)
__all__ = [
    # Primary interfaces
    "IdealVerifier",                    # Backward compatible interface
    "UnifiedIdealVerifier",            # New unified interface
    "create_compatible_verifier",       # Factory for backward compatibility
    "create_unified_verifier",         # Factory for unified verifier
    
    # Core data types
    "Claim",                           # Individual claims
    "ConsensusProposal",              # Consensus proposals
    "VerificationResult",             # Individual results
    "UnifiedResult",                  # Cross-mode results
    
    # Enums and configuration
    "Domain",                         # Verification domains
    "VerificationFramework",          # Philosophical frameworks
    "VerificationMode",               # Individual/Consensus/Hybrid
    "ProposalType",                   # Types of consensus proposals
    "ConsensusConfig",               # Consensus configuration
    "NodeContext",                   # Node information for consensus
    
    # Component interfaces (for custom components)
    "UnifiedVerificationComponent",   # Base class for unified components
    
    # Utility functions
    "get_framework_descriptions",     # Framework documentation
    "get_domain_characteristics",     # Domain information
    
    # Version info
    "__version__"
]

# Convenience imports for common use cases
def quick_verify(content: str, domain: Domain = Domain.EMPIRICAL) -> dict:
    """
    Quick verification for simple use cases
    
    Args:
        content: The claim content to verify
        domain: The domain of the claim
        
    Returns:
        Verification result dictionary
    """
    verifier = IdealVerifier()
    claim = Claim(content=content, domain=domain)
    return verifier.verify(claim)


def quick_consensus(content: str, proposal_type: ProposalType = ProposalType.CLAIM_VALIDATION,
                   peer_nodes: list = None) -> dict:
    """
    Quick consensus verification for simple use cases
    
    Args:
        content: The proposal content
        proposal_type: Type of consensus proposal
        peer_nodes: List of peer node IDs
        
    Returns:
        Consensus result (note: this is a demo - real implementation needs async)
    """
    import asyncio
    
    config = ConsensusConfig(
        node_id="quick_consensus_node",
        peer_nodes=peer_nodes or ["peer_1", "peer_2"]
    )
    
    verifier = UnifiedIdealVerifier(
        mode=VerificationMode.CONSENSUS,
        consensus_config=config
    )
    
    proposal = ConsensusProposal(
        proposal_type=proposal_type,
        content={"description": content}
    )
    
    # For demo purposes - real usage should handle async properly
    try:
        return asyncio.run(verifier.propose_consensus(proposal))
    except Exception as e:
        return {"error": str(e), "note": "Consensus requires proper async handling"}


# Add quick functions to exports for convenience
__all__.extend(["quick_verify", "quick_consensus"])

# Package information
def get_package_info() -> dict:
    """Get information about the BrahminyKite package"""
    return {
        "name": "BrahminyKite Unified Verification Framework",
        "version": __version__,
        "description": __description__,
        "author": __author__,
        "features": [
            "Multi-framework philosophical verification",
            "Individual claim verification",
            "Distributed consensus protocols",
            "Seamless mode switching",
            "Backward compatibility",
            "Adaptive learning",
            "Cross-mode analysis"
        ],
        "philosophical_frameworks": [f.value for f in VerificationFramework],
        "verification_domains": [d.value for d in Domain],
        "verification_modes": [m.value for m in VerificationMode],
        "components": [
            "EmpiricalVerifier", "ContextualVerifier", "ConsistencyVerifier",
            "PowerDynamicsVerifier", "UtilityVerifier", "EvolutionaryVerifier"
        ]
    }

# Add package info to exports
__all__.append("get_package_info")