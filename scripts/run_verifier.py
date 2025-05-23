#!/usr/bin/env python3
"""Script to run the unified verifier with various configurations."""

import asyncio
import argparse
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from chil import (
    UnifiedIdealVerifier, 
    VerificationMode, 
    Claim, 
    ConsensusProposal, 
    ProposalType,
    DEFAULT_CONFIG
)


def create_sample_claim() -> Claim:
    """Create a sample claim for testing."""
    return Claim(
        content="Climate change is primarily caused by human activities",
        context={"domain": "environmental_science", "urgency": "high"},
        metadata={"source": "IPCC Report", "confidence": 0.95}
    )


def create_sample_proposal() -> ConsensusProposal:
    """Create a sample consensus proposal for testing."""
    claim = create_sample_claim()
    return ConsensusProposal(
        proposal_type=ProposalType.CLAIM_VALIDATION,
        content={
            "claim": claim.__dict__,
            "validation_criteria": ["empirical", "contextual", "consensus"]
        },
        metadata={"priority": "high", "deadline": "2024-01-01"}
    )


async def run_individual_mode():
    """Run verifier in individual mode."""
    print("Running in Individual Mode...")
    verifier = UnifiedIdealVerifier(mode=VerificationMode.INDIVIDUAL)
    
    claim = create_sample_claim()
    result = verifier.verify(claim)
    
    print(f"Verification Result: {result}")
    return result


async def run_consensus_mode():
    """Run verifier in consensus mode."""
    print("Running in Consensus Mode...")
    verifier = UnifiedIdealVerifier(mode=VerificationMode.CONSENSUS)
    
    proposal = create_sample_proposal()
    result = await verifier.propose_consensus(proposal)
    
    print(f"Consensus Result: {result}")
    return result


async def run_hybrid_mode():
    """Run verifier in hybrid mode."""
    print("Running in Hybrid Mode...")
    verifier = UnifiedIdealVerifier(mode=VerificationMode.HYBRID)
    
    # Test individual verification
    claim = create_sample_claim()
    individual_result = verifier.verify(claim)
    print(f"Individual Result: {individual_result}")
    
    # Test consensus proposal
    proposal = create_sample_proposal()
    consensus_result = await verifier.propose_consensus(proposal)
    print(f"Consensus Result: {consensus_result}")
    
    return {"individual": individual_result, "consensus": consensus_result}


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Run Chil Verifier")
    parser.add_argument(
        "--mode", 
        choices=["individual", "consensus", "hybrid"],
        default="individual",
        help="Verification mode to run"
    )
    parser.add_argument(
        "--config",
        help="Path to configuration file"
    )
    
    args = parser.parse_args()
    
    print(f"Chil Unified Verifier")
    print(f"Mode: {args.mode}")
    print("-" * 40)
    
    try:
        if args.mode == "individual":
            await run_individual_mode()
        elif args.mode == "consensus":
            await run_consensus_mode()
        elif args.mode == "hybrid":
            await run_hybrid_mode()
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))