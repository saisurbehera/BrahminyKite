"""
Unit tests for consensus components
"""

import unittest
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from verifier.consensus_types import (
    ConsensusProposal, ProposalType, ConsensusConfig, 
    VerificationMode, NodeContext, NodeRole
)


class TestConsensusTypes(unittest.TestCase):
    """Test consensus data types"""
    
    def test_consensus_proposal_creation(self):
        """Test ConsensusProposal creation"""
        proposal = ConsensusProposal(
            proposal_type=ProposalType.CLAIM_VALIDATION,
            content={"test": "data"},
            timeout=30
        )
        
        self.assertEqual(proposal.proposal_type, ProposalType.CLAIM_VALIDATION)
        self.assertEqual(proposal.content["test"], "data")
        self.assertEqual(proposal.timeout, 30)
        self.assertIsNotNone(proposal.proposal_id)
    
    def test_consensus_config_validation(self):
        """Test ConsensusConfig validation"""
        config = ConsensusConfig(
            node_id="test_node",
            peer_nodes=["peer1", "peer2"],
            required_quorum=0.75
        )
        
        errors = config.validate()
        self.assertEqual(len(errors), 0)
        
        # Test invalid config
        invalid_config = ConsensusConfig(required_quorum=1.5)
        errors = invalid_config.validate()
        self.assertGreater(len(errors), 0)
    
    def test_node_context(self):
        """Test NodeContext functionality"""
        context = NodeContext(
            node_id="test_node",
            node_role=NodeRole.PROPOSER,
            peer_nodes=["peer1", "peer2"]
        )
        
        # Test trust management
        context.update_trust("peer1", 0.8)
        self.assertEqual(context.get_trust("peer1"), 0.8)
        self.assertTrue(context.is_trusted_peer("peer1", threshold=0.7))
        self.assertFalse(context.is_trusted_peer("peer1", threshold=0.9))


if __name__ == '__main__':
    unittest.main()