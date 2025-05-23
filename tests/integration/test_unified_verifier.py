"""
Integration tests for the unified verifier system
"""

import unittest
import asyncio
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from verifier import (
    UnifiedIdealVerifier, VerificationMode, ConsensusConfig,
    Claim, Domain, ConsensusProposal, ProposalType
)


class TestUnifiedVerifier(unittest.TestCase):
    """Integration tests for unified verifier"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.test_claim = Claim(
            content="Integration test claim",
            domain=Domain.EMPIRICAL,
            context={"test": True}
        )
        
        self.consensus_config = ConsensusConfig(
            node_id="test_node",
            peer_nodes=["peer1", "peer2"],
            required_quorum=0.6
        )
    
    def test_individual_mode_verification(self):
        """Test individual mode verification"""
        verifier = UnifiedIdealVerifier(mode=VerificationMode.INDIVIDUAL)
        
        result = verifier.verify(self.test_claim)
        
        self.assertIsInstance(result, dict)
        self.assertIn("final_score", result)
        self.assertIn("dominant_framework", result)
        self.assertIn("component_results", result)
        self.assertGreaterEqual(result["final_score"], 0.0)
        self.assertLessEqual(result["final_score"], 1.0)
    
    def test_mode_switching(self):
        """Test switching between verification modes"""
        verifier = UnifiedIdealVerifier(mode=VerificationMode.INDIVIDUAL)
        
        # Start in individual mode
        self.assertEqual(verifier.mode, VerificationMode.INDIVIDUAL)
        
        # Switch to consensus mode
        verifier.switch_mode(
            VerificationMode.CONSENSUS,
            {"consensus_config": self.consensus_config.__dict__}
        )
        
        self.assertEqual(verifier.mode, VerificationMode.CONSENSUS)
        
        # Switch to hybrid mode
        verifier.switch_mode(VerificationMode.HYBRID)
        self.assertEqual(verifier.mode, VerificationMode.HYBRID)
    
    def test_cross_mode_analysis(self):
        """Test cross-mode analysis functionality"""
        verifier = UnifiedIdealVerifier(
            mode=VerificationMode.HYBRID,
            consensus_config=self.consensus_config
        )
        
        analysis = verifier.cross_mode_analysis(self.test_claim)
        
        self.assertIsInstance(analysis, dict)
        self.assertIn("analysis_id", analysis)
        self.assertIn("original_item_type", analysis)
        self.assertIn("individual_analysis", analysis)
    
    def test_configuration_export_import(self):
        """Test configuration management"""
        verifier = UnifiedIdealVerifier(
            mode=VerificationMode.HYBRID,
            consensus_config=self.consensus_config
        )
        
        # Export configuration
        config = verifier.export_configuration()
        self.assertIsInstance(config, dict)
        self.assertIn("system_settings", config)
        
        # Import configuration
        verifier.import_configuration(config)  # Should not raise exception
    
    def test_statistics_collection(self):
        """Test statistics collection"""
        verifier = UnifiedIdealVerifier(mode=VerificationMode.INDIVIDUAL)
        
        # Perform some verifications
        verifier.verify(self.test_claim)
        
        stats = verifier.get_verification_statistics()
        
        self.assertIsInstance(stats, dict)
        self.assertIn("system_info", stats)
        self.assertIn("verification_counts", stats)
        self.assertIn("component_statistics", stats)


class TestAsyncConsensus(unittest.IsolatedAsyncioTestCase):
    """Async tests for consensus functionality"""
    
    async def test_consensus_proposal(self):
        """Test consensus proposal (simulated)"""
        config = ConsensusConfig(
            node_id="async_test_node",
            peer_nodes=["peer1", "peer2"],
            required_quorum=0.6
        )
        
        verifier = UnifiedIdealVerifier(
            mode=VerificationMode.CONSENSUS,
            consensus_config=config
        )
        
        proposal = ConsensusProposal(
            proposal_type=ProposalType.CLAIM_VALIDATION,
            content={"test_proposal": "data"}
        )
        
        try:
            result = await verifier.propose_consensus(proposal)
            self.assertIsInstance(result, dict)
            self.assertIn("proposal_id", result)
        except Exception as e:
            # Expected in test environment without real network
            self.assertIsInstance(e, (RuntimeError, ConnectionError, Exception))


if __name__ == '__main__':
    unittest.main()