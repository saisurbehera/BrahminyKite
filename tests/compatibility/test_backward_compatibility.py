"""
Backward compatibility tests
"""

import unittest
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from verifier.compatibility import IdealVerifier, create_compatible_verifier
from verifier import Claim, Domain


class TestBackwardCompatibility(unittest.TestCase):
    """Test backward compatibility with original API"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.test_claim = Claim(
            content="Backward compatibility test",
            domain=Domain.EMPIRICAL
        )
    
    def test_original_api_unchanged(self):
        """Test that original API works unchanged"""
        # This should work exactly like the original
        verifier = IdealVerifier()
        result = verifier.verify(self.test_claim)
        
        # Check standard result format
        self.assertIsInstance(result, dict)
        self.assertIn("final_score", result)
        self.assertIn("dominant_framework", result)
        self.assertIn("component_results", result)
        self.assertIn("explanation", result)
    
    def test_factory_function(self):
        """Test factory function for creating compatible verifier"""
        verifier = create_compatible_verifier(
            enable_debate=True,
            enable_parallel=True
        )
        
        result = verifier.verify(self.test_claim)
        self.assertIsInstance(result, dict)
        self.assertIn("final_score", result)
    
    def test_learning_api_compatibility(self):
        """Test learning API compatibility"""
        verifier = IdealVerifier()
        result = verifier.verify(self.test_claim)
        
        feedback = {
            "framework_preference": "positivist",
            "weight_adjustment": 1.1
        }
        
        # This should not raise an exception
        verifier.learn_from_feedback(self.test_claim, result, feedback)
    
    def test_statistics_api_compatibility(self):
        """Test statistics API compatibility"""
        verifier = IdealVerifier()
        verifier.verify(self.test_claim)  # Generate some data
        
        stats = verifier.get_verification_statistics()
        
        self.assertIsInstance(stats, dict)
        # Should have the same structure as original
        self.assertIn("system_config", stats)
        self.assertIn("verification_history", stats)
        self.assertIn("component_statistics", stats)
    
    def test_consensus_extensions_opt_in(self):
        """Test that consensus extensions are opt-in"""
        verifier = IdealVerifier()
        
        # Initially, consensus should not be available
        self.assertFalse(verifier.consensus_enabled)
        
        # But the interface should exist
        self.assertTrue(hasattr(verifier, 'enable_consensus_extensions'))
        self.assertTrue(hasattr(verifier, 'get_consensus_statistics'))
    
    def test_migration_info(self):
        """Test migration information availability"""
        verifier = IdealVerifier()
        
        migration_info = verifier.get_migration_info()
        
        self.assertIsInstance(migration_info, dict)
        self.assertIn("current_state", migration_info)
        self.assertIn("available_migrations", migration_info)
        self.assertIn("recommendations", migration_info)
    
    def test_compatibility_test_suite(self):
        """Test the built-in compatibility test suite"""
        verifier = IdealVerifier()
        
        test_results = verifier.test_backward_compatibility()
        
        self.assertIsInstance(test_results, dict)
        self.assertIn("api_compatibility", test_results)
        self.assertIn("functionality_compatibility", test_results)
        self.assertIn("overall_status", test_results)


if __name__ == '__main__':
    unittest.main()