"""
Unit tests for framework components
"""

import unittest
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from verifier.framework import VerificationFramework, Domain, Claim, VerificationResult


class TestFramework(unittest.TestCase):
    """Test core framework components"""
    
    def test_verification_framework_enum(self):
        """Test VerificationFramework enum"""
        self.assertEqual(len(VerificationFramework), 6)
        self.assertIn(VerificationFramework.POSITIVIST, VerificationFramework)
        self.assertIn(VerificationFramework.CONSTRUCTIVIST, VerificationFramework)
    
    def test_domain_enum(self):
        """Test Domain enum"""
        self.assertEqual(len(Domain), 6)
        self.assertIn(Domain.EMPIRICAL, Domain)
        self.assertIn(Domain.AESTHETIC, Domain)
    
    def test_claim_creation(self):
        """Test Claim data structure"""
        claim = Claim(
            content="Test claim",
            domain=Domain.EMPIRICAL,
            context={"test": True}
        )
        
        self.assertEqual(claim.content, "Test claim")
        self.assertEqual(claim.domain, Domain.EMPIRICAL)
        self.assertEqual(claim.context["test"], True)
    
    def test_verification_result_creation(self):
        """Test VerificationResult data structure"""
        result = VerificationResult(
            score=0.85,
            framework=VerificationFramework.POSITIVIST,
            component="test",
            evidence={"test_evidence": True},
            confidence_interval=(0.75, 0.95)
        )
        
        self.assertEqual(result.score, 0.85)
        self.assertEqual(result.framework, VerificationFramework.POSITIVIST)
        self.assertEqual(result.confidence_interval, (0.75, 0.95))


if __name__ == '__main__':
    unittest.main()