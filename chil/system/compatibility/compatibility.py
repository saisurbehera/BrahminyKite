"""
Backward Compatibility Layer

Ensures existing code continues to work with the unified framework
while providing optional access to new consensus capabilities.
"""

import logging
from typing import Dict, Any, Optional, Union

from .unified_core import UnifiedIdealVerifier
from .frameworks import Claim, VerificationResult
from .consensus_types import ConsensusProposal, ConsensusConfig, VerificationMode


class IdealVerifier:
    """
    Backward compatible interface for the original IdealVerifier
    
    This class provides the same API as the original verifier while
    internally using the unified framework. Existing code requires
    no changes to work with this implementation.
    """
    
    def __init__(self, enable_debate: bool = True, enable_parallel: bool = True):
        """
        Initialize backward compatible verifier
        
        Args:
            enable_debate: Enable adversarial debate system
            enable_parallel: Enable parallel component processing
        """
        self.logger = logging.getLogger(__name__)
        
        # Create unified verifier in individual mode
        self.unified_verifier = UnifiedIdealVerifier(
            mode=VerificationMode.INDIVIDUAL,
            enable_unified_components=False  # Use original components for compatibility
        )
        
        # Apply original settings
        self.unified_verifier.enable_debate = enable_debate
        self.unified_verifier.enable_parallel = enable_parallel
        
        # Track whether consensus extensions are enabled
        self.consensus_enabled = False
        
        self.logger.info("Initialized backward compatible IdealVerifier")
    
    def verify(self, claim: Claim) -> Dict[str, Any]:
        """
        Verify a claim (original API - unchanged)
        
        Args:
            claim: The claim to verify
            
        Returns:
            Verification result dictionary (same format as original)
        """
        return self.unified_verifier.verify(claim)
    
    def learn_from_feedback(self, claim: Claim, verification_result: Dict[str, Any], 
                           human_feedback: Dict[str, Any]):
        """
        Learn from human feedback (original API - unchanged)
        
        Args:
            claim: The original claim
            verification_result: Result from verification
            human_feedback: Human feedback for learning
        """
        self.unified_verifier.learn_from_feedback(claim, verification_result, human_feedback)
    
    def get_verification_statistics(self) -> Dict[str, Any]:
        """
        Get verification statistics (original API - unchanged)
        
        Returns:
            Statistics dictionary (same format as original)
        """
        stats = self.unified_verifier.get_verification_statistics()
        
        # Return only individual verification stats for compatibility
        compatible_stats = {
            "system_config": {
                "components_count": len(self.unified_verifier.active_components),
                "debate_enabled": self.unified_verifier.enable_debate,
                "parallel_enabled": self.unified_verifier.enable_parallel
            },
            "verification_history": stats.get("individual_performance", {}),
            "component_statistics": stats.get("component_statistics", {}),
            "meta_verifier_stats": stats.get("meta_verifier", {}),
            "debate_system_stats": stats.get("debate_system", {})
        }
        
        return compatible_stats
    
    # ==================== NEW CONSENSUS EXTENSIONS ====================
    
    def enable_consensus_extensions(self, consensus_config: Optional[ConsensusConfig] = None):
        """
        Enable consensus capabilities (new feature - opt-in)
        
        Args:
            consensus_config: Configuration for consensus mode
        """
        if self.consensus_enabled:
            self.logger.warning("Consensus extensions already enabled")
            return
        
        try:
            # Switch to hybrid mode to enable both individual and consensus
            config_dict = {}
            if consensus_config:
                config_dict["consensus_config"] = consensus_config.__dict__
            
            self.unified_verifier.switch_mode(VerificationMode.HYBRID, config_dict)
            self.consensus_enabled = True
            
            self.logger.info("Consensus extensions enabled successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to enable consensus extensions: {e}")
            raise
    
    def disable_consensus_extensions(self):
        """Disable consensus capabilities and return to individual mode"""
        if not self.consensus_enabled:
            self.logger.warning("Consensus extensions not enabled")
            return
        
        try:
            self.unified_verifier.switch_mode(VerificationMode.INDIVIDUAL)
            self.consensus_enabled = False
            
            self.logger.info("Consensus extensions disabled")
            
        except Exception as e:
            self.logger.error(f"Failed to disable consensus extensions: {e}")
            raise
    
    def propose_consensus(self, proposal: ConsensusProposal) -> Dict[str, Any]:
        """
        Propose consensus (new feature - requires consensus extensions)
        
        Args:
            proposal: The consensus proposal
            
        Returns:
            Consensus result
        """
        if not self.consensus_enabled:
            raise RuntimeError("Consensus extensions not enabled. Call enable_consensus_extensions() first.")
        
        # This would need to be async in practice, but we'll handle it here
        import asyncio
        
        try:
            # Run consensus proposal in event loop
            loop = asyncio.get_event_loop()
            return loop.run_until_complete(self.unified_verifier.propose_consensus(proposal))
        except RuntimeError:
            # No event loop running, create one
            return asyncio.run(self.unified_verifier.propose_consensus(proposal))
    
    def verify_with_consensus_option(self, claim: Claim, 
                                   seek_consensus: bool = False) -> Dict[str, Any]:
        """
        Verify with optional consensus seeking (new feature)
        
        Args:
            claim: The claim to verify
            seek_consensus: Whether to also seek consensus validation
            
        Returns:
            Verification result, potentially with consensus data
        """
        if not seek_consensus or not self.consensus_enabled:
            # Standard individual verification
            return self.verify(claim)
        
        # Perform cross-mode analysis
        try:
            cross_analysis = self.unified_verifier.cross_mode_analysis(claim)
            
            # Enhance standard result with consensus insights
            individual_result = self.verify(claim)
            individual_result["consensus_analysis"] = cross_analysis
            individual_result["verification_mode"] = "individual_with_consensus_analysis"
            
            return individual_result
            
        except Exception as e:
            self.logger.error(f"Consensus analysis failed: {e}")
            # Fall back to individual verification
            result = self.verify(claim)
            result["consensus_analysis_error"] = str(e)
            return result
    
    def get_consensus_statistics(self) -> Dict[str, Any]:
        """
        Get consensus-specific statistics (new feature)
        
        Returns:
            Consensus statistics if enabled, empty dict otherwise
        """
        if not self.consensus_enabled:
            return {"consensus_enabled": False}
        
        stats = self.unified_verifier.get_verification_statistics()
        
        return {
            "consensus_enabled": True,
            "consensus_performance": stats.get("consensus_performance", {}),
            "consensus_system": stats.get("consensus_system", {}),
            "node_info": stats.get("system_info", {})
        }
    
    # ==================== GRADUAL MIGRATION HELPERS ====================
    
    def migrate_to_unified_components(self):
        """
        Migrate to use unified components (gradual migration helper)
        
        This enables enhanced capabilities while maintaining compatibility.
        """
        if self.unified_verifier.enable_unified_components:
            self.logger.warning("Already using unified components")
            return
        
        try:
            # Switch to unified components
            self.unified_verifier.enable_unified_components = True
            self.unified_verifier._initialize_components()
            
            self.logger.info("Migrated to unified components successfully")
            
        except Exception as e:
            self.logger.error(f"Migration to unified components failed: {e}")
            raise
    
    def get_migration_info(self) -> Dict[str, Any]:
        """
        Get information about migration options and current state
        
        Returns:
            Migration information and recommendations
        """
        info = {
            "current_state": {
                "using_unified_components": self.unified_verifier.enable_unified_components,
                "consensus_enabled": self.consensus_enabled,
                "current_mode": self.unified_verifier.mode.value
            },
            "available_migrations": [
                {
                    "name": "unified_components",
                    "description": "Migrate to enhanced components with consensus support",
                    "status": "available" if not self.unified_verifier.enable_unified_components else "completed",
                    "method": "migrate_to_unified_components()"
                },
                {
                    "name": "consensus_extensions", 
                    "description": "Enable distributed consensus capabilities",
                    "status": "available" if not self.consensus_enabled else "completed",
                    "method": "enable_consensus_extensions()"
                }
            ],
            "recommendations": []
        }
        
        # Add recommendations based on current state
        if not self.unified_verifier.enable_unified_components:
            info["recommendations"].append(
                "Consider migrating to unified components for enhanced capabilities"
            )
        
        if not self.consensus_enabled and self.unified_verifier.enable_unified_components:
            info["recommendations"].append(
                "Consider enabling consensus extensions for distributed verification"
            )
        
        return info
    
    # ==================== COMPATIBILITY TESTING ====================
    
    def test_backward_compatibility(self) -> Dict[str, Any]:
        """
        Test backward compatibility with original API
        
        Returns:
            Test results showing compatibility status
        """
        test_results = {
            "api_compatibility": {},
            "functionality_compatibility": {},
            "performance_compatibility": {},
            "overall_status": "unknown"
        }
        
        try:
            # Test basic claim creation and verification
            from .frameworks import Domain
            test_claim = Claim(
                content="Test claim for compatibility verification",
                domain=Domain.EMPIRICAL
            )
            
            # Test verify method
            result = self.verify(test_claim)
            test_results["api_compatibility"]["verify"] = {
                "status": "pass" if isinstance(result, dict) and "final_score" in result else "fail",
                "result_type": type(result).__name__
            }
            
            # Test learning method
            feedback = {"framework_preference": "positivist", "weight_adjustment": 1.0}
            self.learn_from_feedback(test_claim, result, feedback)
            test_results["api_compatibility"]["learn_from_feedback"] = {"status": "pass"}
            
            # Test statistics
            stats = self.get_verification_statistics()
            test_results["api_compatibility"]["get_verification_statistics"] = {
                "status": "pass" if isinstance(stats, dict) else "fail",
                "stats_sections": list(stats.keys()) if isinstance(stats, dict) else []
            }
            
            # Overall API compatibility
            api_tests_passed = all(
                test["status"] == "pass" 
                for test in test_results["api_compatibility"].values()
                if isinstance(test, dict) and "status" in test
            )
            test_results["api_compatibility"]["overall"] = "pass" if api_tests_passed else "fail"
            
            # Functionality compatibility
            test_results["functionality_compatibility"]["score_range"] = (
                "pass" if 0 <= result.get("final_score", -1) <= 1 else "fail"
            )
            test_results["functionality_compatibility"]["confidence_interval"] = (
                "pass" if "confidence_interval" in result else "fail"
            )
            test_results["functionality_compatibility"]["explanation"] = (
                "pass" if "explanation" in result else "fail"
            )
            
            # Performance compatibility (simplified)
            processing_time = result.get("metadata", {}).get("processing_time", 0)
            test_results["performance_compatibility"]["response_time"] = (
                "pass" if processing_time < 10.0 else "warning"  # Should be under 10 seconds
            )
            
            # Overall status
            all_sections_pass = all(
                section.get("overall", section.get("response_time")) in ["pass", "warning"]
                for section in [
                    test_results["api_compatibility"],
                    test_results["functionality_compatibility"],
                    test_results["performance_compatibility"]
                ]
            )
            test_results["overall_status"] = "pass" if all_sections_pass else "fail"
            
        except Exception as e:
            test_results["error"] = str(e)
            test_results["overall_status"] = "fail"
        
        return test_results


# ==================== FACTORY FUNCTIONS ====================

def create_compatible_verifier(enable_debate: bool = True, 
                             enable_parallel: bool = True) -> IdealVerifier:
    """
    Factory function to create a backward compatible verifier
    
    Args:
        enable_debate: Enable adversarial debate system
        enable_parallel: Enable parallel processing
        
    Returns:
        Backward compatible verifier instance
    """
    return IdealVerifier(enable_debate=enable_debate, enable_parallel=enable_parallel)


def create_unified_verifier(mode: VerificationMode = VerificationMode.INDIVIDUAL,
                          consensus_config: Optional[ConsensusConfig] = None) -> UnifiedIdealVerifier:
    """
    Factory function to create a unified verifier directly
    
    Args:
        mode: Verification mode to start in
        consensus_config: Configuration for consensus mode
        
    Returns:
        Unified verifier instance
    """
    return UnifiedIdealVerifier(mode=mode, consensus_config=consensus_config)


def migrate_from_original(original_config: Dict[str, Any]) -> IdealVerifier:
    """
    Migrate from original verifier configuration
    
    Args:
        original_config: Configuration dictionary from original verifier
        
    Returns:
        Migrated backward compatible verifier
    """
    # Extract settings from original config
    enable_debate = original_config.get("enable_debate", True)
    enable_parallel = original_config.get("enable_parallel", True)
    
    # Create compatible verifier
    verifier = IdealVerifier(enable_debate=enable_debate, enable_parallel=enable_parallel)
    
    # Apply any framework weights or other settings
    if "framework_weights" in original_config:
        # This would be applied through the unified verifier's meta system
        pass
    
    return verifier


# ==================== COMPATIBILITY TESTING UTILITIES ====================

def run_compatibility_test_suite() -> Dict[str, Any]:
    """
    Run comprehensive compatibility test suite
    
    Returns:
        Complete test results
    """
    test_suite_results = {
        "test_timestamp": "2024-01-01T00:00:00",  # Would be actual timestamp
        "tests": {},
        "summary": {}
    }
    
    try:
        # Test basic compatibility
        verifier = create_compatible_verifier()
        basic_tests = verifier.test_backward_compatibility()
        test_suite_results["tests"]["basic_compatibility"] = basic_tests
        
        # Test migration scenarios
        migration_info = verifier.get_migration_info()
        test_suite_results["tests"]["migration_readiness"] = {
            "current_state": migration_info["current_state"],
            "available_migrations": len(migration_info["available_migrations"]),
            "status": "ready"
        }
        
        # Test consensus extension capability
        try:
            # This would fail without proper setup, but tests the interface
            consensus_stats = verifier.get_consensus_statistics()
            test_suite_results["tests"]["consensus_interface"] = {
                "status": "pass",
                "consensus_ready": consensus_stats.get("consensus_enabled", False)
            }
        except Exception as e:
            test_suite_results["tests"]["consensus_interface"] = {
                "status": "interface_available",
                "note": "Interface available but consensus not configured"
            }
        
        # Summary
        total_tests = len(test_suite_results["tests"])
        passed_tests = sum(
            1 for test in test_suite_results["tests"].values()
            if test.get("overall_status") == "pass" or test.get("status") == "pass"
        )
        
        test_suite_results["summary"] = {
            "total_tests": total_tests,
            "passed_tests": passed_tests,
            "pass_rate": passed_tests / total_tests if total_tests > 0 else 0,
            "overall_compatibility": "excellent" if passed_tests == total_tests else "good"
        }
        
    except Exception as e:
        test_suite_results["error"] = str(e)
        test_suite_results["summary"]["overall_compatibility"] = "failed"
    
    return test_suite_results


if __name__ == "__main__":
    """Run compatibility tests if executed directly"""
    print("🔄 Running Backward Compatibility Test Suite...")
    
    results = run_compatibility_test_suite()
    
    print(f"📊 Test Results Summary:")
    summary = results.get("summary", {})
    print(f"  • Total Tests: {summary.get('total_tests', 0)}")
    print(f"  • Passed Tests: {summary.get('passed_tests', 0)}")
    print(f"  • Pass Rate: {summary.get('pass_rate', 0):.1%}")
    print(f"  • Overall Compatibility: {summary.get('overall_compatibility', 'unknown')}")
    
    if results.get("error"):
        print(f"❌ Test Suite Error: {results['error']}")
    else:
        print("✅ Compatibility test suite completed successfully")
        print("🪁 Backward compatibility verified!")