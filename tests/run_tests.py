#!/usr/bin/env python3
"""
BrahminyKite Test Runner

Comprehensive test suite runner for the unified verification framework.
"""

import unittest
import sys
import os
import argparse
import time
from typing import List, Optional

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class BrahminyKiteTestRunner:
    """Custom test runner with enhanced reporting"""
    
    def __init__(self, verbosity: int = 2):
        self.verbosity = verbosity
        self.test_categories = {
            "unit": "tests.unit",
            "integration": "tests.integration", 
            "compatibility": "tests.compatibility",
            "performance": "tests.performance"
        }
    
    def discover_tests(self, category: Optional[str] = None) -> unittest.TestSuite:
        """Discover tests in specified category or all categories"""
        loader = unittest.TestLoader()
        
        if category:
            if category not in self.test_categories:
                raise ValueError(f"Unknown test category: {category}")
            
            suite = loader.discover(
                start_dir=os.path.join("tests", category),
                pattern="test_*.py"
            )
        else:
            # Discover all tests
            suite = unittest.TestSuite()
            for cat_name, cat_path in self.test_categories.items():
                try:
                    cat_suite = loader.discover(
                        start_dir=os.path.join("tests", cat_name),
                        pattern="test_*.py"
                    )
                    suite.addTest(cat_suite)
                except Exception as e:
                    print(f"Warning: Could not load {cat_name} tests: {e}")
        
        return suite
    
    def run_tests(self, category: Optional[str] = None, pattern: Optional[str] = None) -> bool:
        """Run tests and return success status"""
        print("🪁 BrahminyKite Test Suite")
        print("=" * 50)
        
        if category:
            print(f"Running {category} tests...")
        else:
            print("Running all tests...")
        
        # Discover tests
        suite = self.discover_tests(category)
        
        # Filter by pattern if provided
        if pattern:
            suite = self._filter_by_pattern(suite, pattern)
        
        # Run tests
        runner = unittest.TextTestRunner(
            verbosity=self.verbosity,
            stream=sys.stdout,
            buffer=True
        )
        
        start_time = time.time()
        result = runner.run(suite)
        end_time = time.time()
        
        # Print summary
        self._print_summary(result, end_time - start_time)
        
        return result.wasSuccessful()
    
    def _filter_by_pattern(self, suite: unittest.TestSuite, pattern: str) -> unittest.TestSuite:
        """Filter tests by name pattern"""
        filtered_suite = unittest.TestSuite()
        
        def filter_recursive(test_item):
            if isinstance(test_item, unittest.TestSuite):
                for sub_item in test_item:
                    filter_recursive(sub_item)
            elif isinstance(test_item, unittest.TestCase):
                if pattern.lower() in test_item._testMethodName.lower():
                    filtered_suite.addTest(test_item)
        
        filter_recursive(suite)
        return filtered_suite
    
    def _print_summary(self, result: unittest.TestResult, duration: float):
        """Print test summary"""
        print("\n" + "=" * 50)
        print("🎯 Test Summary")
        print("=" * 50)
        print(f"Tests run: {result.testsRun}")
        print(f"Failures: {len(result.failures)}")
        print(f"Errors: {len(result.errors)}")
        print(f"Skipped: {len(result.skipped) if hasattr(result, 'skipped') else 0}")
        print(f"Duration: {duration:.2f}s")
        
        if result.wasSuccessful():
            print("✅ All tests passed!")
        else:
            print("❌ Some tests failed")
            
            if result.failures:
                print("\n🔥 Failures:")
                for test, traceback in result.failures:
                    error_line = traceback.split('\n')[-2] if traceback else 'Unknown'
                    print(f"  • {test}: {error_line}")
            
            if result.errors:
                print("\n💥 Errors:")
                for test, traceback in result.errors:
                    error_line = traceback.split('\n')[-2] if traceback else 'Unknown'
                    print(f"  • {test}: {error_line}")


def run_specific_tests():
    """Run specific test scenarios"""
    print("🧪 Running Specific Test Scenarios")
    print("-" * 40)
    
    # Test basic functionality
    print("1. Testing basic framework functionality...")
    try:
        from verifier import quick_verify, Domain
        result = quick_verify("Test claim", Domain.EMPIRICAL)
        print(f"   ✅ Quick verify: {result.get('final_score', 0):.3f}")
    except Exception as e:
        print(f"   ❌ Quick verify failed: {e}")
    
    # Test backward compatibility
    print("2. Testing backward compatibility...")
    try:
        from verifier.compatibility import run_compatibility_test_suite
        compat_results = run_compatibility_test_suite()
        overall = compat_results.get("summary", {}).get("overall_compatibility", "unknown")
        print(f"   ✅ Compatibility: {overall}")
    except Exception as e:
        print(f"   ❌ Compatibility test failed: {e}")
    
    # Test unified framework
    print("3. Testing unified framework...")
    try:
        from verifier import UnifiedIdealVerifier, VerificationMode
        verifier = UnifiedIdealVerifier(mode=VerificationMode.INDIVIDUAL)
        print(f"   ✅ Unified verifier initialized in {verifier.mode.value} mode")
    except Exception as e:
        print(f"   ❌ Unified framework failed: {e}")


def main():
    """Main test runner entry point"""
    parser = argparse.ArgumentParser(description="BrahminyKite Test Runner")
    parser.add_argument(
        "--category", "-c",
        choices=["unit", "integration", "compatibility", "performance"],
        help="Run tests in specific category"
    )
    parser.add_argument(
        "--pattern", "-p",
        help="Filter tests by name pattern"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="count",
        default=1,
        help="Increase verbosity"
    )
    parser.add_argument(
        "--quick", "-q",
        action="store_true",
        help="Run quick functionality tests only"
    )
    
    args = parser.parse_args()
    
    if args.quick:
        run_specific_tests()
        return
    
    # Run full test suite
    runner = BrahminyKiteTestRunner(verbosity=args.verbose)
    success = runner.run_tests(category=args.category, pattern=args.pattern)
    
    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()