#!/usr/bin/env python3
"""
BrahminyKite Unified Framework - Comprehensive Example

Demonstrates the complete unified verification system including:
- Individual verification mode
- Consensus verification mode
- Mode switching and bridging
- Cross-mode analysis
- Learning and adaptation
"""

import asyncio
import logging
from typing import Dict, Any

# Import the unified framework
from verifier.unified_core import UnifiedIdealVerifier
from verifier.frameworks import Claim, Domain
from verifier.consensus_types import (
    ConsensusProposal, ProposalType, ConsensusConfig, 
    VerificationMode, NodeRole, NodeContext
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def create_test_claims() -> list[Claim]:
    """Create a diverse set of test claims"""
    return [
        # Empirical claim
        Claim(
            content="Machine learning models trained on diverse datasets show 15% better performance on cross-cultural tasks",
            domain=Domain.EMPIRICAL,
            context={
                "scientific_domain": "machine_learning",
                "peer_reviewed": True,
                "study_size": "large_scale",
                "replication_studies": 3
            },
            source_metadata={
                "sources": ["arxiv_papers", "conference_proceedings"],
                "citation_count": 45,
                "institutional_affiliation": "multiple_universities"
            }
        ),
        
        # Ethical claim
        Claim(
            content="AI systems should always provide explanations for decisions that significantly impact individuals",
            domain=Domain.ETHICAL,
            context={
                "ethical_framework": "transparency_principle",
                "stakeholder_groups": ["users", "developers", "regulators"],
                "implementation_complexity": "high"
            },
            source_metadata={
                "sources": ["ethics_committees", "policy_papers"],
                "consensus_level": "emerging"
            }
        ),
        
        # Social claim
        Claim(
            content="Remote work technologies have fundamentally changed team collaboration patterns in knowledge work",
            domain=Domain.SOCIAL,
            context={
                "research_domain": "organizational_psychology",
                "temporal_scope": "2020-2024",
                "cultural_context": "global",
                "evidence_types": ["surveys", "behavioral_data", "interviews"]
            },
            source_metadata={
                "sources": ["workplace_studies", "hr_analytics"],
                "sample_size": "10000+"
            }
        )
    ]


def create_test_proposals() -> list[ConsensusProposal]:
    """Create test consensus proposals"""
    return [
        # Model update proposal
        ConsensusProposal(
            proposal_type=ProposalType.MODEL_UPDATE,
            content={
                "update_description": "Implement bias detection layer in recommendation system",
                "justification": "Improve fairness in content recommendations",
                "performance_metrics": {
                    "accuracy": "maintained",
                    "fairness": "improved_by_20_percent",
                    "latency": "increased_by_5ms"
                },
                "affected_components": ["recommendation_engine", "bias_detector"],
                "rollback_plan": "automatic_rollback_on_performance_degradation"
            },
            domain=Domain.ETHICAL,
            priority_level=2,
            timeout=60,
            required_verifiers=["empirical", "power_dynamics", "utility"]
        ),
        
        # Policy change proposal
        ConsensusProposal(
            proposal_type=ProposalType.POLICY_CHANGE,
            content={
                "policy_statement": "All AI model updates must undergo multi-stakeholder review",
                "rationale": "Ensure diverse perspectives in AI development",
                "scope": "organization_wide",
                "implementation_timeline": "3_months",
                "stakeholders": ["developers", "users", "ethics_committee", "legal_team"],
                "compliance_requirements": ["audit_trail", "documentation", "approval_tracking"]
            },
            domain=Domain.ETHICAL,
            priority_level=1,  # Critical
            timeout=120,
            required_verifiers=["consistency", "power_dynamics", "utility", "contextual"]
        )
    ]


async def demonstrate_individual_verification():
    """Demonstrate individual verification capabilities"""
    print("\n" + "="*60)
    print("🔍 INDIVIDUAL VERIFICATION DEMONSTRATION")
    print("="*60)
    
    # Initialize unified verifier in individual mode
    verifier = UnifiedIdealVerifier(
        mode=VerificationMode.INDIVIDUAL,
        enable_unified_components=True
    )
    
    claims = create_test_claims()
    
    for i, claim in enumerate(claims, 1):
        print(f"\n📝 Claim {i}: {claim.content}")
        print(f"🏷️  Domain: {claim.domain.value}")
        print("-" * 50)
        
        # Verify the claim
        result = verifier.verify(claim)
        
        if result.get("error"):
            print(f"❌ Error: {result['message']}")
            continue
        
        # Display results
        print(f"📊 Final Score: {result['final_score']:.3f}")
        print(f"🎯 Framework: {result['dominant_framework']}")
        print(f"📏 Confidence: [{result['confidence_interval'][0]:.3f}, {result['confidence_interval'][1]:.3f}]")
        
        # Component breakdown
        print("\n🔧 Component Results:")
        for comp_name, comp_result in result['component_results'].items():
            print(f"  • {comp_name.title()}: {comp_result['score']:.3f} ({comp_result['framework']})")
        
        # Debate results
        if result.get('debate_result'):
            debate = result['debate_result']
            print(f"\n⚔️  Debate: {len(debate['session']['rounds'])} rounds, consensus: {debate['final_consensus']:.3f}")
        
        print(f"\n⏱️  Processing Time: {result['metadata']['processing_time']:.3f}s")


async def demonstrate_consensus_verification():
    """Demonstrate consensus verification capabilities"""
    print("\n" + "="*60)
    print("🤝 CONSENSUS VERIFICATION DEMONSTRATION")
    print("="*60)
    
    # Configure consensus with multiple nodes
    consensus_config = ConsensusConfig(
        node_id="demo_node_1",
        peer_nodes=["demo_node_2", "demo_node_3", "demo_node_4"],
        required_quorum=0.75,
        consensus_timeout=30,
        enable_debate=True,
        max_debate_rounds=3
    )
    
    # Initialize unified verifier in consensus mode
    verifier = UnifiedIdealVerifier(
        mode=VerificationMode.CONSENSUS,
        consensus_config=consensus_config,
        enable_unified_components=True
    )
    
    proposals = create_test_proposals()
    
    for i, proposal in enumerate(proposals, 1):
        print(f"\n📋 Proposal {i}: {proposal.proposal_type.value}")
        print(f"📝 Content: {proposal.content.get('update_description', proposal.content.get('policy_statement', 'N/A'))}")
        print(f"⏰ Timeout: {proposal.timeout}s, Priority: {proposal.priority_level}")
        print("-" * 50)
        
        try:
            # Propose consensus
            result = await verifier.propose_consensus(proposal)
            
            if result.get("success"):
                print(f"✅ Consensus Result: {result.get('final_decision', 'UNKNOWN')}")
                print(f"⏱️  Consensus Time: {result.get('consensus_time', 0.0):.3f}s")
                print(f"👥 Participating Nodes: {result.get('participating_nodes', 0)}")
                
                # Verifier results
                verifier_results = result.get('verifier_results', {})
                if verifier_results:
                    print("\n🔧 Verifier Consensus:")
                    for verifier_name, verifier_data in verifier_results.items():
                        approval_rate = verifier_data.get('approval_rate', 0.0)
                        confidence = verifier_data.get('confidence', 0.0)
                        print(f"  • {verifier_name.title()}: {approval_rate:.1%} approval, {confidence:.3f} confidence")
                
                # Network health
                network_health = result.get('network_health', {})
                if network_health:
                    print(f"\n🌐 Network: {network_health.get('active_peers', 0)}/{network_health.get('total_peers', 0)} active")
            
            else:
                print(f"❌ Consensus Failed: {result.get('failure_reason', 'Unknown')}")
                if result.get('failure_stage'):
                    print(f"📍 Failed at: {result['failure_stage']}")
        
        except Exception as e:
            print(f"💥 Exception during consensus: {e}")


async def demonstrate_mode_switching():
    """Demonstrate switching between verification modes"""
    print("\n" + "="*60)
    print("🔄 MODE SWITCHING DEMONSTRATION")
    print("="*60)
    
    # Start with individual mode
    verifier = UnifiedIdealVerifier(mode=VerificationMode.INDIVIDUAL)
    print(f"🎯 Initial Mode: {verifier.mode.value}")
    
    # Test individual verification
    test_claim = create_test_claims()[0]
    print(f"\n📝 Testing individual verification...")
    individual_result = verifier.verify(test_claim)
    print(f"✅ Individual Score: {individual_result.get('final_score', 0.0):.3f}")
    
    # Switch to consensus mode
    print(f"\n🔄 Switching to consensus mode...")
    consensus_config = ConsensusConfig(
        node_id="switching_demo_node",
        peer_nodes=["peer_1", "peer_2"],
        required_quorum=0.6
    )
    
    verifier.switch_mode(
        VerificationMode.CONSENSUS,
        {"consensus_config": consensus_config.__dict__}
    )
    print(f"🎯 New Mode: {verifier.mode.value}")
    
    # Test consensus verification
    test_proposal = create_test_proposals()[0]
    print(f"\n📋 Testing consensus verification...")
    consensus_result = await verifier.propose_consensus(test_proposal)
    print(f"✅ Consensus Result: {consensus_result.get('final_decision', 'UNKNOWN')}")
    
    # Switch to hybrid mode
    print(f"\n🔄 Switching to hybrid mode...")
    verifier.switch_mode(VerificationMode.HYBRID)
    print(f"🎯 New Mode: {verifier.mode.value}")
    print("💡 Hybrid mode enables both individual and consensus verification")


async def demonstrate_cross_mode_analysis():
    """Demonstrate cross-mode analysis capabilities"""
    print("\n" + "="*60)
    print("🔀 CROSS-MODE ANALYSIS DEMONSTRATION")
    print("="*60)
    
    # Initialize in hybrid mode
    verifier = UnifiedIdealVerifier(
        mode=VerificationMode.HYBRID,
        consensus_config=ConsensusConfig(
            node_id="cross_mode_node",
            peer_nodes=["analysis_peer_1", "analysis_peer_2"]
        )
    )
    
    # Test with a claim
    test_claim = create_test_claims()[1]  # Ethical claim
    print(f"📝 Analyzing Claim: {test_claim.content[:80]}...")
    
    cross_analysis = verifier.cross_mode_analysis(test_claim)
    
    print(f"🔍 Analysis ID: {cross_analysis.get('analysis_id', 'N/A')}")
    print(f"📊 Original Type: {cross_analysis.get('original_item_type', 'N/A')}")
    
    if cross_analysis.get('individual_analysis'):
        print(f"👤 Individual Score: {cross_analysis['individual_analysis'].get('final_score', 0.0):.3f}")
    
    if cross_analysis.get('consensus_analysis'):
        print(f"🤝 Consensus Available: {type(cross_analysis['consensus_analysis']) == dict}")
    
    # Comparison results
    comparison = cross_analysis.get('comparison', {})
    if comparison:
        print(f"📈 Agreement Level: {comparison.get('agreement_level', 0.0):.3f}")
        discrepancies = comparison.get('discrepancies', [])
        if discrepancies:
            print(f"⚠️  Discrepancies: {', '.join(discrepancies)}")
    
    # Recommendations
    recommendations = cross_analysis.get('recommendations', [])
    if recommendations:
        print("\n💡 Recommendations:")
        for rec in recommendations:
            print(f"  • {rec}")


async def demonstrate_learning_adaptation():
    """Demonstrate learning and adaptation capabilities"""
    print("\n" + "="*60)
    print("🧠 LEARNING & ADAPTATION DEMONSTRATION")
    print("="*60)
    
    verifier = UnifiedIdealVerifier(mode=VerificationMode.HYBRID)
    
    # Get initial statistics
    initial_stats = verifier.get_verification_statistics()
    print("📊 Initial System State:")
    print(f"  • Mode: {initial_stats['system_info']['current_mode']}")
    print(f"  • Components: {len(initial_stats.get('component_statistics', {}))}")
    
    # Perform some verifications
    test_claim = create_test_claims()[0]
    verification_result = verifier.verify(test_claim)
    
    print(f"\n🔍 Performed verification with score: {verification_result.get('final_score', 0.0):.3f}")
    
    # Simulate human feedback
    human_feedback = {
        "framework_preference": "positivist",
        "weight_adjustment": 1.15,  # Increase positivist weight
        "quality_rating": 0.85,
        "accuracy": 0.9,
        "philosophical_adjustments": {
            "positivist": 1.1,  # Slight increase
            "interpretivist": 0.95  # Slight decrease
        }
    }
    
    print(f"\n📝 Applying human feedback...")
    verifier.learn_from_feedback(test_claim, verification_result, human_feedback)
    
    # Get updated statistics
    updated_stats = verifier.get_verification_statistics()
    print(f"✅ Feedback applied successfully")
    
    # Show learning impact
    if updated_stats.get('meta_verifier'):
        print(f"🎯 Meta-verifier adapted framework weights")
    
    print(f"📈 Total verifications: {updated_stats['verification_counts']['total_individual']}")


async def demonstrate_configuration_management():
    """Demonstrate configuration export/import"""
    print("\n" + "="*60)
    print("⚙️  CONFIGURATION MANAGEMENT DEMONSTRATION")
    print("="*60)
    
    # Create configured verifier
    verifier = UnifiedIdealVerifier(
        mode=VerificationMode.HYBRID,
        consensus_config=ConsensusConfig(
            node_id="config_demo_node",
            peer_nodes=["config_peer_1", "config_peer_2"],
            required_quorum=0.8,
            enable_debate=True
        )
    )
    
    # Perform some operations to create state
    test_claim = create_test_claims()[0]
    verifier.verify(test_claim)
    
    # Export configuration
    print("📤 Exporting configuration...")
    config = verifier.export_configuration()
    
    print(f"✅ Exported configuration with {len(config)} top-level sections:")
    for section in config.keys():
        print(f"  • {section}")
    
    # Show some configuration details
    system_settings = config.get('system_settings', {})
    print(f"\n⚙️  System Settings:")
    print(f"  • Mode: {system_settings.get('mode', 'N/A')}")
    print(f"  • Parallel Processing: {system_settings.get('enable_parallel', False)}")
    print(f"  • Debate Enabled: {system_settings.get('enable_debate', False)}")
    
    # Simulate importing configuration
    print(f"\n📥 Configuration can be imported to restore system state")
    print(f"💾 Configuration saved for future use")


async def main():
    """Run the complete unified framework demonstration"""
    print("🪁 BrahminyKite Unified Verification Framework")
    print("=" * 60)
    print("Comprehensive demonstration of individual + consensus verification")
    
    try:
        # Individual verification
        await demonstrate_individual_verification()
        
        # Consensus verification  
        await demonstrate_consensus_verification()
        
        # Mode switching
        await demonstrate_mode_switching()
        
        # Cross-mode analysis
        await demonstrate_cross_mode_analysis()
        
        # Learning and adaptation
        await demonstrate_learning_adaptation()
        
        # Configuration management
        await demonstrate_configuration_management()
        
        # Final summary
        print("\n" + "="*60)
        print("🎉 UNIFIED FRAMEWORK DEMONSTRATION COMPLETE!")
        print("="*60)
        print("✅ Successfully demonstrated:")
        print("  • Individual claim verification")
        print("  • Distributed consensus verification")
        print("  • Seamless mode switching")
        print("  • Cross-mode analysis capabilities")
        print("  • Learning and adaptation")
        print("  • Configuration management")
        print("\n🪁 The BrahminyKite soars gracefully across all verification modes! ✨")
        
    except Exception as e:
        print(f"\n💥 Demonstration failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())