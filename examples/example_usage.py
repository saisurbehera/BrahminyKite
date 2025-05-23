#!/usr/bin/env python3
"""
BrahminyKite Ideal Verifier - Example Usage

Demonstrates comprehensive verification across different domains
using the modular philosophical verification system.
"""

import json
import logging
from verifier import IdealVerifier, Claim, Domain

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def main():
    """Demonstrate the ideal verifier with various claim types"""
    
    print("🪁 BrahminyKite Ideal Verifier - Example Usage")
    print("=" * 60)
    
    # Initialize the verifier
    print("\n📋 Initializing Ideal Verifier...")
    verifier = IdealVerifier(enable_debate=True, enable_parallel=True)
    
    # Define test claims across different domains
    test_claims = [
        Claim(
            content="The Earth orbits around the Sun in an elliptical path",
            domain=Domain.EMPIRICAL,
            context={
                "scientific_domain": "astronomy",
                "historical_data": True,
                "peer_reviewed": True
            },
            source_metadata={
                "sources": ["NASA", "ESA", "peer_reviewed_journals"],
                "citation_count": 1000,
                "institutional_affiliation": "Space agencies"
            }
        ),
        
        Claim(
            content="This poem beautifully captures the essence of longing and melancholy",
            domain=Domain.AESTHETIC,
            context={
                "artistic_medium": "poetry",
                "cultural_context": "romantic_literature",
                "subjective_assessment": True
            },
            source_metadata={
                "sources": ["literary_critics", "poetry_journals"],
                "peer_reviewed": False
            }
        ),
        
        Claim(
            content="Universal healthcare improves societal wellbeing and reduces inequality",
            domain=Domain.ETHICAL,
            context={
                "policy_domain": "healthcare",
                "ethical_framework": "utilitarianism",
                "implementation_plan": True
            },
            source_metadata={
                "sources": ["WHO", "policy_research", "health_economists"],
                "funding_source": "public_health_organizations"
            }
        ),
        
        Claim(
            content="If all ravens are black, then finding a non-black raven would falsify the statement",
            domain=Domain.LOGICAL,
            context={
                "logical_form": "universal_quantification",
                "philosophical_context": "falsification_theory"
            },
            source_metadata={
                "sources": ["logic_textbooks", "philosophy_papers"]
            }
        ),
        
        Claim(
            content="Social media algorithms significantly influence political opinions and voting behavior",
            domain=Domain.SOCIAL,
            context={
                "research_domain": "digital_sociology",
                "temporal_scope": "2016-2024",
                "cross_cultural": True
            },
            source_metadata={
                "sources": ["academic_studies", "data_analysis"],
                "peer_reviewed": True,
                "conflicts_of_interest": []
            }
        ),
        
        Claim(
            content="This innovative design solution elegantly addresses user needs while maintaining aesthetic appeal",
            domain=Domain.CREATIVE,
            context={
                "design_domain": "product_design",
                "user_testing": True,
                "innovation_metrics": True
            },
            source_metadata={
                "sources": ["design_experts", "user_feedback"],
                "peer_reviewed": False
            }
        )
    ]
    
    # Verify each claim
    results = []
    for i, claim in enumerate(test_claims, 1):
        print(f"\n🔍 Verification {i}/{len(test_claims)}")
        print(f"📝 Claim: {claim.content}")
        print(f"🏷️  Domain: {claim.domain.value}")
        print("-" * 50)
        
        # Perform verification
        result = verifier.verify(claim)
        results.append(result)
        
        # Display results
        if result.get("error"):
            print(f"❌ Error: {result['message']}")
            continue
        
        print(f"📊 Final Score: {result['final_score']:.3f}")
        print(f"🎯 Dominant Framework: {result['dominant_framework']}")
        print(f"📏 Confidence: [{result['confidence_interval'][0]:.3f}, {result['confidence_interval'][1]:.3f}]")
        
        # Show component breakdown
        print("\n🔧 Component Scores:")
        for comp_name, comp_result in result['component_results'].items():
            print(f"  • {comp_name.title()}: {comp_result['score']:.3f} ({comp_result['framework']})")
        
        # Show debate results if available
        if result.get('debate_result'):
            debate = result['debate_result']
            print(f"\n⚔️  Debate Results:")
            print(f"  • Rounds: {len(debate['session']['rounds'])}")
            print(f"  • Consensus: {debate['final_consensus']:.3f}")
            print(f"  • Winner: {debate['winner']}")
        
        # Show explanation
        print(f"\n📖 Explanation:")
        explanation_lines = result['explanation'].split('\n')
        for line in explanation_lines[:10]:  # First 10 lines
            print(f"   {line}")
        if len(explanation_lines) > 10:
            print(f"   ... ({len(explanation_lines) - 10} more lines)")
        
        print()
    
    # Demonstrate learning from feedback
    print("\n🧠 Demonstrating Adaptive Learning")
    print("-" * 50)
    
    # Simulate human feedback for the first claim
    if results and not results[0].get("error"):
        feedback = {
            "framework_preference": "positivist",
            "weight_adjustment": 1.1,
            "quality_rating": 0.9,
            "domain_adjustments": {
                "empirical": {
                    "positivist": 1.05,
                    "correspondence": 1.02
                }
            },
            "accuracy": 0.85,
            "predicted_confidence": results[0]['final_score']
        }
        
        print("📝 Providing feedback for empirical claim...")
        verifier.learn_from_feedback(test_claims[0], results[0], feedback)
        print("✅ Feedback integrated successfully")
    
    # Display system statistics
    print("\n📈 System Performance Statistics")
    print("-" * 50)
    
    stats = verifier.get_verification_statistics()
    
    print(f"📊 Total Verifications: {stats['verification_history']['total_verifications']}")
    print(f"📊 Average Score: {stats['verification_history']['recent_average_score']:.3f}")
    
    print("\n🏷️  Domain Distribution:")
    for domain, count in stats['verification_history']['domain_distribution'].items():
        print(f"  • {domain}: {count}")
    
    print("\n🎯 Framework Usage:")
    for framework, count in stats['verification_history']['framework_usage'].items():
        print(f"  • {framework}: {count}")
    
    # Component statistics
    print("\n🔧 Component Performance:")
    for comp_name, comp_stats in stats['component_statistics'].items():
        if isinstance(comp_stats, dict) and 'applicable_frameworks' in comp_stats:
            frameworks = ', '.join(comp_stats['applicable_frameworks'])
            print(f"  • {comp_name.title()}: {frameworks}")
    
    # Meta-verifier analysis
    print("\n🎯 Meta-Verifier Analysis:")
    meta_stats = stats['meta_verifier_stats']
    print(f"  • Resolution History: {meta_stats['resolution_history_size']} conflicts resolved")
    print(f"  • Learning Rate: {meta_stats['learning_rate']}")
    
    # Debate system performance
    if stats['debate_system_stats'].get('total_debates', 0) > 0:
        print("\n⚔️  Debate System Performance:")
        debate_stats = stats['debate_system_stats']
        print(f"  • Total Debates: {debate_stats['total_debates']}")
        print(f"  • Average Consensus: {debate_stats['average_consensus']:.3f}")
        print(f"  • Most Common Winner: {debate_stats['most_common_winner']}")
    
    # Export configuration example
    print("\n💾 Configuration Export/Import")
    print("-" * 50)
    
    config = verifier.export_configuration()
    print("✅ Configuration exported")
    print(f"📦 Framework weights: {len(config['framework_weights'])} frameworks")
    print(f"📦 Domain preferences: {len(config['domain_preferences'])} domains")
    
    # Save configuration to file
    with open('verifier_config.json', 'w') as f:
        json.dump(config, f, indent=2)
    print("💾 Configuration saved to 'verifier_config.json'")
    
    # Summary
    print("\n🎉 Verification Demo Complete!")
    print("-" * 50)
    print(f"✅ Verified {len([r for r in results if not r.get('error')])} claims successfully")
    print(f"📊 Average confidence: {sum(r['final_score'] for r in results if not r.get('error')) / max(len([r for r in results if not r.get('error')]), 1):.3f}")
    print("\n📚 Key Features Demonstrated:")
    print("  • Multi-framework philosophical integration")
    print("  • Domain-appropriate verification strategies") 
    print("  • Adversarial debate between frameworks")
    print("  • Meta-verification conflict resolution")
    print("  • Adaptive learning from feedback")
    print("  • Comprehensive uncertainty quantification")
    print("  • Transparent explanation generation")
    
    print(f"\n🪁 BrahminyKite flies gracefully across the verification sky! ✨")


if __name__ == "__main__":
    main()