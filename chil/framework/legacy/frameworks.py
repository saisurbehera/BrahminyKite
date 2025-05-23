"""
Philosophical frameworks and domain definitions for verification
"""

from enum import Enum
from typing import Dict, List
from dataclasses import dataclass, field


class VerificationFramework(Enum):
    """Philosophical frameworks for verification"""
    POSITIVIST = "positivist"
    INTERPRETIVIST = "interpretivist" 
    PRAGMATIST = "pragmatist"
    CORRESPONDENCE = "correspondence"
    COHERENCE = "coherence"
    CONSTRUCTIVIST = "constructivist"


class Domain(Enum):
    """Domains of knowledge and verification"""
    EMPIRICAL = "empirical"
    AESTHETIC = "aesthetic"
    ETHICAL = "ethical"
    LOGICAL = "logical"
    SOCIAL = "social"
    CREATIVE = "creative"


@dataclass
class VerificationResult:
    """Result of a verification process"""
    score: float  # 0-1 confidence score
    framework: VerificationFramework
    component: str
    evidence: Dict[str, any]
    confidence_interval: tuple[float, float]
    metadata: Dict[str, any] = field(default_factory=dict)
    

@dataclass
class Claim:
    """A claim to be verified"""
    content: str
    domain: Domain
    context: Dict[str, any] = field(default_factory=dict)
    source_metadata: Dict[str, any] = field(default_factory=dict)


def get_framework_descriptions() -> Dict[VerificationFramework, str]:
    """Get detailed descriptions of each philosophical framework"""
    return {
        VerificationFramework.POSITIVIST: 
            "Verification requires empirical, objective evidence. Emphasizes scientific method and measurable data.",
        
        VerificationFramework.INTERPRETIVIST: 
            "Verification involves understanding context, intentions, and cultural frameworks. Focuses on meaning and interpretation.",
        
        VerificationFramework.PRAGMATIST: 
            "Verification is about practical consequences rather than absolute truth. Judges claims by their utility and effectiveness.",
        
        VerificationFramework.CORRESPONDENCE: 
            "Truth aligns with objective reality. Claims are verified by how well they match external facts.",
        
        VerificationFramework.COHERENCE: 
            "Truth depends on internal consistency within a system of beliefs. Verified by logical coherence and systematic fit.",
        
        VerificationFramework.CONSTRUCTIVIST: 
            "Truth and verification are shaped by power structures and social construction. Considers institutional bias and authority."
    }


def get_domain_characteristics() -> Dict[Domain, Dict[str, any]]:
    """Get characteristics and preferred frameworks for each domain"""
    return {
        Domain.EMPIRICAL: {
            "description": "Scientific facts, measurable phenomena, objective data",
            "preferred_frameworks": [VerificationFramework.POSITIVIST, VerificationFramework.CORRESPONDENCE],
            "verification_methods": ["sensor_data", "database_queries", "mathematical_proofs"],
            "examples": ["Earth orbits Sun", "Water boils at 100°C", "DNA structure"]
        },
        
        Domain.AESTHETIC: {
            "description": "Art, beauty, creative expression, subjective appreciation",
            "preferred_frameworks": [VerificationFramework.INTERPRETIVIST, VerificationFramework.CONSTRUCTIVIST],
            "verification_methods": ["cultural_analysis", "aesthetic_theory", "consensus_building"],
            "examples": ["Poem captures longing", "Painting is beautiful", "Music evokes emotion"]
        },
        
        Domain.ETHICAL: {
            "description": "Moral claims, values, rights and wrongs, justice",
            "preferred_frameworks": [VerificationFramework.COHERENCE, VerificationFramework.PRAGMATIST],
            "verification_methods": ["ethical_consistency", "outcome_analysis", "moral_theory"],
            "examples": ["Universal healthcare is just", "Lying is wrong", "Rights are universal"]
        },
        
        Domain.LOGICAL: {
            "description": "Mathematical proofs, logical arguments, formal reasoning",
            "preferred_frameworks": [VerificationFramework.COHERENCE, VerificationFramework.POSITIVIST],
            "verification_methods": ["proof_checking", "logical_validation", "consistency_analysis"],
            "examples": ["2+2=4", "All men are mortal", "If P then Q"]
        },
        
        Domain.SOCIAL: {
            "description": "Social phenomena, cultural practices, human behavior",
            "preferred_frameworks": [VerificationFramework.INTERPRETIVIST, VerificationFramework.CONSTRUCTIVIST],
            "verification_methods": ["ethnographic_analysis", "statistical_studies", "social_theory"],
            "examples": ["Culture influences behavior", "Democracy reduces conflict", "Language shapes thought"]
        },
        
        Domain.CREATIVE: {
            "description": "Creative works, innovation, artistic merit, originality",
            "preferred_frameworks": [VerificationFramework.INTERPRETIVIST, VerificationFramework.PRAGMATIST],
            "verification_methods": ["peer_review", "impact_assessment", "innovation_metrics"],
            "examples": ["Design is innovative", "Story is compelling", "Solution is creative"]
        }
    }