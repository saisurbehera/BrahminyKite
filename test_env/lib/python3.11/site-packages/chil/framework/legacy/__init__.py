"""
Framework Module

Core philosophical frameworks and domain definitions.
"""

from .frameworks import (
    VerificationFramework, Domain, Claim, VerificationResult,
    get_framework_descriptions, get_domain_characteristics
)

__all__ = [
    "VerificationFramework",
    "Domain", 
    "Claim",
    "VerificationResult",
    "get_framework_descriptions",
    "get_domain_characteristics"
]