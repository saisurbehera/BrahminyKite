"""
Abstract base class for verification components
"""

from abc import ABC, abstractmethod
from typing import List
from ..frameworks import VerificationFramework, VerificationResult, Claim


class VerificationComponent(ABC):
    """Abstract base class for verification components"""
    
    @abstractmethod
    def verify(self, claim: Claim) -> VerificationResult:
        """
        Verify a claim and return a verification result
        
        Args:
            claim: The claim to verify
            
        Returns:
            VerificationResult with score, evidence, and metadata
        """
        pass
    
    @abstractmethod
    def get_applicable_frameworks(self) -> List[VerificationFramework]:
        """
        Get the philosophical frameworks this component implements
        
        Returns:
            List of applicable verification frameworks
        """
        pass
    
    def get_component_name(self) -> str:
        """Get the name of this component"""
        return self.__class__.__name__.replace("Verifier", "").lower()
    
    def validate_claim(self, claim: Claim) -> bool:
        """
        Validate that a claim is suitable for this component
        
        Args:
            claim: The claim to validate
            
        Returns:
            True if claim can be processed by this component
        """
        return True  # Default: accept all claims
    
    def get_confidence_bounds(self, base_score: float, uncertainty: float = 0.1) -> tuple[float, float]:
        """
        Calculate confidence interval for a verification score
        
        Args:
            base_score: The base verification score
            uncertainty: The uncertainty level (default 0.1)
            
        Returns:
            Tuple of (lower_bound, upper_bound)
        """
        lower = max(0.0, base_score - uncertainty)
        upper = min(1.0, base_score + uncertainty)
        return (lower, upper)