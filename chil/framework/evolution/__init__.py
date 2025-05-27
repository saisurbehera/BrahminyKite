"""
Evolution Verification Framework

Handles temporal robustness and adaptive learning:
- Temporal stability analysis
- Learning and adaptation mechanisms
- Future robustness prediction
- Evidence evolution tracking
- Framework weight optimization
- Historical consistency validation
"""

from .evolutionary import EvolutionaryVerifier
from .evolution_real import RealEvolutionFramework

__all__ = ["EvolutionaryVerifier", "RealEvolutionFramework"]