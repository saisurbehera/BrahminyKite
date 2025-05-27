"""
Repository pattern implementations for data access.

Provides a clean abstraction layer over database operations.
"""

from .base import BaseRepository, RepositoryError
from .consensus import ConsensusRepository
from .verification import VerificationRepository
from .framework import FrameworkRepository
from .power_dynamics import PowerDynamicsRepository
from .evolution import EvolutionRepository

__all__ = [
    "BaseRepository",
    "RepositoryError",
    "ConsensusRepository",
    "VerificationRepository",
    "FrameworkRepository",
    "PowerDynamicsRepository",
    "EvolutionRepository",
]