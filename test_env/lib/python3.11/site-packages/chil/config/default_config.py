"""Default configuration for Chil framework."""

from typing import Dict, Any
from dataclasses import dataclass
from ..framework.consensus_types import VerificationMode


@dataclass
class FrameworkConfig:
    """Configuration for individual verification frameworks."""
    
    empirical_weight: float = 1.0
    contextual_weight: float = 1.0
    consistency_weight: float = 1.0
    power_dynamics_weight: float = 1.0
    utility_weight: float = 1.0
    evolution_weight: float = 1.0
    
    confidence_threshold: float = 0.7
    enable_meta_verification: bool = True


@dataclass
class ConsensusConfig:
    """Configuration for consensus operations."""
    
    node_timeout: float = 30.0
    proposal_timeout: float = 60.0
    max_retries: int = 3
    quorum_size: int = 3
    
    enable_philosophical_validation: bool = True
    require_unanimous_meta: bool = False


@dataclass
class SystemConfig:
    """Overall system configuration."""
    
    default_mode: VerificationMode = VerificationMode.INDIVIDUAL
    enable_backward_compatibility: bool = True
    log_level: str = "INFO"
    
    framework: FrameworkConfig = FrameworkConfig()
    consensus: ConsensusConfig = ConsensusConfig()


# Default configuration instance
DEFAULT_CONFIG = SystemConfig()


def load_config(config_path: str = None) -> SystemConfig:
    """Load configuration from file or return default."""
    if config_path:
        # TODO: Implement config file loading
        pass
    return DEFAULT_CONFIG