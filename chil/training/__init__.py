"""
GRPO Training Infrastructure for BrahminyKite.

Implements Group Relative Policy Optimization for training LLMs
with philosophical framework verification.
"""

from .grpo import GRPOOptimizer, GRPOConfig
from .reward_model import MultiFrameworkRewardModel
from .mini_llms import MiniLLMRegistry
from .trainer import BrahminyKiteTrainer

__all__ = [
    'GRPOOptimizer',
    'GRPOConfig', 
    'MultiFrameworkRewardModel',
    'MiniLLMRegistry',
    'BrahminyKiteTrainer'
]