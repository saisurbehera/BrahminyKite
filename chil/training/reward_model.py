"""
Multi-Framework Reward Model for GRPO Training.

Aggregates verification results from all frameworks into a unified
reward signal for training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum

from ..tools.clients import (
    EmpiricalToolsClient,
    # ContextualToolsClient,
    # ConsistencyToolsClient,
    # PowerDynamicsToolsClient,
    # UtilityToolsClient,
    # EvolutionToolsClient
)
from .mini_llms import FrameworkType, MiniLLMRegistry


@dataclass
class RewardConfig:
    """Configuration for reward model."""
    
    # Framework weights (can be learned)
    framework_weights: Dict[str, float] = None
    weight_learning_rate: float = 0.001
    
    # Reward aggregation
    aggregation_method: str = "weighted_sum"  # "weighted_sum", "max", "min", "product"
    confidence_weighting: bool = True
    disagreement_penalty: float = 0.1
    
    # Normalization
    normalize_rewards: bool = True
    reward_scale: float = 1.0
    reward_clip: float = 5.0
    
    # Temporal aspects
    temporal_discount: float = 0.99
    history_weight: float = 0.1
    
    # Framework-specific parameters
    empirical_weight: float = 0.25
    contextual_weight: float = 0.15
    consistency_weight: float = 0.20
    power_dynamics_weight: float = 0.15
    utility_weight: float = 0.15
    evolution_weight: float = 0.10


class RewardAggregationMethod(Enum):
    """Methods for aggregating framework rewards."""
    WEIGHTED_SUM = "weighted_sum"
    WEIGHTED_PRODUCT = "weighted_product"
    MAX_CONSENSUS = "max_consensus"
    MIN_CONSENSUS = "min_consensus"
    MAJORITY_VOTE = "majority_vote"
    UNCERTAINTY_WEIGHTED = "uncertainty_weighted"


class MultiFrameworkRewardModel(nn.Module):
    """
    Reward model that aggregates verification results from multiple frameworks.
    """
    
    def __init__(self, config: RewardConfig, mini_llm_registry: MiniLLMRegistry, device: str = "cuda"):
        super().__init__()
        self.config = config
        self.registry = mini_llm_registry
        self.device = device
        
        # Initialize framework weights as learnable parameters
        if config.framework_weights is None:
            initial_weights = torch.tensor([
                config.empirical_weight,
                config.contextual_weight,
                config.consistency_weight,
                config.power_dynamics_weight,
                config.utility_weight,
                config.evolution_weight
            ], device=device)
        else:
            initial_weights = torch.tensor(
                list(config.framework_weights.values()), 
                device=device
            )
        
        self.framework_weights = nn.Parameter(initial_weights)
        
        # Reward combination networks
        self.reward_combiner = nn.Sequential(
            nn.Linear(6, 32),  # 6 frameworks
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )
        
        # Confidence combination network
        self.confidence_combiner = nn.Sequential(
            nn.Linear(6, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
        
        # Disagreement penalty network
        self.disagreement_estimator = nn.Sequential(
            nn.Linear(6, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
        
        # Tool clients for verification
        self.tool_clients = self._init_tool_clients()
        
        # History tracking
        self.reward_history = []
        self.confidence_history = []
    
    def _init_tool_clients(self) -> Dict[str, Any]:
        """Initialize tool clients for each framework."""
        clients = {}
        
        # Only initialize available clients
        try:
            clients['empirical'] = EmpiricalToolsClient()
        except Exception as e:
            print(f"Failed to initialize empirical client: {e}")
        
        # Add other clients as they become available
        # clients['contextual'] = ContextualToolsClient()
        # clients['consistency'] = ConsistencyToolsClient()
        # clients['power_dynamics'] = PowerDynamicsToolsClient()
        # clients['utility'] = UtilityToolsClient()
        # clients['evolution'] = EvolutionToolsClient()
        
        return clients
    
    async def compute_reward(self, claim: str, response: str, 
                           context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Compute reward for a claim-response pair using all frameworks.
        
        Args:
            claim: Original claim to verify
            response: Generated response to evaluate
            context: Additional context information
            
        Returns:
            Dictionary with reward components and metadata
        """
        
        # Get framework verifications
        framework_results = await self._get_framework_verifications(claim, response, context)
        
        # Extract scores and confidences
        scores = torch.tensor([
            framework_results.get('empirical', {}).get('score', 0.0),
            framework_results.get('contextual', {}).get('score', 0.0),
            framework_results.get('consistency', {}).get('score', 0.0),
            framework_results.get('power_dynamics', {}).get('score', 0.0),
            framework_results.get('utility', {}).get('score', 0.0),
            framework_results.get('evolution', {}).get('score', 0.0)
        ], device=self.device)
        
        confidences = torch.tensor([
            framework_results.get('empirical', {}).get('confidence', 0.0),
            framework_results.get('contextual', {}).get('confidence', 0.0),
            framework_results.get('consistency', {}).get('confidence', 0.0),
            framework_results.get('power_dynamics', {}).get('confidence', 0.0),
            framework_results.get('utility', {}).get('confidence', 0.0),
            framework_results.get('evolution', {}).get('confidence', 0.0)
        ], device=self.device)
        
        # Compute aggregated reward
        reward_components = self._aggregate_rewards(scores, confidences)
        
        # Add temporal and historical factors
        reward_components = self._add_temporal_factors(reward_components)
        
        # Normalize and clip
        final_reward = self._normalize_reward(reward_components['total_reward'])
        
        # Store in history
        self.reward_history.append(final_reward.item())
        self.confidence_history.append(reward_components['avg_confidence'].item())
        
        return {
            'total_reward': final_reward,
            'framework_scores': scores,
            'framework_confidences': confidences,
            'framework_weights': F.softmax(self.framework_weights, dim=0),
            'disagreement_penalty': reward_components['disagreement_penalty'],
            'confidence_bonus': reward_components['confidence_bonus'],
            'temporal_bonus': reward_components.get('temporal_bonus', 0.0),
            'framework_results': framework_results,
            'metadata': {
                'claim': claim,
                'response': response,
                'aggregation_method': self.config.aggregation_method
            }
        }
    
    async def _get_framework_verifications(self, claim: str, response: str, 
                                         context: Optional[Dict[str, Any]] = None) -> Dict[str, Dict[str, float]]:
        """Get verification results from all frameworks."""
        results = {}
        
        # Empirical framework
        if 'empirical' in self.tool_clients:
            try:
                empirical_result = await self._verify_empirical(claim, response)
                results['empirical'] = empirical_result
            except Exception as e:
                print(f"Empirical verification failed: {e}")
                results['empirical'] = {'score': 0.0, 'confidence': 0.0}
        
        # Contextual framework
        if 'contextual' in self.tool_clients:
            try:
                contextual_result = await self._verify_contextual(claim, response, context)
                results['contextual'] = contextual_result
            except Exception as e:
                print(f"Contextual verification failed: {e}")
                results['contextual'] = {'score': 0.0, 'confidence': 0.0}
        
        # Add other frameworks as clients become available
        for framework in ['consistency', 'power_dynamics', 'utility', 'evolution']:
            if framework not in results:
                results[framework] = {'score': 0.0, 'confidence': 0.0}
        
        return results
    
    async def _verify_empirical(self, claim: str, response: str) -> Dict[str, float]:
        """Verify using empirical framework."""
        client = self.tool_clients['empirical']
        
        # Use mini-LLM to process claim-response pair
        model = self.registry.get_model(FrameworkType.EMPIRICAL)
        tokenizer = self.registry.get_tokenizer(FrameworkType.EMPIRICAL)
        
        # Prepare input for verification
        input_text = f"Claim: {claim}\nResponse: {response}\nVerify:"
        inputs = tokenizer(
            input_text,
            return_tensors="pt",
            max_length=512,
            truncation=True,
            padding=True
        ).to(self.device)
        
        # Get model prediction
        with torch.no_grad():
            outputs = model(**inputs)
            
            if 'verification_score' in outputs:
                verification_score = outputs['verification_score'].item()
                confidence_score = outputs['confidence_score'].item()
            else:
                # Fallback for models without verification head
                verification_score = 0.5  # Neutral
                confidence_score = 0.5
        
        # Also use tool-based verification
        try:
            z3_result = client.check_logical_consistency(f"verify({claim}, {response})")
            tool_score = 1.0 if z3_result.get('satisfiable', False) else 0.0
            tool_confidence = 0.9 if 'error' not in z3_result else 0.1
        except:
            tool_score = 0.5
            tool_confidence = 0.1
        
        # Combine model and tool results
        combined_score = 0.6 * verification_score + 0.4 * tool_score
        combined_confidence = 0.7 * confidence_score + 0.3 * tool_confidence
        
        return {
            'score': float(combined_score),
            'confidence': float(combined_confidence),
            'model_score': verification_score,
            'tool_score': tool_score
        }
    
    async def _verify_contextual(self, claim: str, response: str, 
                               context: Optional[Dict[str, Any]] = None) -> Dict[str, float]:
        """Verify using contextual framework."""
        # Placeholder implementation
        # In practice, would use contextual tools and mini-LLM
        
        # Simulate contextual analysis
        context_score = np.random.random()  # Replace with actual analysis
        context_confidence = np.random.random()
        
        return {
            'score': float(context_score),
            'confidence': float(context_confidence)
        }
    
    def _aggregate_rewards(self, scores: torch.Tensor, confidences: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Aggregate framework rewards using specified method."""
        
        # Normalize weights
        weights = F.softmax(self.framework_weights, dim=0)
        
        if self.config.aggregation_method == "weighted_sum":
            # Confidence-weighted sum
            if self.config.confidence_weighting:
                effective_weights = weights * confidences
                effective_weights = effective_weights / (effective_weights.sum() + 1e-8)
                base_reward = (scores * effective_weights).sum()
            else:
                base_reward = (scores * weights).sum()
                
        elif self.config.aggregation_method == "weighted_product":
            # Geometric mean with weights
            log_scores = torch.log(torch.clamp(scores, min=1e-8))
            base_reward = torch.exp((log_scores * weights).sum())
            
        elif self.config.aggregation_method == "max_consensus":
            # Weighted maximum
            base_reward = (scores * weights).max()
            
        elif self.config.aggregation_method == "min_consensus":
            # Weighted minimum (conservative)
            base_reward = (scores * weights).min()
            
        else:
            # Default to weighted sum
            base_reward = (scores * weights).sum()
        
        # Compute additional components
        avg_confidence = confidences.mean()
        confidence_bonus = avg_confidence * 0.1
        
        # Disagreement penalty
        score_std = scores.std()
        disagreement_penalty = score_std * self.config.disagreement_penalty
        
        # Use neural networks for more complex combinations
        nn_reward = self.reward_combiner(scores.unsqueeze(0)).squeeze()
        nn_confidence = self.confidence_combiner(confidences.unsqueeze(0)).squeeze()
        nn_disagreement = self.disagreement_estimator(scores.unsqueeze(0)).squeeze()
        
        # Combine traditional and neural approaches
        total_reward = (
            0.7 * base_reward + 
            0.3 * nn_reward +
            confidence_bonus * nn_confidence -
            disagreement_penalty * nn_disagreement
        )
        
        return {
            'total_reward': total_reward,
            'base_reward': base_reward,
            'nn_reward': nn_reward,
            'confidence_bonus': confidence_bonus,
            'disagreement_penalty': disagreement_penalty,
            'avg_confidence': avg_confidence
        }
    
    def _add_temporal_factors(self, reward_components: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Add temporal and historical factors to reward."""
        
        # Temporal bonus based on recent performance
        if len(self.reward_history) > 1:
            recent_trend = np.mean(self.reward_history[-5:]) - np.mean(self.reward_history[-10:-5]) if len(self.reward_history) >= 10 else 0
            temporal_bonus = torch.tensor(recent_trend * self.config.history_weight, device=self.device)
            reward_components['temporal_bonus'] = temporal_bonus
            reward_components['total_reward'] += temporal_bonus
        
        return reward_components
    
    def _normalize_reward(self, reward: torch.Tensor) -> torch.Tensor:
        """Normalize and clip reward."""
        if self.config.normalize_rewards and len(self.reward_history) > 10:
            # Z-score normalization based on history
            mean_reward = np.mean(self.reward_history)
            std_reward = np.std(self.reward_history) + 1e-8
            reward = (reward - mean_reward) / std_reward
        
        # Scale and clip
        reward = reward * self.config.reward_scale
        reward = torch.clamp(reward, -self.config.reward_clip, self.config.reward_clip)
        
        return reward
    
    def update_framework_weights(self, framework_performance: Dict[str, float]):
        """Update framework weights based on performance."""
        # This could be called during training to adapt weights
        
        performance_tensor = torch.tensor([
            framework_performance.get('empirical', 0.5),
            framework_performance.get('contextual', 0.5),
            framework_performance.get('consistency', 0.5),
            framework_performance.get('power_dynamics', 0.5),
            framework_performance.get('utility', 0.5),
            framework_performance.get('evolution', 0.5)
        ], device=self.device)
        
        # Exponential moving average update
        alpha = self.config.weight_learning_rate
        self.framework_weights.data = (
            (1 - alpha) * self.framework_weights.data + 
            alpha * performance_tensor
        )
    
    def get_framework_weights(self) -> Dict[str, float]:
        """Get current framework weights."""
        weights = F.softmax(self.framework_weights, dim=0)
        return {
            'empirical': weights[0].item(),
            'contextual': weights[1].item(),
            'consistency': weights[2].item(),
            'power_dynamics': weights[3].item(),
            'utility': weights[4].item(),
            'evolution': weights[5].item()
        }
    
    def reset_history(self):
        """Reset reward and confidence history."""
        self.reward_history = []
        self.confidence_history = []
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get reward model statistics."""
        if not self.reward_history:
            return {'message': 'No history available'}
        
        return {
            'mean_reward': np.mean(self.reward_history),
            'std_reward': np.std(self.reward_history),
            'mean_confidence': np.mean(self.confidence_history),
            'std_confidence': np.std(self.confidence_history),
            'num_samples': len(self.reward_history),
            'framework_weights': self.get_framework_weights(),
            'recent_trend': np.mean(self.reward_history[-10:]) - np.mean(self.reward_history[-20:-10]) if len(self.reward_history) >= 20 else 0
        }
    
    def save_state(self, path: str):
        """Save reward model state."""
        state = {
            'framework_weights': self.framework_weights.data,
            'reward_combiner_state_dict': self.reward_combiner.state_dict(),
            'confidence_combiner_state_dict': self.confidence_combiner.state_dict(),
            'disagreement_estimator_state_dict': self.disagreement_estimator.state_dict(),
            'reward_history': self.reward_history,
            'confidence_history': self.confidence_history,
            'config': self.config
        }
        torch.save(state, path)
    
    def load_state(self, path: str):
        """Load reward model state."""
        state = torch.load(path, map_location=self.device)
        
        self.framework_weights.data = state['framework_weights']
        self.reward_combiner.load_state_dict(state['reward_combiner_state_dict'])
        self.confidence_combiner.load_state_dict(state['confidence_combiner_state_dict'])
        self.disagreement_estimator.load_state_dict(state['disagreement_estimator_state_dict'])
        self.reward_history = state['reward_history']
        self.confidence_history = state['confidence_history']


def create_default_reward_model(device: str = "cuda") -> MultiFrameworkRewardModel:
    """Create reward model with default configuration."""
    config = RewardConfig()
    registry = MiniLLMRegistry(device=device)
    
    return MultiFrameworkRewardModel(config, registry, device)