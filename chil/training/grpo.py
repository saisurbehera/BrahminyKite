"""
Group Relative Policy Optimization (GRPO) for BrahminyKite.

GRPO extends standard policy optimization by considering multiple
philosophical frameworks as different "groups" and optimizing
relative performance across these groups.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
from collections import defaultdict

from ..env import AsyncBrahminyKiteEnv


@dataclass
class GRPOConfig:
    """Configuration for GRPO training."""
    
    # Model architecture
    model_dim: int = 512
    num_heads: int = 8
    num_layers: int = 6
    vocab_size: int = 50000
    max_seq_length: int = 512
    
    # Training hyperparameters
    learning_rate: float = 3e-4
    batch_size: int = 32
    gradient_accumulation_steps: int = 4
    max_grad_norm: float = 1.0
    
    # GRPO-specific parameters
    num_frameworks: int = 6
    temperature: float = 0.7
    kl_coeff: float = 0.1
    value_coeff: float = 0.5
    entropy_coeff: float = 0.01
    
    # Group relative parameters
    group_balance_weight: float = 0.2
    diversity_bonus: float = 0.1
    disagreement_penalty: float = 0.05
    
    # Training schedule
    warmup_steps: int = 1000
    total_steps: int = 100000
    eval_interval: int = 1000
    save_interval: int = 5000
    
    # Framework weights (learned dynamically)
    initial_framework_weights: List[float] = None


class PolicyNetwork(nn.Module):
    """Policy network for generating text responses."""
    
    def __init__(self, config: GRPOConfig):
        super().__init__()
        self.config = config
        
        # Embedding layers
        self.token_embedding = nn.Embedding(config.vocab_size, config.model_dim)
        self.position_embedding = nn.Embedding(config.max_seq_length, config.model_dim)
        
        # Transformer layers
        self.transformer = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model=config.model_dim,
                nhead=config.num_heads,
                dim_feedforward=config.model_dim * 4,
                dropout=0.1,
                batch_first=True
            ),
            num_layers=config.num_layers
        )
        
        # Output head
        self.output_head = nn.Linear(config.model_dim, config.vocab_size)
        
        # Framework-specific heads for adaptation
        self.framework_adapters = nn.ModuleDict({
            f'framework_{i}': nn.Linear(config.model_dim, config.model_dim)
            for i in range(config.num_frameworks)
        })
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, input_ids: torch.Tensor, 
                framework_weights: torch.Tensor = None,
                active_frameworks: List[int] = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass with framework-aware adaptation.
        
        Args:
            input_ids: Token IDs [batch_size, seq_len]
            framework_weights: Weights for each framework [batch_size, num_frameworks]
            active_frameworks: List of active framework indices
        """
        batch_size, seq_len = input_ids.shape
        
        # Embeddings
        token_embeds = self.token_embedding(input_ids)
        position_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        position_embeds = self.position_embedding(position_ids)
        
        hidden_states = token_embeds + position_embeds
        
        # Create causal mask
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=input_ids.device) * float('-inf'),
            diagonal=1
        )
        
        # Transformer forward pass
        hidden_states = self.transformer(
            hidden_states,
            memory=hidden_states,
            tgt_mask=causal_mask
        )
        
        # Framework-specific adaptation
        if framework_weights is not None and active_frameworks is not None:
            adapted_states = torch.zeros_like(hidden_states)
            
            for i, fw_idx in enumerate(active_frameworks):
                if fw_idx < len(self.framework_adapters):
                    adapter = self.framework_adapters[f'framework_{fw_idx}']
                    weight = framework_weights[:, fw_idx].unsqueeze(-1).unsqueeze(-1)
                    adapted_states += weight * adapter(hidden_states)
            
            hidden_states = hidden_states + adapted_states
        
        # Output logits
        logits = self.output_head(hidden_states)
        
        return {
            'logits': logits,
            'hidden_states': hidden_states
        }


class ValueNetwork(nn.Module):
    """Value network for estimating state values."""
    
    def __init__(self, config: GRPOConfig):
        super().__init__()
        self.config = config
        
        # Shared backbone with policy
        self.backbone = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=config.model_dim,
                nhead=config.num_heads,
                dim_feedforward=config.model_dim * 4,
                dropout=0.1,
                batch_first=True
            ),
            num_layers=config.num_layers // 2
        )
        
        # Framework-specific value heads
        self.framework_value_heads = nn.ModuleDict({
            f'framework_{i}': nn.Sequential(
                nn.Linear(config.model_dim, config.model_dim // 2),
                nn.ReLU(),
                nn.Linear(config.model_dim // 2, 1)
            )
            for i in range(config.num_frameworks)
        })
        
        # Global value head
        self.global_value_head = nn.Sequential(
            nn.Linear(config.model_dim, config.model_dim // 2),
            nn.ReLU(),
            nn.Linear(config.model_dim // 2, 1)
        )
    
    def forward(self, hidden_states: torch.Tensor,
                active_frameworks: List[int] = None) -> Dict[str, torch.Tensor]:
        """
        Compute values for each framework and globally.
        
        Args:
            hidden_states: Hidden states from policy network
            active_frameworks: List of active framework indices
        """
        # Process through backbone
        encoded = self.backbone(hidden_states)
        
        # Pool sequence dimension (mean pooling)
        pooled = encoded.mean(dim=1)
        
        # Compute global value
        global_value = self.global_value_head(pooled)
        
        # Compute framework-specific values
        framework_values = {}
        if active_frameworks:
            for fw_idx in active_frameworks:
                if f'framework_{fw_idx}' in self.framework_value_heads:
                    fw_value = self.framework_value_heads[f'framework_{fw_idx}'](pooled)
                    framework_values[f'framework_{fw_idx}'] = fw_value
        
        return {
            'global_value': global_value,
            'framework_values': framework_values
        }


class GRPOOptimizer:
    """Group Relative Policy Optimization trainer."""
    
    def __init__(self, config: GRPOConfig, env: AsyncBrahminyKiteEnv, device: str = 'cuda'):
        self.config = config
        self.env = env
        self.device = device
        
        # Initialize networks
        self.policy = PolicyNetwork(config).to(device)
        self.value = ValueNetwork(config).to(device)
        
        # Optimizers
        self.policy_optimizer = Adam(self.policy.parameters(), lr=config.learning_rate)
        self.value_optimizer = Adam(self.value.parameters(), lr=config.learning_rate)
        
        # Learning rate schedulers
        self.policy_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.policy_optimizer, T_max=config.total_steps
        )
        self.value_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.value_optimizer, T_max=config.total_steps
        )
        
        # Framework weights (learnable parameters)
        if config.initial_framework_weights:
            initial_weights = torch.tensor(config.initial_framework_weights)
        else:
            initial_weights = torch.ones(config.num_frameworks) / config.num_frameworks
        
        self.framework_weights = nn.Parameter(initial_weights.to(device))
        self.framework_optimizer = Adam([self.framework_weights], lr=config.learning_rate * 0.1)
        
        # Training state
        self.step_count = 0
        self.episode_count = 0
        self.training_stats = defaultdict(list)
    
    def collect_rollouts(self, num_episodes: int) -> Dict[str, torch.Tensor]:
        """Collect rollout data from environment."""
        episodes_data = []
        
        for _ in range(num_episodes):
            episode_data = self._collect_single_episode()
            episodes_data.append(episode_data)
        
        # Batch episodes
        batched_data = self._batch_episodes(episodes_data)
        return batched_data
    
    async def _collect_single_episode(self) -> Dict[str, Any]:
        """Collect data from a single episode."""
        obs = self.env.reset()
        episode_data = {
            'observations': [],
            'actions': [],
            'rewards': [],
            'framework_rewards': [],
            'log_probs': [],
            'values': [],
            'dones': []
        }
        
        done = False
        while not done:
            # Convert observation to tensor
            obs_tensor = self._obs_to_tensor(obs)
            
            # Get current framework weights (softmax normalized)
            current_weights = F.softmax(self.framework_weights, dim=0)
            
            # Policy forward pass
            with torch.no_grad():
                policy_output = self.policy(
                    obs_tensor['input_ids'],
                    framework_weights=current_weights.unsqueeze(0),
                    active_frameworks=list(range(self.config.num_frameworks))
                )
                
                # Sample action
                logits = policy_output['logits'][:, -1, :]  # Last token logits
                probs = F.softmax(logits / self.config.temperature, dim=-1)
                action = torch.multinomial(probs, 1)
                log_prob = F.log_softmax(logits / self.config.temperature, dim=-1)
                action_log_prob = log_prob.gather(-1, action)
                
                # Value estimation
                value_output = self.value(
                    policy_output['hidden_states'],
                    active_frameworks=list(range(self.config.num_frameworks))
                )
        
            # Convert action to environment format
            env_action = self._tensor_to_action(action, current_weights)
            
            # Step environment
            next_obs, reward, done, info = await self.env.async_step(env_action)
            
            # Store data
            episode_data['observations'].append(obs)
            episode_data['actions'].append(action)
            episode_data['rewards'].append(reward)
            episode_data['framework_rewards'].append(info.get('framework_results', {}))
            episode_data['log_probs'].append(action_log_prob)
            episode_data['values'].append(value_output)
            episode_data['dones'].append(done)
            
            obs = next_obs
        
        return episode_data
    
    def _compute_grpo_loss(self, batch_data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Compute GRPO loss with group relative optimization."""
        
        # Unpack batch data
        observations = batch_data['observations']
        actions = batch_data['actions']
        rewards = batch_data['rewards']
        framework_rewards = batch_data['framework_rewards']
        old_log_probs = batch_data['log_probs']
        
        # Forward pass
        current_weights = F.softmax(self.framework_weights, dim=0)
        policy_output = self.policy(
            observations,
            framework_weights=current_weights.unsqueeze(0).expand(observations.shape[0], -1),
            active_frameworks=list(range(self.config.num_frameworks))
        )
        
        value_output = self.value(
            policy_output['hidden_states'],
            active_frameworks=list(range(self.config.num_frameworks))
        )
        
        # Compute advantages
        advantages = self._compute_advantages(rewards, value_output['global_value'])
        
        # Policy loss (PPO-style clipped objective)
        logits = policy_output['logits']
        new_log_probs = F.log_softmax(logits / self.config.temperature, dim=-1)
        action_log_probs = new_log_probs.gather(-1, actions)
        
        ratio = torch.exp(action_log_probs - old_log_probs)
        clipped_ratio = torch.clamp(ratio, 1.0 - 0.2, 1.0 + 0.2)
        
        policy_loss = -torch.min(
            ratio * advantages,
            clipped_ratio * advantages
        ).mean()
        
        # Value loss
        value_loss = F.mse_loss(value_output['global_value'], rewards)
        
        # Framework-specific losses
        framework_losses = []
        for i, fw_name in enumerate(['empirical', 'contextual', 'consistency', 'power', 'utility', 'evolution']):
            if f'framework_{i}' in value_output['framework_values']:
                fw_rewards = framework_rewards[:, i] if framework_rewards.dim() > 1 else framework_rewards
                fw_value = value_output['framework_values'][f'framework_{i}']
                fw_loss = F.mse_loss(fw_value, fw_rewards.unsqueeze(-1))
                framework_losses.append(fw_loss)
        
        framework_loss = torch.stack(framework_losses).mean() if framework_losses else torch.tensor(0.0)
        
        # Group relative terms
        group_balance_loss = self._compute_group_balance_loss(framework_rewards, current_weights)
        diversity_bonus = self._compute_diversity_bonus(current_weights)
        disagreement_penalty = self._compute_disagreement_penalty(framework_rewards)
        
        # Entropy loss for exploration
        entropy = -(F.softmax(logits, dim=-1) * F.log_softmax(logits, dim=-1)).sum(-1).mean()
        entropy_loss = -self.config.entropy_coeff * entropy
        
        # Total loss
        total_loss = (
            policy_loss + 
            self.config.value_coeff * value_loss +
            self.config.value_coeff * framework_loss +
            self.config.group_balance_weight * group_balance_loss +
            self.config.diversity_bonus * diversity_bonus +
            self.config.disagreement_penalty * disagreement_penalty +
            entropy_loss
        )
        
        return {
            'total_loss': total_loss,
            'policy_loss': policy_loss,
            'value_loss': value_loss,
            'framework_loss': framework_loss,
            'group_balance_loss': group_balance_loss,
            'diversity_bonus': diversity_bonus,
            'disagreement_penalty': disagreement_penalty,
            'entropy_loss': entropy_loss
        }
    
    def _compute_group_balance_loss(self, framework_rewards: torch.Tensor, 
                                  weights: torch.Tensor) -> torch.Tensor:
        """Encourage balanced performance across frameworks."""
        # Compute weighted framework performance
        weighted_performance = (framework_rewards * weights.unsqueeze(0)).sum(-1)
        
        # Penalize large variance in framework performance
        framework_variance = framework_rewards.var(dim=-1)
        
        return framework_variance.mean()
    
    def _compute_diversity_bonus(self, weights: torch.Tensor) -> torch.Tensor:
        """Encourage diversity in framework usage."""
        # Entropy of framework weights (higher is better)
        weight_entropy = -(weights * torch.log(weights + 1e-8)).sum()
        return -weight_entropy  # Negative because we want to maximize entropy
    
    def _compute_disagreement_penalty(self, framework_rewards: torch.Tensor) -> torch.Tensor:
        """Penalize high disagreement between frameworks."""
        # Standard deviation across frameworks
        disagreement = framework_rewards.std(dim=-1)
        return disagreement.mean()
    
    def _compute_advantages(self, rewards: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
        """Compute generalized advantage estimation."""
        advantages = rewards - values.squeeze(-1)
        return advantages
    
    def _obs_to_tensor(self, obs: Dict[str, np.ndarray]) -> Dict[str, torch.Tensor]:
        """Convert environment observation to tensors."""
        return {
            'input_ids': torch.tensor(obs['claim'], dtype=torch.long, device=self.device).unsqueeze(0)
        }
    
    def _tensor_to_action(self, action_tensor: torch.Tensor, weights: torch.Tensor) -> Dict[str, np.ndarray]:
        """Convert tensor action to environment format."""
        # Convert action tensor back to text (simplified)
        claim_text = "Generated claim based on action"  # In practice, decode from vocab
        
        return {
            'claim': np.array([ord(c) for c in claim_text.ljust(512)[:512]], dtype=np.uint8),
            'framework_weights': weights.detach().cpu().numpy()
        }
    
    def _batch_episodes(self, episodes_data: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """Batch episode data for training."""
        # This is simplified - in practice, you'd need proper batching with padding
        return episodes_data[0]  # Placeholder
    
    def train_step(self, batch_data: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Perform a single training step."""
        
        # Compute losses
        losses = self._compute_grpo_loss(batch_data)
        
        # Policy update
        self.policy_optimizer.zero_grad()
        losses['policy_loss'].backward(retain_graph=True)
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.config.max_grad_norm)
        self.policy_optimizer.step()
        
        # Value update
        self.value_optimizer.zero_grad()
        (losses['value_loss'] + losses['framework_loss']).backward(retain_graph=True)
        torch.nn.utils.clip_grad_norm_(self.value.parameters(), self.config.max_grad_norm)
        self.value_optimizer.step()
        
        # Framework weights update
        self.framework_optimizer.zero_grad()
        (losses['group_balance_loss'] + losses['diversity_bonus'] + losses['disagreement_penalty']).backward()
        self.framework_optimizer.step()
        
        # Update learning rates
        self.policy_scheduler.step()
        self.value_scheduler.step()
        
        # Update counters
        self.step_count += 1
        
        # Return loss values for logging
        return {k: v.item() for k, v in losses.items()}
    
    def save_checkpoint(self, path: str):
        """Save training checkpoint."""
        checkpoint = {
            'step_count': self.step_count,
            'policy_state_dict': self.policy.state_dict(),
            'value_state_dict': self.value.state_dict(),
            'framework_weights': self.framework_weights.data,
            'policy_optimizer_state_dict': self.policy_optimizer.state_dict(),
            'value_optimizer_state_dict': self.value_optimizer.state_dict(),
            'framework_optimizer_state_dict': self.framework_optimizer.state_dict(),
            'config': self.config
        }
        torch.save(checkpoint, path)
    
    def load_checkpoint(self, path: str):
        """Load training checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.step_count = checkpoint['step_count']
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.value.load_state_dict(checkpoint['value_state_dict'])
        self.framework_weights.data = checkpoint['framework_weights']
        self.policy_optimizer.load_state_dict(checkpoint['policy_optimizer_state_dict'])
        self.value_optimizer.load_state_dict(checkpoint['value_optimizer_state_dict'])
        self.framework_optimizer.load_state_dict(checkpoint['framework_optimizer_state_dict'])