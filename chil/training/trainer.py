"""
Main trainer for BrahminyKite GRPO system.

Integrates all components: environment, tools, mini-LLMs, reward model, and GRPO optimizer.
"""

import asyncio
import torch
import numpy as np
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import wandb
import logging
from pathlib import Path
import json
from tqdm import tqdm

from .grpo import GRPOOptimizer, GRPOConfig
from .reward_model import MultiFrameworkRewardModel, RewardConfig
from .mini_llms import MiniLLMRegistry, FrameworkType
from ..env import AsyncBrahminyKiteEnv
from ..tools.service_manager import ServiceManager


@dataclass
class TrainingConfig:
    """Configuration for training pipeline."""
    
    # Training parameters
    num_episodes: int = 1000
    max_steps_per_episode: int = 100
    batch_size: int = 32
    gradient_accumulation_steps: int = 4
    
    # Evaluation
    eval_episodes: int = 50
    eval_interval: int = 100
    
    # Checkpointing
    save_interval: int = 500
    checkpoint_dir: str = "./checkpoints"
    
    # Logging
    log_interval: int = 10
    use_wandb: bool = True
    wandb_project: str = "brahminy-kite"
    wandb_entity: Optional[str] = None
    
    # Environment config
    env_config: Dict[str, Any] = None
    
    # Early stopping
    patience: int = 50
    min_delta: float = 0.001


class BrahminyKiteTrainer:
    """Main training orchestrator."""
    
    def __init__(self, 
                 training_config: TrainingConfig,
                 grpo_config: GRPOConfig,
                 reward_config: RewardConfig,
                 device: str = "cuda"):
        
        self.training_config = training_config
        self.grpo_config = grpo_config
        self.reward_config = reward_config
        self.device = device
        
        # Setup logging
        self.logger = self._setup_logging()
        
        # Initialize components
        self.service_manager = None
        self.env = None
        self.mini_llm_registry = None
        self.reward_model = None
        self.grpo_optimizer = None
        
        # Training state
        self.episode_count = 0
        self.step_count = 0
        self.best_reward = float('-inf')
        self.patience_counter = 0
        
        # Metrics storage
        self.training_metrics = []
        self.eval_metrics = []
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration."""
        logger = logging.getLogger('brahminy_kite_trainer')
        logger.setLevel(logging.INFO)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        # File handler
        Path("logs").mkdir(exist_ok=True)
        file_handler = logging.FileHandler('logs/training.log')
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        return logger
    
    async def initialize(self):
        """Initialize all training components."""
        self.logger.info("Initializing BrahminyKite training pipeline...")
        
        # Start tool services
        self.logger.info("Starting tool services...")
        self.service_manager = ServiceManager()
        self.service_manager.start_all()
        
        # Wait for services to be ready
        await asyncio.sleep(2)
        
        # Initialize environment
        self.logger.info("Initializing environment...")
        env_config = self.training_config.env_config or {
            'frameworks': ['empirical', 'contextual', 'consistency', 'power_dynamics', 'utility', 'evolution'],
            'max_steps': self.training_config.max_steps_per_episode
        }
        self.env = AsyncBrahminyKiteEnv(env_config)
        
        # Initialize mini-LLM registry
        self.logger.info("Loading mini-LLMs...")
        self.mini_llm_registry = MiniLLMRegistry(device=self.device)
        self.mini_llm_registry.load_all_models()
        
        # Initialize reward model
        self.logger.info("Initializing reward model...")
        self.reward_model = MultiFrameworkRewardModel(
            self.reward_config, 
            self.mini_llm_registry, 
            self.device
        )
        
        # Initialize GRPO optimizer
        self.logger.info("Initializing GRPO optimizer...")
        self.grpo_optimizer = GRPOOptimizer(
            self.grpo_config,
            self.env,
            self.device
        )
        
        # Setup wandb
        if self.training_config.use_wandb:
            wandb.init(
                project=self.training_config.wandb_project,
                entity=self.training_config.wandb_entity,
                config={
                    "training_config": self.training_config.__dict__,
                    "grpo_config": self.grpo_config.__dict__,
                    "reward_config": self.reward_config.__dict__
                }
            )
        
        self.logger.info("✓ All components initialized successfully!")
    
    async def train(self):
        """Main training loop."""
        self.logger.info("Starting training...")
        
        try:
            for episode in tqdm(range(self.training_config.num_episodes), desc="Training Episodes"):
                self.episode_count = episode
                
                # Collect rollouts
                episode_data = await self._collect_episode_data()
                
                # Compute rewards using reward model
                enhanced_data = await self._enhance_with_rewards(episode_data)
                
                # Train GRPO
                losses = self.grpo_optimizer.train_step(enhanced_data)
                
                # Log metrics
                if episode % self.training_config.log_interval == 0:
                    await self._log_training_metrics(episode, losses, episode_data)
                
                # Evaluate
                if episode % self.training_config.eval_interval == 0:
                    eval_results = await self._evaluate()
                    await self._log_eval_metrics(episode, eval_results)
                    
                    # Check for improvement
                    current_reward = eval_results['mean_reward']
                    if current_reward > self.best_reward + self.training_config.min_delta:
                        self.best_reward = current_reward
                        self.patience_counter = 0
                        await self._save_best_checkpoint(episode)
                    else:
                        self.patience_counter += 1
                
                # Save checkpoint
                if episode % self.training_config.save_interval == 0:
                    await self._save_checkpoint(episode)
                
                # Early stopping
                if self.patience_counter >= self.training_config.patience:
                    self.logger.info(f"Early stopping at episode {episode}")
                    break
        
        except KeyboardInterrupt:
            self.logger.info("Training interrupted by user")
        except Exception as e:
            self.logger.error(f"Training failed with error: {e}")
            raise
        finally:
            await self._cleanup()
    
    async def _collect_episode_data(self) -> Dict[str, Any]:
        """Collect data from a single episode."""
        obs = self.env.reset()
        episode_data = {
            'observations': [],
            'actions': [],
            'claims': [],
            'responses': [],
            'raw_rewards': [],
            'dones': []
        }
        
        done = False
        step = 0
        
        while not done and step < self.training_config.max_steps_per_episode:
            # Generate action using current policy
            action = await self._generate_action(obs)
            
            # Step environment
            next_obs, raw_reward, done, info = await self.env.async_step(action)
            
            # Store data
            episode_data['observations'].append(obs)
            episode_data['actions'].append(action)
            episode_data['claims'].append(info.get('claim', ''))
            episode_data['responses'].append(action.get('response', ''))  # Generated response
            episode_data['raw_rewards'].append(raw_reward)
            episode_data['dones'].append(done)
            
            obs = next_obs
            step += 1
            self.step_count += 1
        
        return episode_data
    
    async def _generate_action(self, obs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Generate action using GRPO policy."""
        # Convert observation to tensor
        obs_tensor = self._obs_to_tensor(obs)
        
        # Get current framework weights
        framework_weights = torch.softmax(self.grpo_optimizer.framework_weights, dim=0)
        
        # Policy forward pass
        with torch.no_grad():
            policy_output = self.grpo_optimizer.policy(
                obs_tensor['input_ids'],
                framework_weights=framework_weights.unsqueeze(0),
                active_frameworks=list(range(self.grpo_config.num_frameworks))
            )
            
            # Sample response
            logits = policy_output['logits'][:, -1, :]
            probs = torch.softmax(logits / self.grpo_config.temperature, dim=-1)
            response_tokens = torch.multinomial(probs, num_samples=50)  # Generate 50 tokens
        
        # Convert to action format
        action = self._tensor_to_action(response_tokens, framework_weights)
        return action
    
    async def _enhance_with_rewards(self, episode_data: Dict[str, Any]) -> Dict[str, Any]:
        """Enhance episode data with reward model computations."""
        enhanced_rewards = []
        
        for i in range(len(episode_data['claims'])):
            claim = episode_data['claims'][i]
            response = episode_data['responses'][i]
            
            # Compute enhanced reward
            reward_info = await self.reward_model.compute_reward(claim, response)
            enhanced_rewards.append(reward_info['total_reward'])
        
        # Replace raw rewards with enhanced rewards
        episode_data['enhanced_rewards'] = enhanced_rewards
        episode_data['reward_components'] = [
            await self.reward_model.compute_reward(
                episode_data['claims'][i], 
                episode_data['responses'][i]
            )
            for i in range(len(episode_data['claims']))
        ]
        
        return episode_data
    
    async def _evaluate(self) -> Dict[str, Any]:
        """Evaluate current policy."""
        self.logger.info("Running evaluation...")
        
        eval_rewards = []
        eval_framework_scores = {fw.value: [] for fw in FrameworkType}
        
        for _ in range(self.training_config.eval_episodes):
            obs = self.env.reset()
            episode_reward = 0
            done = False
            step = 0
            
            while not done and step < self.training_config.max_steps_per_episode:
                action = await self._generate_action(obs)
                next_obs, reward, done, info = await self.env.async_step(action)
                
                # Get framework scores
                framework_results = info.get('framework_results', {})
                for framework, score in framework_results.items():
                    if framework in eval_framework_scores:
                        eval_framework_scores[framework].append(score)
                
                episode_reward += reward
                obs = next_obs
                step += 1
            
            eval_rewards.append(episode_reward)
        
        # Compute statistics
        eval_results = {
            'mean_reward': np.mean(eval_rewards),
            'std_reward': np.std(eval_rewards),
            'min_reward': np.min(eval_rewards),
            'max_reward': np.max(eval_rewards),
            'framework_scores': {
                fw: {
                    'mean': np.mean(scores) if scores else 0.0,
                    'std': np.std(scores) if scores else 0.0
                }
                for fw, scores in eval_framework_scores.items()
            }
        }
        
        return eval_results
    
    async def _log_training_metrics(self, episode: int, losses: Dict[str, float], episode_data: Dict[str, Any]):
        """Log training metrics."""
        metrics = {
            'episode': episode,
            'step': self.step_count,
            'episode_length': len(episode_data['claims']),
            'mean_episode_reward': np.mean(episode_data.get('enhanced_rewards', episode_data['raw_rewards'])),
            **losses
        }
        
        # Add framework weights
        framework_weights = self.reward_model.get_framework_weights()
        for fw, weight in framework_weights.items():
            metrics[f'framework_weight_{fw}'] = weight
        
        self.training_metrics.append(metrics)
        
        # Log to wandb
        if self.training_config.use_wandb:
            wandb.log(metrics)
        
        # Log to console
        self.logger.info(
            f"Episode {episode}: Reward={metrics['mean_episode_reward']:.3f}, "
            f"Policy Loss={losses['policy_loss']:.4f}, "
            f"Value Loss={losses['value_loss']:.4f}"
        )
    
    async def _log_eval_metrics(self, episode: int, eval_results: Dict[str, Any]):
        """Log evaluation metrics."""
        eval_metrics = {
            'eval_episode': episode,
            'eval_mean_reward': eval_results['mean_reward'],
            'eval_std_reward': eval_results['std_reward'],
            'eval_min_reward': eval_results['min_reward'],
            'eval_max_reward': eval_results['max_reward']
        }
        
        # Add framework evaluation scores
        for fw, scores in eval_results['framework_scores'].items():
            eval_metrics[f'eval_{fw}_mean'] = scores['mean']
            eval_metrics[f'eval_{fw}_std'] = scores['std']
        
        self.eval_metrics.append(eval_metrics)
        
        # Log to wandb
        if self.training_config.use_wandb:
            wandb.log(eval_metrics)
        
        self.logger.info(
            f"Evaluation at episode {episode}: "
            f"Mean Reward={eval_results['mean_reward']:.3f} ± {eval_results['std_reward']:.3f}"
        )
    
    async def _save_checkpoint(self, episode: int):
        """Save training checkpoint."""
        checkpoint_dir = Path(self.training_config.checkpoint_dir)
        checkpoint_dir.mkdir(exist_ok=True)
        
        checkpoint_path = checkpoint_dir / f"checkpoint_episode_{episode}.pt"
        
        # Save GRPO optimizer state
        self.grpo_optimizer.save_checkpoint(str(checkpoint_path))
        
        # Save reward model state
        reward_model_path = checkpoint_dir / f"reward_model_episode_{episode}.pt"
        self.reward_model.save_state(str(reward_model_path))
        
        # Save training metrics
        metrics_path = checkpoint_dir / f"metrics_episode_{episode}.json"
        with open(metrics_path, 'w') as f:
            json.dump({
                'training_metrics': self.training_metrics,
                'eval_metrics': self.eval_metrics,
                'episode': episode,
                'step': self.step_count
            }, f, indent=2)
        
        self.logger.info(f"Checkpoint saved at episode {episode}")
    
    async def _save_best_checkpoint(self, episode: int):
        """Save best model checkpoint."""
        best_dir = Path(self.training_config.checkpoint_dir) / "best"
        best_dir.mkdir(exist_ok=True)
        
        # Save GRPO optimizer
        self.grpo_optimizer.save_checkpoint(str(best_dir / "best_grpo.pt"))
        
        # Save reward model
        self.reward_model.save_state(str(best_dir / "best_reward_model.pt"))
        
        # Save metadata
        with open(best_dir / "best_info.json", 'w') as f:
            json.dump({
                'episode': episode,
                'step': self.step_count,
                'best_reward': self.best_reward,
                'framework_weights': self.reward_model.get_framework_weights()
            }, f, indent=2)
        
        self.logger.info(f"Best checkpoint saved at episode {episode} with reward {self.best_reward:.3f}")
    
    async def _cleanup(self):
        """Cleanup resources."""
        self.logger.info("Cleaning up...")
        
        if self.env:
            self.env.close()
        
        if self.service_manager:
            self.service_manager.stop_all()
        
        if self.training_config.use_wandb:
            wandb.finish()
    
    def _obs_to_tensor(self, obs: Dict[str, np.ndarray]) -> Dict[str, torch.Tensor]:
        """Convert observation to tensor."""
        return {
            'input_ids': torch.tensor(obs['claim'], dtype=torch.long, device=self.device).unsqueeze(0)
        }
    
    def _tensor_to_action(self, response_tokens: torch.Tensor, weights: torch.Tensor) -> Dict[str, np.ndarray]:
        """Convert tensor response to action."""
        # Convert tokens back to text (simplified)
        response_text = "Generated response based on tokens"  # In practice, decode from vocab
        
        return {
            'claim': np.array([ord(c) for c in response_text.ljust(512)[:512]], dtype=np.uint8),
            'framework_weights': weights.detach().cpu().numpy(),
            'response': response_text
        }


# Convenience function for quick training setup
def create_trainer(config_overrides: Dict[str, Any] = None) -> BrahminyKiteTrainer:
    """Create trainer with default configurations."""
    
    # Default configurations
    training_config = TrainingConfig()
    grpo_config = GRPOConfig()
    reward_config = RewardConfig()
    
    # Apply overrides
    if config_overrides:
        for key, value in config_overrides.items():
            if hasattr(training_config, key):
                setattr(training_config, key, value)
            elif hasattr(grpo_config, key):
                setattr(grpo_config, key, value)
            elif hasattr(reward_config, key):
                setattr(reward_config, key, value)
    
    return BrahminyKiteTrainer(
        training_config=training_config,
        grpo_config=grpo_config,
        reward_config=reward_config
    )


# Example usage
async def main():
    """Example training run."""
    
    # Create trainer
    trainer = create_trainer({
        'num_episodes': 1000,
        'eval_interval': 50,
        'use_wandb': True,
        'wandb_project': 'brahminy-kite-experiment'
    })
    
    # Initialize and train
    await trainer.initialize()
    await trainer.train()


if __name__ == "__main__":
    asyncio.run(main())