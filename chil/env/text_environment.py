"""
Text-based OpenAI Gym environment for BrahminyKite.
"""

import gym
from gym import spaces
import numpy as np
from typing import Dict, Tuple, Any, List, Optional
import json
import asyncio

from ..tools.clients import (
    EmpiricalToolsClient,
    # ContextualToolsClient,
    # ConsistencyToolsClient,
    # PowerDynamicsToolsClient,
    # UtilityToolsClient,
    # EvolutionToolsClient
)


class BrahminyKiteTextEnv(gym.Env):
    """Text-based environment for claim verification and learning."""
    
    metadata = {'render.modes': ['human', 'json']}
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__()
        
        self.config = config or {}
        self.max_text_length = self.config.get('max_text_length', 512)
        self.frameworks = self.config.get('frameworks', ['empirical'])
        
        # Define action and observation spaces
        # Action: text input (claim to verify)
        self.action_space = spaces.Dict({
            'claim': spaces.Box(
                low=0, high=255, 
                shape=(self.max_text_length,), 
                dtype=np.uint8
            ),
            'framework_weights': spaces.Box(
                low=0.0, high=1.0,
                shape=(len(self.frameworks),),
                dtype=np.float32
            )
        })
        
        # Observation: verification results from all frameworks
        self.observation_space = spaces.Dict({
            'claim': spaces.Box(
                low=0, high=255,
                shape=(self.max_text_length,),
                dtype=np.uint8
            ),
            'verifications': spaces.Box(
                low=-1.0, high=1.0,
                shape=(len(self.frameworks),),
                dtype=np.float32
            ),
            'confidence': spaces.Box(
                low=0.0, high=1.0,
                shape=(len(self.frameworks),),
                dtype=np.float32
            ),
            'metadata': spaces.Box(
                low=0, high=255,
                shape=(1024,),  # JSON metadata
                dtype=np.uint8
            )
        })
        
        # Initialize tool clients
        self._init_clients()
        
        # State tracking
        self.current_claim = ""
        self.verification_history = []
        self.episode_reward = 0.0
        self.steps = 0
    
    def _init_clients(self):
        """Initialize tool clients."""
        self.clients = {}
        
        if 'empirical' in self.frameworks:
            self.clients['empirical'] = EmpiricalToolsClient()
        
        # Add other clients as implemented
        # if 'contextual' in self.frameworks:
        #     self.clients['contextual'] = ContextualToolsClient()
    
    def _text_to_array(self, text: str) -> np.ndarray:
        """Convert text to fixed-size numpy array."""
        encoded = text.encode('utf-8')[:self.max_text_length]
        arr = np.zeros(self.max_text_length, dtype=np.uint8)
        arr[:len(encoded)] = list(encoded)
        return arr
    
    def _array_to_text(self, arr: np.ndarray) -> str:
        """Convert numpy array back to text."""
        # Remove padding zeros
        valid_bytes = arr[arr != 0].tobytes()
        return valid_bytes.decode('utf-8', errors='ignore')
    
    def reset(self) -> Dict[str, np.ndarray]:
        """Reset environment to initial state."""
        self.current_claim = ""
        self.verification_history = []
        self.episode_reward = 0.0
        self.steps = 0
        
        # Return empty initial observation
        return {
            'claim': np.zeros(self.max_text_length, dtype=np.uint8),
            'verifications': np.zeros(len(self.frameworks), dtype=np.float32),
            'confidence': np.zeros(len(self.frameworks), dtype=np.float32),
            'metadata': np.zeros(1024, dtype=np.uint8)
        }
    
    def step(self, action: Dict[str, np.ndarray]) -> Tuple[Dict[str, np.ndarray], float, bool, Dict]:
        """Execute verification step."""
        self.steps += 1
        
        # Extract claim from action
        claim = self._array_to_text(action['claim'])
        framework_weights = action.get('framework_weights', np.ones(len(self.frameworks)))
        
        # Normalize weights
        framework_weights = framework_weights / np.sum(framework_weights)
        
        # Verify claim with each framework
        verifications = []
        confidences = []
        metadata = {}
        
        for i, framework in enumerate(self.frameworks):
            if framework_weights[i] > 0.01:  # Skip if weight too low
                result = self._verify_with_framework(claim, framework)
                verifications.append(result['score'])
                confidences.append(result['confidence'])
                metadata[framework] = result['metadata']
            else:
                verifications.append(0.0)
                confidences.append(0.0)
        
        # Calculate reward
        reward = self._calculate_reward(
            verifications, confidences, framework_weights
        )
        self.episode_reward += reward
        
        # Create observation
        observation = {
            'claim': self._text_to_array(claim),
            'verifications': np.array(verifications, dtype=np.float32),
            'confidence': np.array(confidences, dtype=np.float32),
            'metadata': self._text_to_array(json.dumps(metadata)[:1024])
        }
        
        # Check if episode is done
        done = self.steps >= self.config.get('max_steps', 100)
        
        # Info dictionary
        info = {
            'claim': claim,
            'framework_results': dict(zip(self.frameworks, verifications)),
            'episode_reward': self.episode_reward,
            'steps': self.steps
        }
        
        # Store in history
        self.verification_history.append({
            'claim': claim,
            'verifications': verifications,
            'reward': reward
        })
        
        return observation, reward, done, info
    
    def _verify_with_framework(self, claim: str, framework: str) -> Dict[str, Any]:
        """Verify claim with specific framework."""
        client = self.clients.get(framework)
        if not client:
            return {'score': 0.0, 'confidence': 0.0, 'metadata': {}}
        
        try:
            if framework == 'empirical':
                # Convert claim to logical formula for Z3
                # This is simplified - real implementation would parse claim
                result = client.check_logical_consistency(
                    f"(claim \"{claim}\")"
                )
                
                score = 1.0 if result.get('satisfiable', False) else -1.0
                confidence = 0.9 if 'error' not in result else 0.1
                
                return {
                    'score': score,
                    'confidence': confidence,
                    'metadata': result
                }
            
            # Add other framework implementations
            
        except Exception as e:
            return {
                'score': 0.0,
                'confidence': 0.0,
                'metadata': {'error': str(e)}
            }
    
    def _calculate_reward(self, verifications: List[float], 
                         confidences: List[float],
                         weights: np.ndarray) -> float:
        """Calculate reward based on verification results."""
        # Weighted average of verification scores
        weighted_score = np.sum(
            np.array(verifications) * np.array(confidences) * weights
        )
        
        # Bonus for high confidence
        confidence_bonus = np.mean(confidences) * 0.1
        
        # Penalty for disagreement between frameworks
        if len(verifications) > 1:
            disagreement = np.std(verifications)
            disagreement_penalty = -disagreement * 0.2
        else:
            disagreement_penalty = 0.0
        
        return weighted_score + confidence_bonus + disagreement_penalty
    
    def render(self, mode='human'):
        """Render the environment."""
        if mode == 'human':
            print(f"\n--- Step {self.steps} ---")
            print(f"Claim: {self.current_claim}")
            if self.verification_history:
                latest = self.verification_history[-1]
                for framework, score in zip(self.frameworks, latest['verifications']):
                    print(f"{framework}: {score:.2f}")
                print(f"Reward: {latest['reward']:.3f}")
        
        elif mode == 'json':
            return json.dumps({
                'steps': self.steps,
                'claim': self.current_claim,
                'history': self.verification_history,
                'episode_reward': self.episode_reward
            })
    
    def close(self):
        """Clean up resources."""
        for client in self.clients.values():
            if hasattr(client, 'close'):
                client.close()
    
    def seed(self, seed=None):
        """Set random seed."""
        np.random.seed(seed)
        return [seed]


class AsyncBrahminyKiteEnv(BrahminyKiteTextEnv):
    """Async version for parallel framework execution."""
    
    async def async_step(self, action: Dict[str, np.ndarray]) -> Tuple[Dict[str, np.ndarray], float, bool, Dict]:
        """Execute verification step asynchronously."""
        self.steps += 1
        
        # Extract claim from action
        claim = self._array_to_text(action['claim'])
        framework_weights = action.get('framework_weights', np.ones(len(self.frameworks)))
        
        # Normalize weights
        framework_weights = framework_weights / np.sum(framework_weights)
        
        # Verify claim with all frameworks in parallel
        tasks = []
        active_frameworks = []
        
        for i, framework in enumerate(self.frameworks):
            if framework_weights[i] > 0.01:
                tasks.append(self._async_verify(claim, framework))
                active_frameworks.append((i, framework))
        
        # Wait for all verifications
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        verifications = np.zeros(len(self.frameworks))
        confidences = np.zeros(len(self.frameworks))
        metadata = {}
        
        for (i, framework), result in zip(active_frameworks, results):
            if isinstance(result, Exception):
                verifications[i] = 0.0
                confidences[i] = 0.0
                metadata[framework] = {'error': str(result)}
            else:
                verifications[i] = result['score']
                confidences[i] = result['confidence']
                metadata[framework] = result['metadata']
        
        # Calculate reward
        reward = self._calculate_reward(
            verifications.tolist(), confidences.tolist(), framework_weights
        )
        self.episode_reward += reward
        
        # Create observation
        observation = {
            'claim': self._text_to_array(claim),
            'verifications': verifications.astype(np.float32),
            'confidence': confidences.astype(np.float32),
            'metadata': self._text_to_array(json.dumps(metadata)[:1024])
        }
        
        # Check if episode is done
        done = self.steps >= self.config.get('max_steps', 100)
        
        # Info dictionary
        info = {
            'claim': claim,
            'framework_results': dict(zip(self.frameworks, verifications)),
            'episode_reward': self.episode_reward,
            'steps': self.steps
        }
        
        return observation, reward, done, info
    
    async def _async_verify(self, claim: str, framework: str) -> Dict[str, Any]:
        """Async wrapper for framework verification."""
        return await asyncio.to_thread(
            self._verify_with_framework, claim, framework
        )