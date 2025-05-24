# BrahminyKite Text Environment

OpenAI Gym-compatible text environment with FastAPI interface for claim verification and training.

## Overview

This module provides:
- **Gym Environment**: Text-based environment for RL training
- **FastAPI Server**: RESTful API for environment interaction
- **Python Client**: Easy-to-use client library
- **Async Support**: Parallel framework execution

## Quick Start

### 1. Start the API Server

```bash
# Basic startup
python -m chil.env.run_server

# With options
python -m chil.env.run_server --host 0.0.0.0 --port 8000 --reload
```

### 2. Use the Python Client

```python
from chil.env.client import BrahminyKiteClient

# Create client
client = BrahminyKiteClient("http://localhost:8000")

# One-shot verification
result = client.verify(
    "The Earth orbits the Sun",
    frameworks=["empirical", "contextual"]
)
print(f"Verification: {result['verifications']}")
print(f"Reward: {result['reward']}")

# Environment interaction
client.reset()
for i in range(10):
    obs = client.step("Your claim here")
    if obs['done']:
        break
```

### 3. Direct Gym Usage

```python
from chil.env import BrahminyKiteTextEnv

# Create environment
env = BrahminyKiteTextEnv({
    'frameworks': ['empirical', 'contextual'],
    'max_steps': 100
})

# Reset
obs = env.reset()

# Step through environment
for _ in range(100):
    action = {
        'claim': env._text_to_array("2 + 2 = 4"),
        'framework_weights': np.array([0.7, 0.3])
    }
    obs, reward, done, info = env.step(action)
    if done:
        break
```

## API Endpoints

### Core Endpoints

#### POST `/verify`
One-shot claim verification without session management.

```json
{
  "claim": "Water freezes at 0°C",
  "frameworks": ["empirical"],
  "framework_weights": [1.0]
}
```

#### POST `/env/reset`
Create or reset an environment session.

```json
{
  "session_id": "optional-session-id",
  "config": {
    "frameworks": ["empirical", "contextual"],
    "max_steps": 100
  }
}
```

#### POST `/env/step`
Execute a step in the environment.

```json
{
  "session_id": "your-session-id",
  "claim": "All swans are white",
  "framework_weights": [0.5, 0.5]
}
```

#### GET `/env/render/{session_id}`
Render the current environment state.

#### POST `/batch/verify`
Batch verification of multiple claims.

```json
[
  {"claim": "Claim 1", "frameworks": ["empirical"]},
  {"claim": "Claim 2", "frameworks": ["contextual"]}
]
```

### Session Management

#### GET `/sessions`
List all active sessions.

#### DELETE `/sessions/{session_id}`
Delete a session and free resources.

## Environment Details

### Action Space
```python
{
    'claim': str,              # Text claim to verify (max 512 chars)
    'framework_weights': List[float]  # Weights for each framework
}
```

### Observation Space
```python
{
    'claim': str,              # The claim that was verified
    'verifications': Dict[str, float],  # Scores from each framework
    'confidence': Dict[str, float],     # Confidence levels
    'metadata': Dict           # Additional framework-specific data
}
```

### Reward Calculation
- Base reward: Weighted average of framework scores
- Confidence bonus: Higher confidence = higher reward
- Disagreement penalty: Penalizes conflicting framework results

## Training Example

```python
import gym
from stable_baselines3 import PPO
from chil.env import BrahminyKiteTextEnv

# Create environment
env = BrahminyKiteTextEnv({
    'frameworks': ['empirical', 'contextual', 'consistency'],
    'max_steps': 100
})

# Train with PPO
model = PPO("MultiInputPolicy", env, verbose=1)
model.learn(total_timesteps=10000)

# Evaluate
obs = env.reset()
for _ in range(100):
    action, _ = model.predict(obs)
    obs, reward, done, info = env.step(action)
    if done:
        break
```

## Configuration

### Environment Config
```python
{
    'frameworks': List[str],      # Frameworks to use
    'max_steps': int,            # Max steps per episode
    'max_text_length': int,      # Max claim length
    'reward_config': {           # Reward parameters
        'confidence_weight': float,
        'disagreement_penalty': float
    }
}
```

### Server Config
- Host: `--host` (default: 0.0.0.0)
- Port: `--port` (default: 8000)
- Workers: `--workers` (default: 1)
- Reload: `--reload` (development mode)

## Performance Considerations

1. **Async Execution**: Use `AsyncBrahminyKiteEnv` for parallel framework queries
2. **Batching**: Use `/batch/verify` for multiple claims
3. **Session Management**: Clean up inactive sessions to free memory
4. **Caching**: Framework results are cached in the underlying clients

## Error Handling

All API endpoints return standard HTTP status codes:
- `200`: Success
- `404`: Session not found
- `422`: Validation error
- `500`: Internal server error

Error responses include details:
```json
{
  "detail": "Error message here"
}
```