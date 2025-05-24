"""
FastAPI endpoints for BrahminyKite text environment.
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any
import numpy as np
import asyncio
import uuid
from datetime import datetime

from .text_environment import BrahminyKiteTextEnv, AsyncBrahminyKiteEnv


# Pydantic models for API
class VerificationRequest(BaseModel):
    """Request model for claim verification."""
    claim: str = Field(..., description="Claim to verify", max_length=512)
    frameworks: Optional[List[str]] = Field(
        default=['empirical'],
        description="Frameworks to use for verification"
    )
    framework_weights: Optional[List[float]] = Field(
        default=None,
        description="Weights for each framework"
    )


class StepRequest(BaseModel):
    """Request model for environment step."""
    session_id: str = Field(..., description="Session ID")
    claim: str = Field(..., description="Claim to verify")
    framework_weights: Optional[List[float]] = Field(
        default=None,
        description="Weights for each framework"
    )


class ResetRequest(BaseModel):
    """Request model for environment reset."""
    session_id: Optional[str] = Field(
        default=None,
        description="Session ID (creates new if not provided)"
    )
    config: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Environment configuration"
    )


class VerificationResponse(BaseModel):
    """Response model for verification results."""
    claim: str
    verifications: Dict[str, float]
    confidences: Dict[str, float]
    reward: float
    metadata: Dict[str, Any]


class StepResponse(BaseModel):
    """Response model for environment step."""
    observation: Dict[str, Any]
    reward: float
    done: bool
    info: Dict[str, Any]


class SessionInfo(BaseModel):
    """Session information."""
    session_id: str
    created_at: datetime
    steps: int
    episode_reward: float
    active: bool


# FastAPI app
app = FastAPI(
    title="BrahminyKite Text Environment API",
    description="OpenAI Gym-compatible text environment for claim verification",
    version="0.1.0"
)

# Global session storage (in production, use Redis or similar)
sessions: Dict[str, AsyncBrahminyKiteEnv] = {}
session_info: Dict[str, SessionInfo] = {}


@app.on_event("startup")
async def startup_event():
    """Initialize services on startup."""
    print("BrahminyKite API starting up...")
    # Could start gRPC services here if needed


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    print("Shutting down...")
    for env in sessions.values():
        env.close()


@app.get("/")
async def root():
    """API root endpoint."""
    return {
        "name": "BrahminyKite Text Environment",
        "version": "0.1.0",
        "endpoints": [
            "/verify",
            "/env/reset",
            "/env/step",
            "/env/render",
            "/sessions"
        ]
    }


@app.post("/verify", response_model=VerificationResponse)
async def verify_claim(request: VerificationRequest):
    """One-shot claim verification without environment."""
    try:
        # Create temporary environment
        config = {
            'frameworks': request.frameworks,
            'max_steps': 1
        }
        env = AsyncBrahminyKiteEnv(config)
        
        # Prepare action
        claim_array = env._text_to_array(request.claim)
        
        if request.framework_weights:
            weights = np.array(request.framework_weights)
        else:
            weights = np.ones(len(request.frameworks))
        
        action = {
            'claim': claim_array,
            'framework_weights': weights
        }
        
        # Execute step
        obs, reward, done, info = await env.async_step(action)
        
        # Close environment
        env.close()
        
        # Format response
        framework_results = info['framework_results']
        
        return VerificationResponse(
            claim=request.claim,
            verifications=framework_results,
            confidences={
                framework: float(obs['confidence'][i])
                for i, framework in enumerate(request.frameworks)
            },
            reward=reward,
            metadata={
                'frameworks_used': request.frameworks,
                'weights': weights.tolist()
            }
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/env/reset", response_model=Dict[str, Any])
async def reset_environment(request: ResetRequest):
    """Reset or create a new environment session."""
    try:
        # Generate session ID if not provided
        session_id = request.session_id or str(uuid.uuid4())
        
        # Create or reset environment
        if session_id in sessions:
            env = sessions[session_id]
        else:
            config = request.config or {}
            env = AsyncBrahminyKiteEnv(config)
            sessions[session_id] = env
            session_info[session_id] = SessionInfo(
                session_id=session_id,
                created_at=datetime.now(),
                steps=0,
                episode_reward=0.0,
                active=True
            )
        
        # Reset environment
        observation = env.reset()
        
        # Convert numpy arrays to lists for JSON
        obs_dict = {
            key: value.tolist() if isinstance(value, np.ndarray) else value
            for key, value in observation.items()
        }
        
        return {
            'session_id': session_id,
            'observation': obs_dict,
            'info': {
                'frameworks': env.frameworks,
                'max_steps': env.config.get('max_steps', 100)
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/env/step", response_model=StepResponse)
async def step_environment(request: StepRequest):
    """Execute a step in the environment."""
    try:
        # Get environment
        if request.session_id not in sessions:
            raise HTTPException(
                status_code=404,
                detail=f"Session {request.session_id} not found"
            )
        
        env = sessions[request.session_id]
        
        # Prepare action
        claim_array = env._text_to_array(request.claim)
        
        if request.framework_weights:
            weights = np.array(request.framework_weights)
        else:
            weights = np.ones(len(env.frameworks))
        
        action = {
            'claim': claim_array,
            'framework_weights': weights
        }
        
        # Execute step
        obs, reward, done, info = await env.async_step(action)
        
        # Update session info
        session_info[request.session_id].steps = env.steps
        session_info[request.session_id].episode_reward = env.episode_reward
        
        if done:
            session_info[request.session_id].active = False
        
        # Convert numpy arrays to lists
        obs_dict = {
            key: value.tolist() if isinstance(value, np.ndarray) else value
            for key, value in obs.items()
        }
        
        return StepResponse(
            observation=obs_dict,
            reward=float(reward),
            done=done,
            info=info
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/env/render/{session_id}")
async def render_environment(session_id: str, mode: str = "json"):
    """Render the environment state."""
    try:
        if session_id not in sessions:
            raise HTTPException(
                status_code=404,
                detail=f"Session {session_id} not found"
            )
        
        env = sessions[session_id]
        
        if mode == "json":
            return JSONResponse(content=env.render(mode='json'))
        else:
            # For human mode, return formatted text
            env.render(mode='human')
            return {
                "rendered": "Check console output",
                "history": env.verification_history
            }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/sessions", response_model=List[SessionInfo])
async def list_sessions():
    """List all active sessions."""
    return list(session_info.values())


@app.delete("/sessions/{session_id}")
async def delete_session(session_id: str):
    """Delete a session and clean up resources."""
    try:
        if session_id not in sessions:
            raise HTTPException(
                status_code=404,
                detail=f"Session {session_id} not found"
            )
        
        # Close environment
        env = sessions[session_id]
        env.close()
        
        # Remove from storage
        del sessions[session_id]
        del session_info[session_id]
        
        return {"message": f"Session {session_id} deleted"}
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/batch/verify", response_model=List[VerificationResponse])
async def batch_verify(requests: List[VerificationRequest]):
    """Batch verification of multiple claims."""
    try:
        tasks = []
        for request in requests:
            tasks.append(verify_claim(request))
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter out exceptions
        responses = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                # Return error response
                responses.append(
                    VerificationResponse(
                        claim=requests[i].claim,
                        verifications={},
                        confidences={},
                        reward=0.0,
                        metadata={'error': str(result)}
                    )
                )
            else:
                responses.append(result)
        
        return responses
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "active_sessions": len(sessions),
        "timestamp": datetime.now().isoformat()
    }