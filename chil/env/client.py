"""
Python client for BrahminyKite API.
"""

import requests
from typing import Dict, List, Optional, Any
import json


class BrahminyKiteClient:
    """Client for interacting with BrahminyKite API."""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.session_id = None
    
    def verify(self, claim: str, frameworks: List[str] = None, 
               weights: List[float] = None) -> Dict[str, Any]:
        """One-shot claim verification."""
        payload = {
            "claim": claim,
            "frameworks": frameworks or ["empirical"],
            "framework_weights": weights
        }
        
        response = requests.post(
            f"{self.base_url}/verify",
            json=payload
        )
        response.raise_for_status()
        return response.json()
    
    def reset(self, config: Dict[str, Any] = None) -> Dict[str, Any]:
        """Reset environment and create new session."""
        payload = {
            "session_id": self.session_id,
            "config": config
        }
        
        response = requests.post(
            f"{self.base_url}/env/reset",
            json=payload
        )
        response.raise_for_status()
        
        result = response.json()
        self.session_id = result['session_id']
        return result
    
    def step(self, claim: str, weights: List[float] = None) -> Dict[str, Any]:
        """Execute environment step."""
        if not self.session_id:
            raise ValueError("No active session. Call reset() first.")
        
        payload = {
            "session_id": self.session_id,
            "claim": claim,
            "framework_weights": weights
        }
        
        response = requests.post(
            f"{self.base_url}/env/step",
            json=payload
        )
        response.raise_for_status()
        return response.json()
    
    def render(self, mode: str = "json") -> Dict[str, Any]:
        """Render current environment state."""
        if not self.session_id:
            raise ValueError("No active session.")
        
        response = requests.get(
            f"{self.base_url}/env/render/{self.session_id}",
            params={"mode": mode}
        )
        response.raise_for_status()
        return response.json()
    
    def list_sessions(self) -> List[Dict[str, Any]]:
        """List all sessions."""
        response = requests.get(f"{self.base_url}/sessions")
        response.raise_for_status()
        return response.json()
    
    def delete_session(self, session_id: str = None):
        """Delete a session."""
        sid = session_id or self.session_id
        if not sid:
            raise ValueError("No session ID provided.")
        
        response = requests.delete(f"{self.base_url}/sessions/{sid}")
        response.raise_for_status()
        
        if sid == self.session_id:
            self.session_id = None
    
    def batch_verify(self, claims: List[str], 
                    frameworks: List[str] = None) -> List[Dict[str, Any]]:
        """Batch verification of multiple claims."""
        requests_list = [
            {
                "claim": claim,
                "frameworks": frameworks or ["empirical"]
            }
            for claim in claims
        ]
        
        response = requests.post(
            f"{self.base_url}/batch/verify",
            json=requests_list
        )
        response.raise_for_status()
        return response.json()


# Example usage script
if __name__ == "__main__":
    # Create client
    client = BrahminyKiteClient()
    
    # One-shot verification
    result = client.verify(
        "Water boils at 100 degrees Celsius at sea level",
        frameworks=["empirical"]
    )
    print("Verification result:", json.dumps(result, indent=2))
    
    # Environment interaction
    # Reset environment
    reset_result = client.reset(config={
        "frameworks": ["empirical"],
        "max_steps": 10
    })
    print(f"\nCreated session: {client.session_id}")
    
    # Take steps
    claims = [
        "The Earth is flat",
        "2 + 2 = 4",
        "All birds can fly"
    ]
    
    for claim in claims:
        step_result = client.step(claim)
        print(f"\nClaim: {claim}")
        print(f"Reward: {step_result['reward']:.3f}")
        print(f"Done: {step_result['done']}")
    
    # Render final state
    final_state = client.render()
    print("\nFinal state:", json.dumps(final_state, indent=2))
    
    # Cleanup
    client.delete_session()
    print("\nSession deleted")