"""
Redis-based session management for distributed applications.
"""

import asyncio
import json
import logging
import secrets
from typing import Dict, Any, Optional, List, Set, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
import hashlib

from .client import RedisClient, get_redis_client

logger = logging.getLogger(__name__)


@dataclass
class Session:
    """Session data structure."""
    
    session_id: str
    user_id: Optional[str] = None
    data: Dict[str, Any] = None
    created_at: datetime = None
    last_accessed: datetime = None
    expires_at: Optional[datetime] = None
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    is_authenticated: bool = False
    roles: List[str] = None
    
    def __post_init__(self):
        if self.data is None:
            self.data = {}
        if self.roles is None:
            self.roles = []
        if self.created_at is None:
            self.created_at = datetime.now()
        if self.last_accessed is None:
            self.last_accessed = datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert session to dictionary."""
        data = asdict(self)
        # Convert datetime objects to ISO strings
        for key, value in data.items():
            if isinstance(value, datetime):
                data[key] = value.isoformat()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Session":
        """Create session from dictionary."""
        # Convert ISO strings back to datetime objects
        datetime_fields = ["created_at", "last_accessed", "expires_at"]
        for field in datetime_fields:
            if field in data and data[field]:
                if isinstance(data[field], str):
                    data[field] = datetime.fromisoformat(data[field])
        
        return cls(**data)
    
    def is_expired(self) -> bool:
        """Check if session is expired."""
        return self.expires_at and datetime.now() > self.expires_at
    
    def touch(self):
        """Update last accessed time."""
        self.last_accessed = datetime.now()
    
    def set_expiry(self, ttl_seconds: int):
        """Set session expiry."""
        self.expires_at = datetime.now() + timedelta(seconds=ttl_seconds)


class RedisSessionManager:
    """
    Redis-based session management with advanced features.
    
    Features:
    - Distributed session storage
    - Session expiration and cleanup
    - Security features (IP validation, etc.)
    - Session data encryption
    - Concurrent session limits
    - Session analytics
    """
    
    def __init__(
        self,
        client: Optional[RedisClient] = None,
        default_ttl: int = 3600,  # 1 hour
        key_prefix: str = "session",
        encrypt_data: bool = False,
        max_sessions_per_user: int = 5,
        validate_ip: bool = False
    ):
        self.client = client
        self.default_ttl = default_ttl
        self.key_prefix = key_prefix
        self.encrypt_data = encrypt_data
        self.max_sessions_per_user = max_sessions_per_user
        self.validate_ip = validate_ip
        
        # Encryption key (in production, load from secure config)
        self._encryption_key = None
        if encrypt_data:
            self._setup_encryption()
        
        # Statistics
        self._stats = {
            "sessions_created": 0,
            "sessions_destroyed": 0,
            "sessions_expired": 0,
            "invalid_access_attempts": 0
        }
    
    async def initialize(self):
        """Initialize the session manager."""
        if self.client is None:
            self.client = await get_redis_client()
        
        # Start cleanup task
        asyncio.create_task(self._cleanup_expired_sessions())
        
        logger.info("Redis session manager initialized")
    
    def _setup_encryption(self):
        """Set up encryption for session data."""
        try:
            from cryptography.fernet import Fernet
            self._fernet = Fernet(Fernet.generate_key())
            logger.info("Session encryption enabled")
        except ImportError:
            logger.warning("cryptography package not available, disabling encryption")
            self.encrypt_data = False
    
    def _encrypt_data(self, data: str) -> str:
        """Encrypt session data."""
        if not self.encrypt_data or not hasattr(self, '_fernet'):
            return data
        
        return self._fernet.encrypt(data.encode()).decode()
    
    def _decrypt_data(self, encrypted_data: str) -> str:
        """Decrypt session data."""
        if not self.encrypt_data or not hasattr(self, '_fernet'):
            return encrypted_data
        
        return self._fernet.decrypt(encrypted_data.encode()).decode()
    
    def _make_session_key(self, session_id: str) -> str:
        """Create Redis key for session."""
        return f"{self.key_prefix}:{session_id}"
    
    def _make_user_sessions_key(self, user_id: str) -> str:
        """Create Redis key for user's active sessions."""
        return f"{self.key_prefix}:user:{user_id}"
    
    def generate_session_id(self) -> str:
        """Generate a secure session ID."""
        return secrets.token_urlsafe(32)
    
    async def create_session(
        self,
        user_id: Optional[str] = None,
        data: Optional[Dict[str, Any]] = None,
        ttl: Optional[int] = None,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
        roles: Optional[List[str]] = None
    ) -> Session:
        """
        Create a new session.
        
        Args:
            user_id: User ID for authenticated sessions
            data: Initial session data
            ttl: Time to live in seconds
            ip_address: Client IP address
            user_agent: Client user agent
            roles: User roles
        
        Returns:
            Created session
        """
        if not self.client:
            await self.initialize()
        
        # Generate session ID
        session_id = self.generate_session_id()
        
        # Check concurrent session limit
        if user_id and self.max_sessions_per_user > 0:
            await self._enforce_session_limit(user_id)
        
        # Create session
        session = Session(
            session_id=session_id,
            user_id=user_id,
            data=data or {},
            ip_address=ip_address,
            user_agent=user_agent,
            is_authenticated=user_id is not None,
            roles=roles or []
        )
        
        # Set expiry
        ttl = ttl or self.default_ttl
        session.set_expiry(ttl)
        
        # Store session
        await self._store_session(session, ttl)
        
        # Track user sessions
        if user_id:
            await self._add_user_session(user_id, session_id, ttl)
        
        self._stats["sessions_created"] += 1
        logger.info(f"Created session {session_id} for user {user_id}")
        
        return session
    
    async def get_session(
        self,
        session_id: str,
        validate_ip_address: Optional[str] = None
    ) -> Optional[Session]:
        """
        Get session by ID.
        
        Args:
            session_id: Session ID
            validate_ip_address: IP address to validate against
        
        Returns:
            Session object or None if not found/invalid
        """
        if not self.client:
            await self.initialize()
        
        session_key = self._make_session_key(session_id)
        
        # Get session data
        session_data = await self.client.get(session_key)
        if not session_data:
            return None
        
        try:
            # Decrypt if needed
            if self.encrypt_data:
                session_data = self._decrypt_data(session_data)
            
            # Parse session
            if isinstance(session_data, str):
                session_data = json.loads(session_data)
            
            session = Session.from_dict(session_data)
            
            # Validate session
            if session.is_expired():
                await self.destroy_session(session_id)
                self._stats["sessions_expired"] += 1
                return None
            
            # Validate IP if required
            if self.validate_ip and validate_ip_address:
                if session.ip_address != validate_ip_address:
                    self._stats["invalid_access_attempts"] += 1
                    logger.warning(f"IP mismatch for session {session_id}: {session.ip_address} != {validate_ip_address}")
                    return None
            
            # Update last accessed
            session.touch()
            await self._store_session(session, self.default_ttl)
            
            return session
        
        except Exception as e:
            logger.error(f"Error loading session {session_id}: {e}")
            return None
    
    async def update_session(
        self,
        session_id: str,
        data: Optional[Dict[str, Any]] = None,
        extend_ttl: Optional[int] = None
    ) -> bool:
        """
        Update session data.
        
        Args:
            session_id: Session ID
            data: Data to update (merged with existing)
            extend_ttl: Extend TTL by this many seconds
        
        Returns:
            True if updated successfully
        """
        session = await self.get_session(session_id)
        if not session:
            return False
        
        # Update data
        if data:
            session.data.update(data)
        
        # Extend TTL
        if extend_ttl:
            current_ttl = await self.client.ttl(self._make_session_key(session_id))
            new_ttl = max(current_ttl, 0) + extend_ttl
            session.set_expiry(new_ttl)
        else:
            new_ttl = self.default_ttl
        
        # Store updated session
        await self._store_session(session, new_ttl)
        
        return True
    
    async def destroy_session(self, session_id: str) -> bool:
        """
        Destroy a session.
        
        Args:
            session_id: Session ID to destroy
        
        Returns:
            True if destroyed successfully
        """
        if not self.client:
            await self.initialize()
        
        # Get session to find user ID
        session = await self.get_session(session_id)
        
        # Delete session
        session_key = self._make_session_key(session_id)
        deleted = await self.client.delete(session_key)
        
        # Remove from user sessions
        if session and session.user_id:
            await self._remove_user_session(session.user_id, session_id)
        
        if deleted:
            self._stats["sessions_destroyed"] += 1
            logger.info(f"Destroyed session {session_id}")
        
        return bool(deleted)
    
    async def destroy_user_sessions(
        self,
        user_id: str,
        except_session: Optional[str] = None
    ) -> int:
        """
        Destroy all sessions for a user.
        
        Args:
            user_id: User ID
            except_session: Session ID to preserve
        
        Returns:
            Number of sessions destroyed
        """
        user_sessions = await self.get_user_sessions(user_id)
        destroyed = 0
        
        for session_id in user_sessions:
            if session_id != except_session:
                if await self.destroy_session(session_id):
                    destroyed += 1
        
        logger.info(f"Destroyed {destroyed} sessions for user {user_id}")
        return destroyed
    
    async def get_user_sessions(self, user_id: str) -> List[str]:
        """Get all active session IDs for a user."""
        if not self.client:
            await self.initialize()
        
        user_sessions_key = self._make_user_sessions_key(user_id)
        sessions = await self.client.get(user_sessions_key)
        
        if sessions:
            return sessions if isinstance(sessions, list) else []
        
        return []
    
    async def get_session_count(self, user_id: Optional[str] = None) -> int:
        """Get session count for user or total."""
        if user_id:
            sessions = await self.get_user_sessions(user_id)
            return len(sessions)
        else:
            # Count all sessions
            pattern = f"{self.key_prefix}:*"
            keys = await self.client.scan(match=pattern)
            # Filter out user session keys
            session_keys = [k for k in keys if not k.startswith(f"{self.key_prefix}:user:")]
            return len(session_keys)
    
    async def list_active_sessions(
        self,
        user_id: Optional[str] = None,
        limit: int = 100
    ) -> List[Session]:
        """List active sessions."""
        sessions = []
        
        if user_id:
            # Get user's sessions
            session_ids = await self.get_user_sessions(user_id)
            for session_id in session_ids[:limit]:
                session = await self.get_session(session_id)
                if session:
                    sessions.append(session)
        else:
            # Get all sessions
            pattern = f"{self.key_prefix}:*"
            keys = await self.client.scan(match=pattern)
            session_keys = [k for k in keys if not k.startswith(f"{self.key_prefix}:user:")][:limit]
            
            for key in session_keys:
                session_id = key.replace(f"{self.key_prefix}:", "")
                session = await self.get_session(session_id)
                if session:
                    sessions.append(session)
        
        return sessions
    
    async def _store_session(self, session: Session, ttl: int):
        """Store session in Redis."""
        session_key = self._make_session_key(session.session_id)
        session_data = json.dumps(session.to_dict())
        
        # Encrypt if needed
        if self.encrypt_data:
            session_data = self._encrypt_data(session_data)
        
        await self.client.set(session_key, session_data, ttl=ttl)
    
    async def _add_user_session(self, user_id: str, session_id: str, ttl: int):
        """Add session to user's session list."""
        user_sessions_key = self._make_user_sessions_key(user_id)
        
        # Get current sessions
        sessions = await self.get_user_sessions(user_id)
        
        # Add new session
        if session_id not in sessions:
            sessions.append(session_id)
        
        # Store updated list
        await self.client.set(user_sessions_key, sessions, ttl=ttl)
    
    async def _remove_user_session(self, user_id: str, session_id: str):
        """Remove session from user's session list."""
        user_sessions_key = self._make_user_sessions_key(user_id)
        
        # Get current sessions
        sessions = await self.get_user_sessions(user_id)
        
        # Remove session
        if session_id in sessions:
            sessions.remove(session_id)
            
            if sessions:
                # Update list
                await self.client.set(user_sessions_key, sessions, ttl=self.default_ttl)
            else:
                # Delete empty list
                await self.client.delete(user_sessions_key)
    
    async def _enforce_session_limit(self, user_id: str):
        """Enforce maximum sessions per user."""
        sessions = await self.get_user_sessions(user_id)
        
        if len(sessions) >= self.max_sessions_per_user:
            # Destroy oldest sessions
            excess = len(sessions) - self.max_sessions_per_user + 1
            
            # Get session objects to find oldest
            session_objects = []
            for session_id in sessions:
                session = await self.get_session(session_id)
                if session:
                    session_objects.append(session)
            
            # Sort by creation time
            session_objects.sort(key=lambda s: s.created_at)
            
            # Destroy oldest sessions
            for session in session_objects[:excess]:
                await self.destroy_session(session.session_id)
                logger.info(f"Destroyed old session {session.session_id} due to limit")
    
    async def _cleanup_expired_sessions(self):
        """Background task to clean up expired sessions."""
        while True:
            try:
                await asyncio.sleep(300)  # Run every 5 minutes
                
                # Scan for session keys
                pattern = f"{self.key_prefix}:*"
                keys = await self.client.scan(match=pattern)
                session_keys = [k for k in keys if not k.startswith(f"{self.key_prefix}:user:")]
                
                expired_count = 0
                for key in session_keys:
                    ttl = await self.client.ttl(key)
                    if ttl == -2:  # Key doesn't exist
                        continue
                    elif ttl == -1:  # Key exists but no expiry
                        # Force expiry
                        await self.client.expire(key, self.default_ttl)
                    
                    # Check if session is expired
                    session_data = await self.client.get(key)
                    if session_data:
                        try:
                            if self.encrypt_data:
                                session_data = self._decrypt_data(session_data)
                            
                            if isinstance(session_data, str):
                                session_data = json.loads(session_data)
                            
                            session = Session.from_dict(session_data)
                            if session.is_expired():
                                await self.destroy_session(session.session_id)
                                expired_count += 1
                        except:
                            # Invalid session data, delete it
                            await self.client.delete(key)
                            expired_count += 1
                
                if expired_count > 0:
                    logger.info(f"Cleaned up {expired_count} expired sessions")
                    self._stats["sessions_expired"] += expired_count
            
            except Exception as e:
                logger.error(f"Error in session cleanup: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get session statistics."""
        return self._stats.copy()
    
    def reset_stats(self):
        """Reset session statistics."""
        self._stats = {
            "sessions_created": 0,
            "sessions_destroyed": 0,
            "sessions_expired": 0,
            "invalid_access_attempts": 0
        }


# Global session manager
_session_manager: Optional[RedisSessionManager] = None


async def get_session_manager() -> RedisSessionManager:
    """Get the global session manager."""
    global _session_manager
    
    if _session_manager is None:
        _session_manager = RedisSessionManager()
        await _session_manager.initialize()
    
    return _session_manager