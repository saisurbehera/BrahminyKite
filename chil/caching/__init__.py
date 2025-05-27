"""
Redis caching implementation for BrahminyKite.

Provides high-performance caching, session management, and pub/sub capabilities.
"""

from .config import RedisConfig, CacheConfig
from .client import RedisClient, create_redis_client, get_redis_client
from .decorators import (
    cache,
    cache_async,
    invalidate_cache,
    cache_key,
    ttl_cache,
    conditional_cache
)
from .session import RedisSessionManager, Session
from .pubsub import RedisPubSub, MessageHandler
from .patterns import (
    CachePattern,
    WriteThrough,
    WriteBehind,
    RefreshAhead,
    CacheAside
)

__all__ = [
    # Configuration
    "RedisConfig",
    "CacheConfig",
    
    # Client
    "RedisClient",
    "create_redis_client",
    "get_redis_client",
    
    # Decorators
    "cache",
    "cache_async",
    "invalidate_cache",
    "cache_key",
    "ttl_cache",
    "conditional_cache",
    
    # Session Management
    "RedisSessionManager",
    "Session",
    
    # Pub/Sub
    "RedisPubSub",
    "MessageHandler",
    
    # Patterns
    "CachePattern",
    "WriteThrough",
    "WriteBehind",
    "RefreshAhead",
    "CacheAside",
]