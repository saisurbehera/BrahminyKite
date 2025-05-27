"""
Redis client implementation with connection management.
"""

import asyncio
import logging
from typing import Optional, Any, Dict, List, Union, Callable, Type
from contextlib import asynccontextmanager
import time

import redis.asyncio as redis
from redis.asyncio import ConnectionPool, Redis
from redis.exceptions import RedisError, ConnectionError, TimeoutError

from .config import RedisConfig, CacheConfig, validate_redis_config
from .serializers import get_serializer, Serializer

logger = logging.getLogger(__name__)


class RedisClient:
    """
    Enhanced Redis client with connection pooling and monitoring.
    
    Features:
    - Connection pooling
    - Automatic retries
    - Health monitoring
    - Metrics collection
    - Serialization support
    """
    
    def __init__(
        self,
        redis_config: RedisConfig,
        cache_config: Optional[CacheConfig] = None
    ):
        validate_redis_config(redis_config)
        
        self.redis_config = redis_config
        self.cache_config = cache_config or CacheConfig()
        
        # Create connection pool
        self._pool: Optional[ConnectionPool] = None
        self._client: Optional[Redis] = None
        
        # Serializer
        self._serializer: Serializer = get_serializer(self.cache_config.serializer)
        
        # Health tracking
        self._last_health_check: Optional[float] = None
        self._is_healthy: bool = True
        
        # Metrics
        self._metrics = {
            "hits": 0,
            "misses": 0,
            "errors": 0,
            "total_ops": 0,
            "total_latency": 0.0
        }
    
    async def initialize(self) -> None:
        """Initialize Redis connection pool."""
        if self._pool is not None:
            return
        
        logger.info(f"Initializing Redis connection to {self.redis_config.host}:{self.redis_config.port}")
        
        # Create connection pool
        pool_kwargs = {
            **self.redis_config.to_redis_kwargs(),
            "connection_class": redis.AsyncConnection,
        }
        
        self._pool = ConnectionPool(**pool_kwargs)
        self._client = Redis(connection_pool=self._pool)
        
        # Test connection
        try:
            await self._client.ping()
            logger.info("Redis connection established successfully")
            self._is_healthy = True
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            self._is_healthy = False
            raise
    
    async def close(self) -> None:
        """Close Redis connection pool."""
        if self._client:
            await self._client.close()
            self._client = None
        
        if self._pool:
            await self._pool.disconnect()
            self._pool = None
        
        logger.info("Redis connections closed")
    
    @property
    def client(self) -> Redis:
        """Get Redis client instance."""
        if self._client is None:
            raise RuntimeError("Redis client not initialized")
        return self._client
    
    async def _execute_with_retry(
        self,
        operation: Callable,
        *args,
        **kwargs
    ) -> Any:
        """Execute operation with retry logic."""
        last_error = None
        
        for attempt in range(self.redis_config.max_retry_attempts):
            try:
                start_time = time.time()
                result = await operation(*args, **kwargs)
                
                # Update metrics
                self._metrics["total_ops"] += 1
                self._metrics["total_latency"] += time.time() - start_time
                
                return result
            
            except (ConnectionError, TimeoutError) as e:
                last_error = e
                if attempt < self.redis_config.max_retry_attempts - 1:
                    wait_time = (2 ** attempt) * 0.1  # Exponential backoff
                    logger.warning(f"Redis operation failed, retrying in {wait_time}s: {e}")
                    await asyncio.sleep(wait_time)
                else:
                    self._metrics["errors"] += 1
                    raise
            
            except Exception as e:
                self._metrics["errors"] += 1
                if self.cache_config.log_errors:
                    logger.error(f"Redis operation failed: {e}")
                
                if self.cache_config.raise_on_error:
                    raise
                return None
        
        raise last_error
    
    # Core cache operations
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        full_key = self._make_key(key)
        
        value = await self._execute_with_retry(
            self.client.get,
            full_key
        )
        
        if value is None:
            self._metrics["misses"] += 1
            return None
        
        self._metrics["hits"] += 1
        
        # Deserialize
        try:
            return self._serializer.deserialize(value)
        except Exception as e:
            logger.error(f"Failed to deserialize value for key {key}: {e}")
            return None
    
    async def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None
    ) -> bool:
        """Set value in cache."""
        full_key = self._make_key(key)
        ttl = ttl or self.cache_config.default_ttl
        
        # Serialize
        try:
            serialized = self._serializer.serialize(value)
        except Exception as e:
            logger.error(f"Failed to serialize value for key {key}: {e}")
            return False
        
        # Compress if needed
        if self.cache_config.compression and len(serialized) > self.cache_config.compression_threshold:
            import zlib
            serialized = zlib.compress(serialized)
        
        result = await self._execute_with_retry(
            self.client.set,
            full_key,
            serialized,
            ex=ttl
        )
        
        return bool(result)
    
    async def delete(self, key: Union[str, List[str]]) -> int:
        """Delete key(s) from cache."""
        if isinstance(key, str):
            keys = [self._make_key(key)]
        else:
            keys = [self._make_key(k) for k in key]
        
        return await self._execute_with_retry(
            self.client.delete,
            *keys
        )
    
    async def exists(self, key: Union[str, List[str]]) -> int:
        """Check if key(s) exist."""
        if isinstance(key, str):
            keys = [self._make_key(key)]
        else:
            keys = [self._make_key(k) for k in key]
        
        return await self._execute_with_retry(
            self.client.exists,
            *keys
        )
    
    async def expire(self, key: str, ttl: int) -> bool:
        """Set TTL for a key."""
        full_key = self._make_key(key)
        
        result = await self._execute_with_retry(
            self.client.expire,
            full_key,
            ttl
        )
        
        return bool(result)
    
    async def ttl(self, key: str) -> int:
        """Get TTL for a key."""
        full_key = self._make_key(key)
        
        return await self._execute_with_retry(
            self.client.ttl,
            full_key
        )
    
    # Batch operations
    
    async def mget(self, keys: List[str]) -> Dict[str, Any]:
        """Get multiple values."""
        full_keys = [self._make_key(k) for k in keys]
        
        values = await self._execute_with_retry(
            self.client.mget,
            full_keys
        )
        
        result = {}
        for key, value in zip(keys, values):
            if value is not None:
                try:
                    result[key] = self._serializer.deserialize(value)
                    self._metrics["hits"] += 1
                except Exception as e:
                    logger.error(f"Failed to deserialize value for key {key}: {e}")
                    self._metrics["misses"] += 1
            else:
                self._metrics["misses"] += 1
        
        return result
    
    async def mset(self, mapping: Dict[str, Any], ttl: Optional[int] = None) -> bool:
        """Set multiple values."""
        ttl = ttl or self.cache_config.default_ttl
        
        # Serialize values
        serialized_mapping = {}
        for key, value in mapping.items():
            full_key = self._make_key(key)
            try:
                serialized_mapping[full_key] = self._serializer.serialize(value)
            except Exception as e:
                logger.error(f"Failed to serialize value for key {key}: {e}")
                continue
        
        if not serialized_mapping:
            return False
        
        # Use pipeline for atomic operation with TTL
        async with self.client.pipeline(transaction=True) as pipe:
            pipe.mset(serialized_mapping)
            
            if ttl:
                for key in serialized_mapping:
                    pipe.expire(key, ttl)
            
            results = await pipe.execute()
        
        return all(results)
    
    # Pattern operations
    
    async def keys(self, pattern: str) -> List[str]:
        """Get keys matching pattern."""
        full_pattern = self._make_key(pattern)
        
        keys = await self._execute_with_retry(
            self.client.keys,
            full_pattern
        )
        
        # Strip prefix
        prefix_len = len(self._make_key(""))
        return [k[prefix_len:] for k in keys]
    
    async def scan(
        self,
        match: Optional[str] = None,
        count: int = 100
    ) -> List[str]:
        """Scan keys incrementally."""
        full_match = self._make_key(match) if match else None
        
        keys = []
        cursor = 0
        
        while True:
            cursor, batch = await self._execute_with_retry(
                self.client.scan,
                cursor=cursor,
                match=full_match,
                count=count
            )
            
            keys.extend(batch)
            
            if cursor == 0:
                break
        
        # Strip prefix
        prefix_len = len(self._make_key(""))
        return [k[prefix_len:] for k in keys]
    
    # Health and monitoring
    
    async def health_check(self) -> Dict[str, Any]:
        """Check Redis health."""
        try:
            start_time = time.time()
            
            # Ping
            await self.client.ping()
            ping_latency = time.time() - start_time
            
            # Get info
            info = await self.client.info()
            
            # Get memory info
            memory_info = await self.client.info("memory")
            
            self._is_healthy = True
            self._last_health_check = time.time()
            
            return {
                "healthy": True,
                "ping_latency_ms": ping_latency * 1000,
                "version": info.get("redis_version"),
                "connected_clients": info.get("connected_clients"),
                "used_memory_mb": memory_info.get("used_memory") / 1024 / 1024 if memory_info.get("used_memory") else 0,
                "used_memory_peak_mb": memory_info.get("used_memory_peak") / 1024 / 1024 if memory_info.get("used_memory_peak") else 0,
                "uptime_days": info.get("uptime_in_seconds", 0) / 86400,
                "metrics": self.get_metrics()
            }
        
        except Exception as e:
            self._is_healthy = False
            return {
                "healthy": False,
                "error": str(e),
                "metrics": self.get_metrics()
            }
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get client metrics."""
        total_requests = self._metrics["hits"] + self._metrics["misses"]
        hit_rate = self._metrics["hits"] / total_requests if total_requests > 0 else 0
        avg_latency = self._metrics["total_latency"] / self._metrics["total_ops"] if self._metrics["total_ops"] > 0 else 0
        
        return {
            "hits": self._metrics["hits"],
            "misses": self._metrics["misses"],
            "errors": self._metrics["errors"],
            "hit_rate": hit_rate,
            "total_operations": self._metrics["total_ops"],
            "avg_latency_ms": avg_latency * 1000
        }
    
    def reset_metrics(self) -> None:
        """Reset client metrics."""
        self._metrics = {
            "hits": 0,
            "misses": 0,
            "errors": 0,
            "total_ops": 0,
            "total_latency": 0.0
        }
    
    # Utility methods
    
    def _make_key(self, key: str) -> str:
        """Create full cache key with prefix and version."""
        parts = [self.cache_config.key_prefix]
        
        if self.cache_config.include_version:
            parts.append(self.cache_config.version)
        
        parts.append(key)
        
        return self.cache_config.key_separator.join(parts)
    
    # Context manager support
    
    async def __aenter__(self):
        await self.initialize()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()


# Global client instance
_redis_client: Optional[RedisClient] = None
_lock = asyncio.Lock()


async def create_redis_client(
    redis_config: Optional[RedisConfig] = None,
    cache_config: Optional[CacheConfig] = None
) -> RedisClient:
    """
    Create or get the global Redis client.
    
    Args:
        redis_config: Redis configuration
        cache_config: Cache configuration
    
    Returns:
        Initialized Redis client
    """
    global _redis_client
    
    async with _lock:
        if _redis_client is None:
            if redis_config is None:
                redis_config = RedisConfig.from_env()
            
            if cache_config is None:
                cache_config = CacheConfig.from_env()
            
            _redis_client = RedisClient(redis_config, cache_config)
            await _redis_client.initialize()
        
        return _redis_client


async def get_redis_client() -> RedisClient:
    """Get the global Redis client."""
    if _redis_client is None:
        raise RuntimeError("Redis client not initialized. Call create_redis_client first.")
    return _redis_client


async def close_redis_client() -> None:
    """Close the global Redis client."""
    global _redis_client
    
    async with _lock:
        if _redis_client:
            await _redis_client.close()
            _redis_client = None