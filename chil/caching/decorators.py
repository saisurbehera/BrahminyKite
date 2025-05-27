"""
Caching decorators for easy cache integration.
"""

import asyncio
import functools
import hashlib
import inspect
import logging
from typing import Optional, Callable, Any, Union, List, Dict, Tuple
import time

from .client import get_redis_client, RedisClient

logger = logging.getLogger(__name__)


def _make_cache_key(
    func: Callable,
    args: tuple,
    kwargs: dict,
    key_prefix: Optional[str] = None,
    include_args: bool = True,
    include_kwargs: Optional[List[str]] = None
) -> str:
    """Generate cache key from function and arguments."""
    parts = []
    
    # Add custom prefix or use function name
    if key_prefix:
        parts.append(key_prefix)
    else:
        parts.append(f"{func.__module__}.{func.__name__}")
    
    # Add positional arguments
    if include_args and args:
        # Skip 'self' or 'cls' for methods
        if inspect.ismethod(func) or (args and hasattr(args[0], func.__name__)):
            args = args[1:]
        
        for arg in args:
            try:
                parts.append(str(arg))
            except:
                # Fallback to hash for non-stringable objects
                parts.append(hashlib.md5(repr(arg).encode()).hexdigest()[:8])
    
    # Add selected keyword arguments
    if include_kwargs and kwargs:
        for key in include_kwargs:
            if key in kwargs:
                try:
                    parts.append(f"{key}={kwargs[key]}")
                except:
                    parts.append(f"{key}={hashlib.md5(repr(kwargs[key]).encode()).hexdigest()[:8]}")
    
    return ":".join(parts)


def cache(
    ttl: Optional[int] = None,
    key_prefix: Optional[str] = None,
    condition: Optional[Callable[[Any], bool]] = None,
    include_args: bool = True,
    include_kwargs: Optional[List[str]] = None,
    cache_none: bool = False,
    cache_errors: bool = False
):
    """
    Cache decorator for synchronous functions.
    
    Args:
        ttl: Time to live in seconds
        key_prefix: Custom key prefix
        condition: Function to determine if result should be cached
        include_args: Include positional args in cache key
        include_kwargs: List of kwargs to include in cache key
        cache_none: Cache None results
        cache_errors: Cache exceptions
    
    Example:
        @cache(ttl=3600, key_prefix="user_data")
        def get_user(user_id: int):
            return fetch_user_from_db(user_id)
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # This is a sync decorator, so we need to run async operations
            # in a new event loop if one isn't running
            try:
                loop = asyncio.get_running_loop()
                # If we're in an async context, we can't use sync cache
                logger.warning(f"Sync cache decorator used in async context for {func.__name__}")
                return func(*args, **kwargs)
            except RuntimeError:
                # No event loop, create one
                return asyncio.run(_async_cache_wrapper(func, args, kwargs))
        
        async def _async_cache_wrapper(func, args, kwargs):
            redis_client = await get_redis_client()
            
            # Generate cache key
            cache_key = _make_cache_key(
                func, args, kwargs,
                key_prefix=key_prefix,
                include_args=include_args,
                include_kwargs=include_kwargs
            )
            
            # Try to get from cache
            try:
                cached_value = await redis_client.get(cache_key)
                if cached_value is not None:
                    logger.debug(f"Cache hit for {cache_key}")
                    
                    # Handle cached exceptions
                    if cache_errors and isinstance(cached_value, dict) and cached_value.get("__error__"):
                        raise Exception(cached_value["error_message"])
                    
                    return cached_value
            except Exception as e:
                logger.error(f"Cache get error for {cache_key}: {e}")
                # Continue to function execution
            
            # Execute function
            try:
                result = func(*args, **kwargs)
                
                # Check condition
                if condition and not condition(result):
                    return result
                
                # Check if we should cache None
                if result is None and not cache_none:
                    return result
                
                # Cache result
                try:
                    await redis_client.set(cache_key, result, ttl=ttl)
                    logger.debug(f"Cached result for {cache_key}")
                except Exception as e:
                    logger.error(f"Cache set error for {cache_key}: {e}")
                
                return result
            
            except Exception as e:
                if cache_errors:
                    # Cache the error
                    error_data = {
                        "__error__": True,
                        "error_type": type(e).__name__,
                        "error_message": str(e)
                    }
                    try:
                        await redis_client.set(cache_key, error_data, ttl=ttl or 300)
                    except:
                        pass
                raise
        
        return wrapper
    
    return decorator


def cache_async(
    ttl: Optional[int] = None,
    key_prefix: Optional[str] = None,
    condition: Optional[Callable[[Any], bool]] = None,
    include_args: bool = True,
    include_kwargs: Optional[List[str]] = None,
    cache_none: bool = False,
    cache_errors: bool = False,
    client: Optional[RedisClient] = None
):
    """
    Cache decorator for async functions.
    
    Args:
        ttl: Time to live in seconds
        key_prefix: Custom key prefix
        condition: Function to determine if result should be cached
        include_args: Include positional args in cache key
        include_kwargs: List of kwargs to include in cache key
        cache_none: Cache None results
        cache_errors: Cache exceptions
        client: Redis client instance (uses global if not provided)
    
    Example:
        @cache_async(ttl=3600, key_prefix="user_data")
        async def get_user(user_id: int):
            return await fetch_user_from_db(user_id)
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            redis_client = client or await get_redis_client()
            
            # Generate cache key
            cache_key = _make_cache_key(
                func, args, kwargs,
                key_prefix=key_prefix,
                include_args=include_args,
                include_kwargs=include_kwargs
            )
            
            # Try to get from cache
            try:
                cached_value = await redis_client.get(cache_key)
                if cached_value is not None:
                    logger.debug(f"Cache hit for {cache_key}")
                    
                    # Handle cached exceptions
                    if cache_errors and isinstance(cached_value, dict) and cached_value.get("__error__"):
                        raise Exception(cached_value["error_message"])
                    
                    return cached_value
            except Exception as e:
                logger.error(f"Cache get error for {cache_key}: {e}")
                # Continue to function execution
            
            # Execute function
            try:
                result = await func(*args, **kwargs)
                
                # Check condition
                if condition and not condition(result):
                    return result
                
                # Check if we should cache None
                if result is None and not cache_none:
                    return result
                
                # Cache result
                try:
                    await redis_client.set(cache_key, result, ttl=ttl)
                    logger.debug(f"Cached result for {cache_key}")
                except Exception as e:
                    logger.error(f"Cache set error for {cache_key}: {e}")
                
                return result
            
            except Exception as e:
                if cache_errors:
                    # Cache the error
                    error_data = {
                        "__error__": True,
                        "error_type": type(e).__name__,
                        "error_message": str(e)
                    }
                    try:
                        await redis_client.set(cache_key, error_data, ttl=ttl or 300)
                    except:
                        pass
                raise
        
        return wrapper
    
    return decorator


def invalidate_cache(
    pattern: Optional[str] = None,
    key_prefix: Optional[str] = None,
    exact_keys: Optional[List[str]] = None
):
    """
    Decorator to invalidate cache entries after function execution.
    
    Args:
        pattern: Pattern to match keys to invalidate
        key_prefix: Prefix for keys to invalidate
        exact_keys: Exact keys to invalidate
    
    Example:
        @invalidate_cache(key_prefix="user_data")
        async def update_user(user_id: int, data: dict):
            return await save_user_to_db(user_id, data)
    """
    def decorator(func: Callable) -> Callable:
        if asyncio.iscoroutinefunction(func):
            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                # Execute function first
                result = await func(*args, **kwargs)
                
                # Invalidate cache
                redis_client = await get_redis_client()
                
                try:
                    if exact_keys:
                        await redis_client.delete(exact_keys)
                        logger.debug(f"Invalidated exact keys: {exact_keys}")
                    
                    if pattern or key_prefix:
                        # Find matching keys
                        search_pattern = pattern or f"{key_prefix}*"
                        matching_keys = await redis_client.scan(match=search_pattern)
                        
                        if matching_keys:
                            await redis_client.delete(matching_keys)
                            logger.debug(f"Invalidated {len(matching_keys)} keys matching {search_pattern}")
                
                except Exception as e:
                    logger.error(f"Cache invalidation error: {e}")
                
                return result
            
            return async_wrapper
        else:
            @functools.wraps(func)
            def sync_wrapper(*args, **kwargs):
                # Execute function first
                result = func(*args, **kwargs)
                
                # Run invalidation in new event loop
                try:
                    asyncio.run(_invalidate_cache(pattern, key_prefix, exact_keys))
                except RuntimeError:
                    logger.warning("Could not invalidate cache in sync context")
                
                return result
            
            return sync_wrapper
    
    async def _invalidate_cache(pattern, key_prefix, exact_keys):
        redis_client = await get_redis_client()
        
        try:
            if exact_keys:
                await redis_client.delete(exact_keys)
            
            if pattern or key_prefix:
                search_pattern = pattern or f"{key_prefix}*"
                matching_keys = await redis_client.scan(match=search_pattern)
                
                if matching_keys:
                    await redis_client.delete(matching_keys)
        except Exception as e:
            logger.error(f"Cache invalidation error: {e}")
    
    return decorator


def cache_key(*key_args, **key_kwargs):
    """
    Decorator to specify custom cache key generation.
    
    Args:
        *key_args: Argument indices to include
        **key_kwargs: Keyword argument names to include
    
    Example:
        @cache_key(0, user_type="type")
        @cache_async(ttl=3600)
        async def get_user_by_type(user_id: int, user_type: str, include_deleted: bool = False):
            # Only user_id and user_type will be in cache key
            pass
    """
    def decorator(func: Callable) -> Callable:
        func._cache_key_args = key_args
        func._cache_key_kwargs = key_kwargs
        return func
    
    return decorator


def ttl_cache(ttl_func: Callable[..., int]):
    """
    Decorator to set dynamic TTL based on function result.
    
    Args:
        ttl_func: Function that takes the result and returns TTL
    
    Example:
        def user_ttl(user):
            return 3600 if user.is_active else 300
        
        @ttl_cache(user_ttl)
        @cache_async()
        async def get_user(user_id: int):
            return await fetch_user(user_id)
    """
    def decorator(func: Callable) -> Callable:
        func._ttl_func = ttl_func
        return func
    
    return decorator


def conditional_cache(condition_func: Callable[..., bool]):
    """
    Decorator to conditionally cache based on arguments.
    
    Args:
        condition_func: Function that takes args/kwargs and returns whether to cache
    
    Example:
        def should_cache(user_id: int, force_refresh: bool = False):
            return not force_refresh and user_id > 0
        
        @conditional_cache(should_cache)
        @cache_async(ttl=3600)
        async def get_user(user_id: int, force_refresh: bool = False):
            return await fetch_user(user_id)
    """
    def decorator(func: Callable) -> Callable:
        func._condition_func = condition_func
        return func
    
    return decorator


class CacheManager:
    """
    Context manager for batch cache operations.
    
    Example:
        async with CacheManager() as cache:
            await cache.set("key1", "value1")
            await cache.set("key2", "value2")
            # All operations are pipelined for efficiency
    """
    
    def __init__(self, client: Optional[RedisClient] = None):
        self.client = client
        self._pipeline = None
        self._operations = []
    
    async def __aenter__(self):
        self.client = self.client or await get_redis_client()
        self._pipeline = self.client.client.pipeline()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if exc_type is None and self._operations:
            # Execute all operations
            try:
                await self._pipeline.execute()
                logger.debug(f"Executed {len(self._operations)} cache operations")
            except Exception as e:
                logger.error(f"Cache pipeline error: {e}")
        
        self._pipeline = None
        self._operations = []
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None):
        """Add SET operation to pipeline."""
        self._operations.append(("set", key, value, ttl))
        self._pipeline.set(key, value, ex=ttl)
    
    def delete(self, key: Union[str, List[str]]):
        """Add DELETE operation to pipeline."""
        if isinstance(key, str):
            key = [key]
        self._operations.append(("delete", key))
        self._pipeline.delete(*key)
    
    def expire(self, key: str, ttl: int):
        """Add EXPIRE operation to pipeline."""
        self._operations.append(("expire", key, ttl))
        self._pipeline.expire(key, ttl)