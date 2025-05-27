"""
Common caching patterns implementation.
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from typing import Optional, Any, Callable, Dict, List
from datetime import datetime, timedelta
import random

from .client import RedisClient, get_redis_client
from .decorators import cache_async

logger = logging.getLogger(__name__)


class CachePattern(ABC):
    """Base class for caching patterns."""
    
    def __init__(self, client: Optional[RedisClient] = None):
        self.client = client
    
    async def _ensure_client(self):
        """Ensure Redis client is available."""
        if self.client is None:
            self.client = await get_redis_client()
    
    @abstractmethod
    async def get(self, key: str, fetch_func: Callable, ttl: Optional[int] = None) -> Any:
        """Get value using the caching pattern."""
        pass


class CacheAside(CachePattern):
    """
    Cache-aside (lazy loading) pattern.
    
    1. Check cache for data
    2. If miss, fetch from source
    3. Update cache
    4. Return data
    """
    
    async def get(self, key: str, fetch_func: Callable, ttl: Optional[int] = None) -> Any:
        """Get value using cache-aside pattern."""
        await self._ensure_client()
        
        # Try cache first
        value = await self.client.get(key)
        if value is not None:
            logger.debug(f"Cache hit for {key}")
            return value
        
        # Cache miss - fetch from source
        logger.debug(f"Cache miss for {key}, fetching from source")
        
        if asyncio.iscoroutinefunction(fetch_func):
            value = await fetch_func()
        else:
            value = fetch_func()
        
        # Update cache
        if value is not None:
            await self.client.set(key, value, ttl=ttl)
            logger.debug(f"Cached value for {key}")
        
        return value


class WriteThrough(CachePattern):
    """
    Write-through caching pattern.
    
    1. Write to cache
    2. Write to source
    3. Return result
    """
    
    async def set(
        self,
        key: str,
        value: Any,
        write_func: Callable,
        ttl: Optional[int] = None
    ) -> Any:
        """Set value using write-through pattern."""
        await self._ensure_client()
        
        # Write to cache first
        await self.client.set(key, value, ttl=ttl)
        logger.debug(f"Updated cache for {key}")
        
        # Write to source
        try:
            if asyncio.iscoroutinefunction(write_func):
                result = await write_func(value)
            else:
                result = write_func(value)
            
            return result
        
        except Exception as e:
            # Rollback cache on error
            logger.error(f"Write-through failed for {key}, rolling back cache: {e}")
            await self.client.delete(key)
            raise
    
    async def get(self, key: str, fetch_func: Callable, ttl: Optional[int] = None) -> Any:
        """Get value (delegates to cache-aside)."""
        cache_aside = CacheAside(self.client)
        return await cache_aside.get(key, fetch_func, ttl)


class WriteBehind(CachePattern):
    """
    Write-behind (write-back) caching pattern.
    
    1. Write to cache immediately
    2. Asynchronously write to source
    3. Return immediately
    """
    
    def __init__(self, client: Optional[RedisClient] = None, batch_size: int = 100, flush_interval: int = 5):
        super().__init__(client)
        self.batch_size = batch_size
        self.flush_interval = flush_interval
        self._write_queue: Dict[str, Any] = {}
        self._flush_task: Optional[asyncio.Task] = None
    
    async def start(self):
        """Start the background flush task."""
        await self._ensure_client()
        
        if self._flush_task is None:
            self._flush_task = asyncio.create_task(self._flush_loop())
            logger.info("Started write-behind flush task")
    
    async def stop(self):
        """Stop the background flush task."""
        if self._flush_task:
            self._flush_task.cancel()
            try:
                await self._flush_task
            except asyncio.CancelledError:
                pass
            self._flush_task = None
            logger.info("Stopped write-behind flush task")
        
        # Flush remaining items
        if self._write_queue:
            await self._flush_queue()
    
    async def set(
        self,
        key: str,
        value: Any,
        write_func: Callable,
        ttl: Optional[int] = None
    ) -> None:
        """Set value using write-behind pattern."""
        await self._ensure_client()
        
        # Write to cache immediately
        await self.client.set(key, value, ttl=ttl)
        logger.debug(f"Updated cache for {key}")
        
        # Queue for async write
        self._write_queue[key] = (value, write_func)
        
        # Flush if queue is full
        if len(self._write_queue) >= self.batch_size:
            await self._flush_queue()
    
    async def get(self, key: str, fetch_func: Callable, ttl: Optional[int] = None) -> Any:
        """Get value (checks write queue first)."""
        # Check write queue first
        if key in self._write_queue:
            value, _ = self._write_queue[key]
            return value
        
        # Otherwise use cache-aside
        cache_aside = CacheAside(self.client)
        return await cache_aside.get(key, fetch_func, ttl)
    
    async def _flush_loop(self):
        """Background task to flush write queue."""
        while True:
            try:
                await asyncio.sleep(self.flush_interval)
                if self._write_queue:
                    await self._flush_queue()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in flush loop: {e}")
    
    async def _flush_queue(self):
        """Flush the write queue to source."""
        if not self._write_queue:
            return
        
        queue_copy = self._write_queue.copy()
        self._write_queue.clear()
        
        logger.info(f"Flushing {len(queue_copy)} items to source")
        
        for key, (value, write_func) in queue_copy.items():
            try:
                if asyncio.iscoroutinefunction(write_func):
                    await write_func(value)
                else:
                    write_func(value)
            except Exception as e:
                logger.error(f"Failed to write {key} to source: {e}")
                # Could implement retry logic here


class RefreshAhead(CachePattern):
    """
    Refresh-ahead caching pattern.
    
    Proactively refreshes cache entries before they expire.
    """
    
    def __init__(
        self,
        client: Optional[RedisClient] = None,
        refresh_threshold: float = 0.8  # Refresh when 80% of TTL has passed
    ):
        super().__init__(client)
        self.refresh_threshold = refresh_threshold
        self._refresh_tasks: Dict[str, asyncio.Task] = {}
    
    async def get(
        self,
        key: str,
        fetch_func: Callable,
        ttl: Optional[int] = None
    ) -> Any:
        """Get value with refresh-ahead."""
        await self._ensure_client()
        
        # Get value and remaining TTL
        value = await self.client.get(key)
        remaining_ttl = await self.client.ttl(key)
        
        if value is None:
            # Cache miss - fetch and cache
            logger.debug(f"Cache miss for {key}, fetching from source")
            
            if asyncio.iscoroutinefunction(fetch_func):
                value = await fetch_func()
            else:
                value = fetch_func()
            
            if value is not None and ttl:
                await self.client.set(key, value, ttl=ttl)
                
                # Schedule refresh
                self._schedule_refresh(key, fetch_func, ttl)
        
        elif ttl and remaining_ttl > 0:
            # Check if we should refresh
            refresh_at = ttl * (1 - self.refresh_threshold)
            time_passed = ttl - remaining_ttl
            
            if time_passed >= refresh_at and key not in self._refresh_tasks:
                # Schedule refresh
                logger.debug(f"Scheduling refresh for {key}")
                self._schedule_refresh(key, fetch_func, ttl)
        
        return value
    
    def _schedule_refresh(self, key: str, fetch_func: Callable, ttl: int):
        """Schedule a cache refresh."""
        if key in self._refresh_tasks:
            return
        
        async def refresh():
            try:
                # Wait until refresh time
                refresh_at = ttl * self.refresh_threshold
                await asyncio.sleep(refresh_at)
                
                logger.debug(f"Refreshing cache for {key}")
                
                # Fetch new value
                if asyncio.iscoroutinefunction(fetch_func):
                    value = await fetch_func()
                else:
                    value = fetch_func()
                
                # Update cache
                if value is not None:
                    await self.client.set(key, value, ttl=ttl)
                    logger.debug(f"Refreshed cache for {key}")
            
            except asyncio.CancelledError:
                pass
            except Exception as e:
                logger.error(f"Error refreshing cache for {key}: {e}")
            
            finally:
                self._refresh_tasks.pop(key, None)
        
        task = asyncio.create_task(refresh())
        self._refresh_tasks[key] = task
    
    async def stop(self):
        """Cancel all refresh tasks."""
        for task in self._refresh_tasks.values():
            task.cancel()
        
        await asyncio.gather(*self._refresh_tasks.values(), return_exceptions=True)
        self._refresh_tasks.clear()


class MultiLevelCache(CachePattern):
    """
    Multi-level caching pattern with L1 (memory) and L2 (Redis) cache.
    """
    
    def __init__(
        self,
        client: Optional[RedisClient] = None,
        l1_max_size: int = 1000,
        l1_ttl: int = 60  # L1 TTL in seconds
    ):
        super().__init__(client)
        self.l1_max_size = l1_max_size
        self.l1_ttl = l1_ttl
        self._l1_cache: Dict[str, tuple[Any, float]] = {}
    
    async def get(self, key: str, fetch_func: Callable, ttl: Optional[int] = None) -> Any:
        """Get value from multi-level cache."""
        await self._ensure_client()
        
        # Check L1 cache
        if key in self._l1_cache:
            value, expiry = self._l1_cache[key]
            if datetime.now().timestamp() < expiry:
                logger.debug(f"L1 cache hit for {key}")
                return value
            else:
                # Expired
                del self._l1_cache[key]
        
        # Check L2 cache (Redis)
        value = await self.client.get(key)
        if value is not None:
            logger.debug(f"L2 cache hit for {key}")
            # Populate L1
            self._set_l1(key, value)
            return value
        
        # Cache miss - fetch from source
        logger.debug(f"Cache miss for {key}, fetching from source")
        
        if asyncio.iscoroutinefunction(fetch_func):
            value = await fetch_func()
        else:
            value = fetch_func()
        
        if value is not None:
            # Update both caches
            await self.client.set(key, value, ttl=ttl)
            self._set_l1(key, value)
        
        return value
    
    def _set_l1(self, key: str, value: Any):
        """Set value in L1 cache with LRU eviction."""
        # Evict if at capacity
        if len(self._l1_cache) >= self.l1_max_size:
            # Remove oldest entry
            oldest_key = min(self._l1_cache.keys(), key=lambda k: self._l1_cache[k][1])
            del self._l1_cache[oldest_key]
        
        expiry = datetime.now().timestamp() + self.l1_ttl
        self._l1_cache[key] = (value, expiry)
    
    async def invalidate(self, key: str):
        """Invalidate entry in both caches."""
        self._l1_cache.pop(key, None)
        await self.client.delete(key)
    
    def clear_l1(self):
        """Clear L1 cache."""
        self._l1_cache.clear()


class CircuitBreakerCache(CachePattern):
    """
    Cache with circuit breaker pattern for fault tolerance.
    """
    
    def __init__(
        self,
        client: Optional[RedisClient] = None,
        failure_threshold: int = 5,
        recovery_timeout: int = 60,
        half_open_requests: int = 3
    ):
        super().__init__(client)
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.half_open_requests = half_open_requests
        
        self._failure_count = 0
        self._last_failure_time = None
        self._state = "closed"  # closed, open, half_open
        self._half_open_count = 0
    
    async def get(self, key: str, fetch_func: Callable, ttl: Optional[int] = None) -> Any:
        """Get value with circuit breaker protection."""
        await self._ensure_client()
        
        # Check circuit state
        if self._state == "open":
            if self._should_attempt_reset():
                self._state = "half_open"
                self._half_open_count = 0
                logger.info("Circuit breaker entering half-open state")
            else:
                # Fast fail
                logger.warning(f"Circuit breaker open, failing fast for {key}")
                # Try to return from source directly
                if asyncio.iscoroutinefunction(fetch_func):
                    return await fetch_func()
                else:
                    return fetch_func()
        
        try:
            # Try cache
            value = await self.client.get(key)
            if value is not None:
                self._on_success()
                return value
            
            # Cache miss - fetch from source
            if asyncio.iscoroutinefunction(fetch_func):
                value = await fetch_func()
            else:
                value = fetch_func()
            
            # Update cache
            if value is not None:
                await self.client.set(key, value, ttl=ttl)
            
            self._on_success()
            return value
        
        except Exception as e:
            self._on_failure()
            logger.error(f"Circuit breaker cache error for {key}: {e}")
            
            # Try source directly as fallback
            try:
                if asyncio.iscoroutinefunction(fetch_func):
                    return await fetch_func()
                else:
                    return fetch_func()
            except:
                raise
    
    def _should_attempt_reset(self) -> bool:
        """Check if circuit should attempt reset."""
        return (
            self._last_failure_time and
            (datetime.now().timestamp() - self._last_failure_time) >= self.recovery_timeout
        )
    
    def _on_success(self):
        """Handle successful operation."""
        if self._state == "half_open":
            self._half_open_count += 1
            if self._half_open_count >= self.half_open_requests:
                self._state = "closed"
                self._failure_count = 0
                logger.info("Circuit breaker closed")
        else:
            self._failure_count = 0
    
    def _on_failure(self):
        """Handle failed operation."""
        self._failure_count += 1
        self._last_failure_time = datetime.now().timestamp()
        
        if self._state == "half_open":
            self._state = "open"
            logger.warning("Circuit breaker reopened")
        elif self._failure_count >= self.failure_threshold:
            self._state = "open"
            logger.warning(f"Circuit breaker opened after {self._failure_count} failures")