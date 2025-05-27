"""
Redis monitoring and metrics collection.
"""

import asyncio
import logging
import time
from typing import Dict, Any, Optional, List, Callable
from datetime import datetime, timedelta
from functools import wraps

from prometheus_client import Counter, Histogram, Gauge, Summary, Info

from .client import RedisClient, get_redis_client
from ..monitoring.metrics.registry import MetricsRegistry

logger = logging.getLogger(__name__)

# Initialize metrics registry
metrics = MetricsRegistry(namespace="brahminykite_cache")

# Cache operation metrics
cache_operations_total = metrics.counter(
    "operations_total",
    "Total number of cache operations",
    ["operation", "status"]
)

cache_operation_duration = metrics.histogram(
    "operation_duration_seconds",
    "Cache operation duration",
    ["operation"],
    buckets=[0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0]
)

cache_hit_rate = metrics.gauge(
    "hit_rate",
    "Cache hit rate"
)

cache_miss_rate = metrics.gauge(
    "miss_rate", 
    "Cache miss rate"
)

cache_key_count = metrics.gauge(
    "keys_total",
    "Total number of cache keys"
)

cache_memory_usage = metrics.gauge(
    "memory_usage_bytes",
    "Cache memory usage in bytes"
)

# Redis-specific metrics
redis_connected_clients = metrics.gauge(
    "redis_connected_clients",
    "Number of connected Redis clients"
)

redis_commands_processed = metrics.counter(
    "redis_commands_processed_total",
    "Total number of Redis commands processed"
)

redis_keyspace_hits = metrics.counter(
    "redis_keyspace_hits_total",
    "Total number of Redis keyspace hits"
)

redis_keyspace_misses = metrics.counter(
    "redis_keyspace_misses_total", 
    "Total number of Redis keyspace misses"
)

redis_memory_used = metrics.gauge(
    "redis_memory_used_bytes",
    "Redis memory usage in bytes"
)

redis_cpu_usage = metrics.gauge(
    "redis_cpu_usage_percent",
    "Redis CPU usage percentage"
)

# Session metrics
session_count_active = metrics.gauge(
    "sessions_active_total",
    "Total number of active sessions"
)

session_operations_total = metrics.counter(
    "session_operations_total",
    "Total number of session operations",
    ["operation", "status"]
)

session_duration = metrics.histogram(
    "session_duration_seconds",
    "Session duration",
    buckets=[60, 300, 900, 1800, 3600, 7200, 14400, 28800, 86400]  # 1min to 1day
)

# Pub/sub metrics
pubsub_messages_sent = metrics.counter(
    "pubsub_messages_sent_total",
    "Total number of pub/sub messages sent",
    ["channel"]
)

pubsub_messages_received = metrics.counter(
    "pubsub_messages_received_total",
    "Total number of pub/sub messages received", 
    ["channel"]
)

pubsub_subscribers = metrics.gauge(
    "pubsub_subscribers_total",
    "Total number of pub/sub subscribers",
    ["channel"]
)


class CacheMetricsCollector:
    """
    Collects and reports cache metrics.
    """
    
    def __init__(self, client: Optional[RedisClient] = None):
        self.client = client
        self._collection_interval = 30  # seconds
        self._collector_task: Optional[asyncio.Task] = None
        
        # Local metrics tracking
        self._hit_count = 0
        self._miss_count = 0
        self._operation_counts: Dict[str, int] = {}
    
    async def start(self):
        """Start metrics collection."""
        if self.client is None:
            self.client = await get_redis_client()
        
        if self._collector_task is None:
            self._collector_task = asyncio.create_task(self._collection_loop())
            logger.info("Cache metrics collection started")
    
    async def stop(self):
        """Stop metrics collection."""
        if self._collector_task:
            self._collector_task.cancel()
            try:
                await self._collector_task
            except asyncio.CancelledError:
                pass
            self._collector_task = None
            logger.info("Cache metrics collection stopped")
    
    async def _collection_loop(self):
        """Main metrics collection loop."""
        while True:
            try:
                await self._collect_redis_metrics()
                await self._collect_cache_metrics()
                await asyncio.sleep(self._collection_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in metrics collection: {e}")
                await asyncio.sleep(self._collection_interval)
    
    async def _collect_redis_metrics(self):
        """Collect Redis server metrics."""
        try:
            # Get Redis info
            info = await self.client.client.info()
            
            # Server metrics
            redis_connected_clients.set(info.get("connected_clients", 0))
            redis_commands_processed.inc(info.get("total_commands_processed", 0))
            
            # Memory metrics
            redis_memory_used.set(info.get("used_memory", 0))
            cache_memory_usage.set(info.get("used_memory", 0))
            
            # CPU metrics
            cpu_sys = info.get("used_cpu_sys", 0)
            cpu_user = info.get("used_cpu_user", 0)
            cpu_total = cpu_sys + cpu_user
            if cpu_total > 0:
                redis_cpu_usage.set(cpu_total)
            
            # Keyspace metrics
            stats = await self.client.client.info("stats")
            keyspace_hits = stats.get("keyspace_hits", 0)
            keyspace_misses = stats.get("keyspace_misses", 0)
            
            redis_keyspace_hits.inc(keyspace_hits)
            redis_keyspace_misses.inc(keyspace_misses)
            
            # Calculate hit rate
            total_requests = keyspace_hits + keyspace_misses
            if total_requests > 0:
                hit_rate = keyspace_hits / total_requests
                cache_hit_rate.set(hit_rate)
                cache_miss_rate.set(1 - hit_rate)
        
        except Exception as e:
            logger.error(f"Error collecting Redis metrics: {e}")
    
    async def _collect_cache_metrics(self):
        """Collect application cache metrics."""
        try:
            # Count total keys
            keys = await self.client.scan()
            cache_key_count.set(len(keys))
            
            # Get client metrics
            client_metrics = self.client.get_metrics()
            
            # Update hit/miss rates
            hits = client_metrics.get("hits", 0)
            misses = client_metrics.get("misses", 0)
            total = hits + misses
            
            if total > 0:
                cache_hit_rate.set(hits / total)
                cache_miss_rate.set(misses / total)
        
        except Exception as e:
            logger.error(f"Error collecting cache metrics: {e}")
    
    def record_operation(self, operation: str, duration: float, status: str = "success"):
        """Record a cache operation."""
        cache_operations_total.labels(operation=operation, status=status).inc()
        cache_operation_duration.labels(operation=operation).observe(duration)
        
        # Update local counts
        self._operation_counts[operation] = self._operation_counts.get(operation, 0) + 1
    
    def record_hit(self):
        """Record a cache hit."""
        self._hit_count += 1
    
    def record_miss(self):
        """Record a cache miss."""
        self._miss_count += 1
    
    def get_local_stats(self) -> Dict[str, Any]:
        """Get local statistics."""
        total_ops = self._hit_count + self._miss_count
        hit_rate = self._hit_count / total_ops if total_ops > 0 else 0
        
        return {
            "hits": self._hit_count,
            "misses": self._miss_count,
            "hit_rate": hit_rate,
            "operations": self._operation_counts.copy()
        }


def monitor_cache_operation(operation: str):
    """
    Decorator to monitor cache operations.
    
    Args:
        operation: Operation name (get, set, delete, etc.)
    
    Example:
        @monitor_cache_operation("get")
        async def get_from_cache(key: str):
            return await redis_client.get(key)
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            start_time = time.time()
            status = "success"
            
            try:
                result = await func(*args, **kwargs)
                
                # Determine if this was a hit or miss for get operations
                if operation == "get":
                    if result is not None:
                        collector.record_hit()
                    else:
                        collector.record_miss()
                
                return result
            
            except Exception as e:
                status = "error"
                raise
            
            finally:
                duration = time.time() - start_time
                collector.record_operation(operation, duration, status)
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            start_time = time.time()
            status = "success"
            
            try:
                result = func(*args, **kwargs)
                
                if operation == "get":
                    if result is not None:
                        collector.record_hit()
                    else:
                        collector.record_miss()
                
                return result
            
            except Exception as e:
                status = "error"
                raise
            
            finally:
                duration = time.time() - start_time
                collector.record_operation(operation, duration, status)
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    
    return decorator


class SessionMetricsCollector:
    """Collects session-related metrics."""
    
    def __init__(self):
        self._session_start_times: Dict[str, datetime] = {}
    
    def record_session_start(self, session_id: str):
        """Record session start."""
        self._session_start_times[session_id] = datetime.now()
        session_operations_total.labels(operation="create", status="success").inc()
    
    def record_session_end(self, session_id: str):
        """Record session end."""
        if session_id in self._session_start_times:
            start_time = self._session_start_times.pop(session_id)
            duration = (datetime.now() - start_time).total_seconds()
            session_duration.observe(duration)
        
        session_operations_total.labels(operation="destroy", status="success").inc()
    
    def update_active_session_count(self, count: int):
        """Update active session count."""
        session_count_active.set(count)


class PubSubMetricsCollector:
    """Collects pub/sub metrics."""
    
    def record_message_sent(self, channel: str):
        """Record a message sent."""
        pubsub_messages_sent.labels(channel=channel).inc()
    
    def record_message_received(self, channel: str):
        """Record a message received."""
        pubsub_messages_received.labels(channel=channel).inc()
    
    def update_subscriber_count(self, channel: str, count: int):
        """Update subscriber count for a channel."""
        pubsub_subscribers.labels(channel=channel).set(count)


class CacheHealthCheck:
    """Health check for cache system."""
    
    def __init__(self, client: Optional[RedisClient] = None):
        self.client = client
    
    async def check_health(self) -> Dict[str, Any]:
        """Perform comprehensive health check."""
        if self.client is None:
            self.client = await get_redis_client()
        
        health_data = {
            "healthy": True,
            "timestamp": datetime.now().isoformat(),
            "checks": {}
        }
        
        # Redis connectivity
        try:
            start_time = time.time()
            await self.client.client.ping()
            ping_time = time.time() - start_time
            
            health_data["checks"]["redis_ping"] = {
                "status": "pass",
                "response_time_ms": ping_time * 1000
            }
        except Exception as e:
            health_data["healthy"] = False
            health_data["checks"]["redis_ping"] = {
                "status": "fail",
                "error": str(e)
            }
        
        # Memory usage
        try:
            info = await self.client.client.info("memory")
            used_memory = info.get("used_memory", 0)
            max_memory = info.get("maxmemory", 0)
            
            if max_memory > 0:
                memory_usage_percent = (used_memory / max_memory) * 100
                
                health_data["checks"]["memory_usage"] = {
                    "status": "pass" if memory_usage_percent < 90 else "warn",
                    "used_bytes": used_memory,
                    "max_bytes": max_memory,
                    "usage_percent": memory_usage_percent
                }
                
                if memory_usage_percent >= 95:
                    health_data["healthy"] = False
            else:
                health_data["checks"]["memory_usage"] = {
                    "status": "pass",
                    "used_bytes": used_memory,
                    "note": "No memory limit set"
                }
        except Exception as e:
            health_data["checks"]["memory_usage"] = {
                "status": "fail",
                "error": str(e)
            }
        
        # Key count
        try:
            keys = await self.client.scan()
            key_count = len(keys)
            
            health_data["checks"]["key_count"] = {
                "status": "pass",
                "count": key_count
            }
        except Exception as e:
            health_data["checks"]["key_count"] = {
                "status": "fail",
                "error": str(e)
            }
        
        # Performance test
        try:
            test_key = "health_check_test"
            test_value = "test_data"
            
            # Set test
            start_time = time.time()
            await self.client.set(test_key, test_value, ttl=60)
            set_time = time.time() - start_time
            
            # Get test
            start_time = time.time()
            retrieved_value = await self.client.get(test_key)
            get_time = time.time() - start_time
            
            # Cleanup
            await self.client.delete(test_key)
            
            if retrieved_value == test_value:
                health_data["checks"]["performance"] = {
                    "status": "pass",
                    "set_time_ms": set_time * 1000,
                    "get_time_ms": get_time * 1000
                }
            else:
                health_data["healthy"] = False
                health_data["checks"]["performance"] = {
                    "status": "fail",
                    "error": "Data integrity check failed"
                }
        except Exception as e:
            health_data["checks"]["performance"] = {
                "status": "fail",
                "error": str(e)
            }
        
        return health_data


# Global collectors
collector = CacheMetricsCollector()
session_collector = SessionMetricsCollector()
pubsub_collector = PubSubMetricsCollector()
health_checker = CacheHealthCheck()


async def start_monitoring():
    """Start all monitoring components."""
    await collector.start()
    logger.info("Cache monitoring started")


async def stop_monitoring():
    """Stop all monitoring components."""
    await collector.stop()
    logger.info("Cache monitoring stopped")


def get_cache_metrics() -> Dict[str, Any]:
    """Get comprehensive cache metrics."""
    return {
        "local_stats": collector.get_local_stats(),
        "timestamp": datetime.now().isoformat()
    }