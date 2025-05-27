"""
Cache invalidation strategies and utilities.
"""

import asyncio
import logging
from typing import List, Set, Dict, Any, Optional, Callable, Pattern
import re
from datetime import datetime, timedelta
from enum import Enum

from .client import RedisClient, get_redis_client
from .pubsub import RedisPubSub

logger = logging.getLogger(__name__)


class InvalidationStrategy(Enum):
    """Cache invalidation strategies."""
    IMMEDIATE = "immediate"
    LAZY = "lazy" 
    TTL_BASED = "ttl_based"
    TAG_BASED = "tag_based"
    DEPENDENCY_BASED = "dependency_based"
    PATTERN_BASED = "pattern_based"


class CacheInvalidator:
    """
    Manages cache invalidation across the system.
    
    Supports multiple invalidation strategies and distributed coordination.
    """
    
    def __init__(
        self,
        client: Optional[RedisClient] = None,
        pubsub: Optional[RedisPubSub] = None,
        default_strategy: InvalidationStrategy = InvalidationStrategy.IMMEDIATE
    ):
        self.client = client
        self.pubsub = pubsub
        self.default_strategy = default_strategy
        
        # Track invalidation rules
        self._tag_mappings: Dict[str, Set[str]] = {}
        self._dependency_graph: Dict[str, Set[str]] = {}
        self._pattern_rules: List[tuple[Pattern, str]] = []
        
        # Statistics
        self._stats = {
            "invalidations": 0,
            "cache_hits_prevented": 0,
            "distributed_invalidations": 0
        }
    
    async def initialize(self):
        """Initialize the invalidator."""
        if self.client is None:
            self.client = await get_redis_client()
        
        if self.pubsub is None:
            from .pubsub import RedisPubSub
            self.pubsub = RedisPubSub(self.client)
            await self.pubsub.subscribe("cache_invalidation", self._handle_distributed_invalidation)
        
        logger.info("Cache invalidator initialized")
    
    async def invalidate(
        self,
        keys: Optional[List[str]] = None,
        tags: Optional[List[str]] = None,
        patterns: Optional[List[str]] = None,
        strategy: Optional[InvalidationStrategy] = None,
        distribute: bool = True
    ) -> int:
        """
        Invalidate cache entries.
        
        Args:
            keys: Exact keys to invalidate
            tags: Tags to invalidate (all keys with these tags)
            patterns: Patterns to match for invalidation
            strategy: Invalidation strategy to use
            distribute: Whether to distribute invalidation to other instances
        
        Returns:
            Number of keys invalidated
        """
        strategy = strategy or self.default_strategy
        all_keys = set()
        
        # Collect keys from different sources
        if keys:
            all_keys.update(keys)
        
        if tags:
            for tag in tags:
                tagged_keys = await self._get_keys_by_tag(tag)
                all_keys.update(tagged_keys)
        
        if patterns:
            for pattern in patterns:
                pattern_keys = await self._get_keys_by_pattern(pattern)
                all_keys.update(pattern_keys)
        
        if not all_keys:
            return 0
        
        # Apply invalidation strategy
        invalidated_count = await self._apply_strategy(strategy, list(all_keys))
        
        # Distribute to other instances
        if distribute and self.pubsub:
            await self._distribute_invalidation(list(all_keys), strategy)
        
        # Update dependencies
        await self._invalidate_dependencies(list(all_keys))
        
        self._stats["invalidations"] += invalidated_count
        logger.info(f"Invalidated {invalidated_count} cache entries")
        
        return invalidated_count
    
    async def invalidate_by_prefix(self, prefix: str, distribute: bool = True) -> int:
        """Invalidate all keys with a given prefix."""
        pattern = f"{prefix}*"
        return await self.invalidate(patterns=[pattern], distribute=distribute)
    
    async def invalidate_by_suffix(self, suffix: str, distribute: bool = True) -> int:
        """Invalidate all keys with a given suffix."""
        pattern = f"*{suffix}"
        return await self.invalidate(patterns=[pattern], distribute=distribute)
    
    async def tag_key(self, key: str, tags: List[str]):
        """Associate tags with a cache key."""
        for tag in tags:
            if tag not in self._tag_mappings:
                self._tag_mappings[tag] = set()
            self._tag_mappings[tag].add(key)
        
        # Store tag mapping in Redis for distributed access
        tag_key = f"cache_tags:{key}"
        await self.client.set(tag_key, tags, ttl=3600)
    
    async def add_dependency(self, key: str, depends_on: List[str]):
        """Add cache key dependencies."""
        if key not in self._dependency_graph:
            self._dependency_graph[key] = set()
        
        self._dependency_graph[key].update(depends_on)
        
        # Store dependency in Redis
        dep_key = f"cache_deps:{key}"
        await self.client.set(dep_key, depends_on, ttl=3600)
    
    async def add_pattern_rule(self, pattern: str, invalidate_pattern: str):
        """Add a pattern-based invalidation rule."""
        compiled_pattern = re.compile(pattern)
        self._pattern_rules.append((compiled_pattern, invalidate_pattern))
        
        # Store rule in Redis
        rule_key = f"cache_rules:{pattern}"
        await self.client.set(rule_key, invalidate_pattern, ttl=3600)
    
    async def _apply_strategy(self, strategy: InvalidationStrategy, keys: List[str]) -> int:
        """Apply the specified invalidation strategy."""
        if strategy == InvalidationStrategy.IMMEDIATE:
            return await self._immediate_invalidation(keys)
        
        elif strategy == InvalidationStrategy.LAZY:
            return await self._lazy_invalidation(keys)
        
        elif strategy == InvalidationStrategy.TTL_BASED:
            return await self._ttl_based_invalidation(keys)
        
        else:
            # Default to immediate
            return await self._immediate_invalidation(keys)
    
    async def _immediate_invalidation(self, keys: List[str]) -> int:
        """Immediately delete cache entries."""
        if not keys:
            return 0
        
        deleted = await self.client.delete(keys)
        logger.debug(f"Immediately invalidated {deleted} keys")
        return deleted
    
    async def _lazy_invalidation(self, keys: List[str]) -> int:
        """Mark entries as invalid (lazy deletion)."""
        if not keys:
            return 0
        
        # Set invalid marker with short TTL
        invalid_mappings = {f"invalid:{key}": True for key in keys}
        await self.client.mset(invalid_mappings, ttl=300)
        
        logger.debug(f"Marked {len(keys)} keys as invalid")
        return len(keys)
    
    async def _ttl_based_invalidation(self, keys: List[str]) -> int:
        """Set very short TTL on entries."""
        if not keys:
            return 0
        
        count = 0
        for key in keys:
            # Set TTL to 1 second
            success = await self.client.expire(key, 1)
            if success:
                count += 1
        
        logger.debug(f"Set short TTL on {count} keys")
        return count
    
    async def _get_keys_by_tag(self, tag: str) -> List[str]:
        """Get all keys associated with a tag."""
        # Check local mapping first
        if tag in self._tag_mappings:
            return list(self._tag_mappings[tag])
        
        # Check Redis for distributed tags
        tag_pattern = f"cache_tags:*"
        tag_keys = await self.client.scan(match=tag_pattern)
        
        matching_keys = []
        for tag_key in tag_keys:
            tags = await self.client.get(tag_key)
            if tags and tag in tags:
                # Extract original key from tag key
                original_key = tag_key.replace("cache_tags:", "")
                matching_keys.append(original_key)
        
        return matching_keys
    
    async def _get_keys_by_pattern(self, pattern: str) -> List[str]:
        """Get all keys matching a pattern."""
        return await self.client.scan(match=pattern)
    
    async def _invalidate_dependencies(self, keys: List[str]):
        """Invalidate dependent cache entries."""
        dependent_keys = set()
        
        for key in keys:
            # Check local dependencies
            for dep_key, deps in self._dependency_graph.items():
                if key in deps:
                    dependent_keys.add(dep_key)
            
            # Check Redis dependencies
            dep_pattern = f"cache_deps:*"
            dep_keys = await self.client.scan(match=dep_pattern)
            
            for dep_key in dep_keys:
                dependencies = await self.client.get(dep_key)
                if dependencies and key in dependencies:
                    # Extract original key
                    original_key = dep_key.replace("cache_deps:", "")
                    dependent_keys.add(original_key)
        
        if dependent_keys:
            logger.debug(f"Invalidating {len(dependent_keys)} dependent keys")
            await self.invalidate(keys=list(dependent_keys), distribute=False)
    
    async def _distribute_invalidation(self, keys: List[str], strategy: InvalidationStrategy):
        """Distribute invalidation to other instances."""
        message = {
            "keys": keys,
            "strategy": strategy.value,
            "timestamp": datetime.now().isoformat(),
            "source": "local"
        }
        
        await self.pubsub.publish("cache_invalidation", message)
        self._stats["distributed_invalidations"] += 1
    
    async def _handle_distributed_invalidation(self, message: Dict[str, Any]):
        """Handle invalidation message from other instances."""
        if message.get("source") == "local":
            return  # Ignore own messages
        
        keys = message.get("keys", [])
        strategy_name = message.get("strategy", "immediate")
        
        try:
            strategy = InvalidationStrategy(strategy_name)
        except ValueError:
            strategy = InvalidationStrategy.IMMEDIATE
        
        if keys:
            await self._apply_strategy(strategy, keys)
            logger.debug(f"Applied distributed invalidation for {len(keys)} keys")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get invalidation statistics."""
        return self._stats.copy()
    
    def reset_stats(self):
        """Reset invalidation statistics."""
        self._stats = {
            "invalidations": 0,
            "cache_hits_prevented": 0,
            "distributed_invalidations": 0
        }


class SmartInvalidator(CacheInvalidator):
    """
    Smart cache invalidator with automatic pattern detection and optimization.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._access_patterns: Dict[str, int] = {}
        self._invalidation_frequency: Dict[str, List[datetime]] = {}
    
    async def track_access(self, key: str):
        """Track cache key access patterns."""
        self._access_patterns[key] = self._access_patterns.get(key, 0) + 1
    
    async def smart_invalidate(self, keys: List[str]) -> int:
        """Intelligently choose invalidation strategy based on patterns."""
        strategies = {}
        
        for key in keys:
            # Analyze access pattern
            access_count = self._access_patterns.get(key, 0)
            
            # Check invalidation frequency
            freq_key = key
            if freq_key not in self._invalidation_frequency:
                self._invalidation_frequency[freq_key] = []
            
            recent_invalidations = [
                ts for ts in self._invalidation_frequency[freq_key]
                if ts > datetime.now() - timedelta(hours=1)
            ]
            
            # Choose strategy based on patterns
            if len(recent_invalidations) > 5:
                # Frequently invalidated - use lazy
                strategies[key] = InvalidationStrategy.LAZY
            elif access_count > 100:
                # High access - use TTL to avoid cache stampede
                strategies[key] = InvalidationStrategy.TTL_BASED
            else:
                # Default immediate
                strategies[key] = InvalidationStrategy.IMMEDIATE
            
            # Record this invalidation
            self._invalidation_frequency[freq_key].append(datetime.now())
        
        # Group keys by strategy
        strategy_groups = {}
        for key, strategy in strategies.items():
            if strategy not in strategy_groups:
                strategy_groups[strategy] = []
            strategy_groups[strategy].append(key)
        
        # Apply each strategy
        total_invalidated = 0
        for strategy, strategy_keys in strategy_groups.items():
            count = await self._apply_strategy(strategy, strategy_keys)
            total_invalidated += count
            
            logger.info(f"Applied {strategy.value} strategy to {len(strategy_keys)} keys")
        
        return total_invalidated


class ConditionalInvalidator:
    """
    Invalidator with conditional rules based on business logic.
    """
    
    def __init__(self, base_invalidator: CacheInvalidator):
        self.base_invalidator = base_invalidator
        self._conditions: List[Callable[[str, Any], bool]] = []
    
    def add_condition(self, condition: Callable[[str, Any], bool]):
        """Add a condition function for invalidation."""
        self._conditions.append(condition)
    
    async def conditional_invalidate(self, key: str, context: Any = None) -> bool:
        """Invalidate key only if conditions are met."""
        for condition in self._conditions:
            if not condition(key, context):
                logger.debug(f"Condition failed for {key}, skipping invalidation")
                return False
        
        count = await self.base_invalidator.invalidate(keys=[key])
        return count > 0


# Predefined condition functions
def time_based_condition(start_hour: int = 0, end_hour: int = 23):
    """Create time-based invalidation condition."""
    def condition(key: str, context: Any) -> bool:
        current_hour = datetime.now().hour
        return start_hour <= current_hour <= end_hour
    return condition


def user_based_condition(allowed_users: Set[str]):
    """Create user-based invalidation condition."""
    def condition(key: str, context: Any) -> bool:
        user_id = getattr(context, 'user_id', None) if context else None
        return user_id in allowed_users
    return condition


def frequency_based_condition(max_per_hour: int = 10):
    """Create frequency-based invalidation condition."""
    frequency_tracker = {}
    
    def condition(key: str, context: Any) -> bool:
        now = datetime.now()
        hour_key = f"{key}:{now.hour}"
        
        if hour_key not in frequency_tracker:
            frequency_tracker[hour_key] = 0
        
        frequency_tracker[hour_key] += 1
        
        # Clean old entries
        for fkey in list(frequency_tracker.keys()):
            if not fkey.endswith(f":{now.hour}"):
                del frequency_tracker[fkey]
        
        return frequency_tracker[hour_key] <= max_per_hour
    
    return condition


# Global invalidator instance
_invalidator: Optional[CacheInvalidator] = None


async def get_cache_invalidator() -> CacheInvalidator:
    """Get the global cache invalidator."""
    global _invalidator
    
    if _invalidator is None:
        _invalidator = CacheInvalidator()
        await _invalidator.initialize()
    
    return _invalidator


async def invalidate_cache(
    keys: Optional[List[str]] = None,
    tags: Optional[List[str]] = None,
    patterns: Optional[List[str]] = None,
    **kwargs
) -> int:
    """Convenience function for cache invalidation."""
    invalidator = await get_cache_invalidator()
    return await invalidator.invalidate(keys=keys, tags=tags, patterns=patterns, **kwargs)