# Redis Caching Implementation

## Overview

The Redis caching layer for BrahminyKite provides high-performance distributed caching, session management, and pub/sub messaging capabilities. It's designed for scalability, reliability, and ease of use.

## Features

### Core Caching
- **High-performance Redis client** with connection pooling
- **Multiple serialization formats** (JSON, Pickle, MessagePack)
- **Automatic compression** for large values
- **TTL-based expiration** with automatic cleanup
- **Circuit breaker pattern** for fault tolerance

### Caching Patterns
- **Cache-aside** (lazy loading)
- **Write-through** caching
- **Write-behind** (write-back) caching
- **Refresh-ahead** caching
- **Multi-level caching** (L1 memory + L2 Redis)

### Cache Invalidation
- **Multiple strategies**: immediate, lazy, TTL-based, tag-based
- **Pattern-based invalidation** with wildcards
- **Dependency tracking** for related cache entries
- **Distributed invalidation** via pub/sub
- **Smart invalidation** with access pattern analysis

### Session Management
- **Distributed session storage** with Redis
- **Session encryption** support
- **Concurrent session limits** per user
- **IP validation** and security features
- **Session analytics** and monitoring

### Pub/Sub Messaging
- **Redis pub/sub** for real-time messaging
- **Message routing** and filtering
- **Event-driven architecture** support
- **Cache synchronization** across instances

### Monitoring & Observability
- **Prometheus metrics** integration
- **Health checks** and diagnostics
- **Performance tracking** and analytics
- **Rich logging** and error handling

## Quick Start

### Basic Caching

```python
from chil.caching import cache_async, create_redis_client

# Simple caching decorator
@cache_async(ttl=3600, key_prefix="user_data")
async def get_user(user_id: int):
    # This will be cached for 1 hour
    return await fetch_user_from_db(user_id)

# Manual cache operations
client = await create_redis_client()
await client.set("key", "value", ttl=300)
value = await client.get("key")
```

### Session Management

```python
from chil.caching.session import get_session_manager

session_manager = await get_session_manager()

# Create session
session = await session_manager.create_session(
    user_id="user123",
    data={"role": "admin"},
    ip_address="192.168.1.1"
)

# Get session
session = await session_manager.get_session(session.session_id)

# Update session
await session_manager.update_session(
    session.session_id,
    data={"last_action": "login"}
)
```

### Cache Invalidation

```python
from chil.caching.invalidation import invalidate_cache

# Invalidate specific keys
await invalidate_cache(keys=["user:123", "user:456"])

# Invalidate by pattern
await invalidate_cache(patterns=["user:*", "session:*"])

# Invalidate by tags
await invalidate_cache(tags=["user_data", "profile"])
```

### Pub/Sub Messaging

```python
from chil.caching.pubsub import get_cache_event_broker

broker = await get_cache_event_broker()

# Subscribe to cache events
async def handle_invalidation(message):
    print(f"Cache invalidated: {message}")

await broker.on_cache_invalidation(handle_invalidation)

# Emit cache events
await broker.emit_invalidation(["user:123"], strategy="immediate")
```

## Configuration

### Environment Variables

```bash
# Redis connection
export REDIS_HOST=localhost
export REDIS_PORT=6379
export REDIS_DB=0
export REDIS_PASSWORD=your_password
export REDIS_SSL=false

# Or use Redis URL
export REDIS_URL=redis://user:password@localhost:6379/0

# Cache settings
export CACHE_DEFAULT_TTL=3600
export CACHE_SERIALIZER=json
export CACHE_COMPRESSION=true
export CACHE_KEY_PREFIX=brahminykite
```

### Programmatic Configuration

```python
from chil.caching import RedisConfig, CacheConfig

# Redis configuration
redis_config = RedisConfig(
    host="redis.example.com",
    port=6379,
    password="secret",
    max_connections=50,
    ssl=True
)

# Cache configuration
cache_config = CacheConfig(
    default_ttl=3600,
    serializer="json",
    compression=True,
    key_prefix="myapp"
)
```

## Advanced Usage

### Custom Caching Patterns

```python
from chil.caching.patterns import WriteThrough, RefreshAhead

# Write-through caching
write_through = WriteThrough()

async def save_user(user_data):
    await write_through.set(
        f"user:{user_data['id']}",
        user_data,
        lambda data: save_to_database(data)
    )

# Refresh-ahead caching
refresh_ahead = RefreshAhead(refresh_threshold=0.8)

@cache_async(ttl=3600)
async def get_expensive_data(key):
    return await refresh_ahead.get(
        key,
        lambda: fetch_expensive_data(key),
        ttl=3600
    )
```

### Cache Monitoring

```python
from chil.caching.monitoring import start_monitoring, get_cache_metrics

# Start monitoring
await start_monitoring()

# Get metrics
metrics = get_cache_metrics()
print(f"Hit rate: {metrics['hit_rate']:.2%}")
```

### Conditional Caching

```python
from chil.caching.decorators import conditional_cache

def should_cache(user_id: int, force_refresh: bool = False):
    return not force_refresh and user_id > 0

@conditional_cache(should_cache)
@cache_async(ttl=1800)
async def get_user_profile(user_id: int, force_refresh: bool = False):
    return await fetch_profile(user_id)
```

## Architecture

### Components

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Application   │    │  Cache Layer    │    │  Redis Cluster  │
│                 │    │                 │    │                 │
│  @cache_async   │───▶│  RedisClient    │───▶│   Primary       │
│  get_user()     │    │  Serializers    │    │   Replicas      │
│                 │    │  Patterns       │    │   Sentinel      │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Sessions      │    │  Invalidation   │    │   Pub/Sub       │
│                 │    │                 │    │                 │
│  Session Mgr    │    │  Strategies     │    │  Channels       │
│  User Auth      │    │  Dependencies   │    │  Message Bus    │
│  Security       │    │  Distributed    │    │  Events         │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

### Data Flow

1. **Cache Miss**: Application requests data → Cache miss → Fetch from source → Store in cache → Return data
2. **Cache Hit**: Application requests data → Cache hit → Return cached data
3. **Invalidation**: Data changes → Invalidation triggered → Keys removed/marked invalid → Pub/sub notification
4. **Session**: User login → Session created → Stored in Redis → Session validated on requests

## Performance Optimization

### Connection Pooling
- Configure appropriate pool sizes based on load
- Monitor connection usage and adjust limits
- Use persistent connections for better performance

### Serialization
- Choose appropriate serializer for your data types
- Use compression for large objects
- Consider binary formats (MessagePack) for performance

### TTL Management
- Set appropriate TTLs based on data volatility
- Use different TTLs for different data types
- Monitor and tune TTL values based on usage patterns

### Memory Management
- Monitor Redis memory usage
- Use appropriate eviction policies
- Consider data compression for large values

## Security

### Data Protection
- Enable Redis AUTH with strong passwords
- Use SSL/TLS for connections in production
- Encrypt sensitive session data
- Validate session IP addresses

### Network Security
- Restrict Redis network access
- Use VPC/private networks
- Configure firewall rules
- Monitor for suspicious access patterns

### Access Control
- Implement role-based session management
- Limit concurrent sessions per user
- Log security events and anomalies
- Regular session cleanup and rotation

## Monitoring

### Key Metrics
- **Hit Rate**: Percentage of cache hits vs misses
- **Response Time**: Cache operation latency
- **Memory Usage**: Redis memory consumption
- **Connection Count**: Active Redis connections
- **Session Count**: Active user sessions

### Alerting
- Cache hit rate below threshold
- High memory usage
- Connection pool exhaustion
- Session security violations
- Redis connectivity issues

### Dashboards
- Real-time cache performance metrics
- Session analytics and user activity
- Redis server health and resources
- Cache invalidation patterns

## Best Practices

### Cache Key Design
- Use consistent naming conventions
- Include version numbers when needed
- Keep keys short but descriptive
- Use hierarchical key structures

### TTL Strategy
- Set TTLs for all cache entries
- Use shorter TTLs for volatile data
- Longer TTLs for stable reference data
- Monitor TTL effectiveness

### Error Handling
- Graceful degradation on cache failures
- Fallback to source systems
- Log but don't fail on cache errors
- Use circuit breakers for resilience

### Testing
- Test cache behavior in unit tests
- Mock Redis for fast test execution
- Integration tests with real Redis
- Load testing with realistic data patterns

## Troubleshooting

### Common Issues

1. **Cache Misses**
   - Check TTL configuration
   - Verify key naming consistency
   - Monitor invalidation patterns

2. **Memory Issues**
   - Check Redis memory limits
   - Monitor key expiration
   - Review data compression settings

3. **Connection Problems**
   - Verify Redis connectivity
   - Check connection pool settings
   - Monitor network latency

4. **Performance Issues**
   - Profile serialization overhead
   - Check Redis server resources
   - Optimize key patterns and sizes

### Debugging Tools
- Use Redis CLI for manual inspection
- Enable debug logging for cache operations
- Monitor with Redis built-in tools
- Use application performance monitoring