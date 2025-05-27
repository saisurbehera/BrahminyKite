# Redis Management Scripts

This directory contains utility scripts for managing the Redis caching infrastructure.

## Scripts

### redis_health.py
Monitor Redis health and performance.

```bash
# One-time health check
python scripts/redis/redis_health.py

# Continuous monitoring
python scripts/redis/redis_health.py --monitor

# Detailed Redis information
python scripts/redis/redis_health.py --info

# JSON output
python scripts/redis/redis_health.py --json
```

### cache_manager.py
Manage cache keys, patterns, and data.

```bash
# List cache keys
python scripts/redis/cache_manager.py list
python scripts/redis/cache_manager.py list --pattern "user:*"

# Get key information
python scripts/redis/cache_manager.py info "user:123"

# Clear cache by pattern
python scripts/redis/cache_manager.py clear "temp:*"
python scripts/redis/cache_manager.py clear "temp:*" --force

# Invalidate cache entries
python scripts/redis/cache_manager.py invalidate --keys key1 key2
python scripts/redis/cache_manager.py invalidate --patterns "user:*" "session:*"
python scripts/redis/cache_manager.py invalidate --tags user_data

# Show cache statistics
python scripts/redis/cache_manager.py stats

# Warm cache from file
python scripts/redis/cache_manager.py warm cache_warmup.json
```

Example warmup file (`cache_warmup.json`):
```json
[
  {
    "key": "config:app",
    "value": {"feature_flags": {"new_ui": true}},
    "ttl": 3600
  },
  {
    "key": "user:popular",
    "value": ["user1", "user2", "user3"],
    "ttl": 1800
  }
]
```

### session_manager.py
Manage user sessions stored in Redis.

```bash
# List active sessions
python scripts/redis/session_manager.py list
python scripts/redis/session_manager.py list --user user123

# Get session details
python scripts/redis/session_manager.py info session_id_here

# Create test session
python scripts/redis/session_manager.py create user123
python scripts/redis/session_manager.py create user123 --data '{"role": "admin"}'

# Destroy session
python scripts/redis/session_manager.py destroy session_id_here

# Destroy all user sessions
python scripts/redis/session_manager.py destroy-user user123

# Session analytics
python scripts/redis/session_manager.py analytics
```

## Environment Variables

All scripts use the following environment variables for Redis connection:

```bash
export REDIS_HOST=localhost
export REDIS_PORT=6379
export REDIS_DB=0
export REDIS_PASSWORD=your_password

# Or use a single URL
export REDIS_URL=redis://user:password@localhost:6379/0
```

## Common Operations

### Daily Health Check
```bash
# Quick health check
python scripts/redis/redis_health.py

# Detailed diagnostics
python scripts/redis/redis_health.py --info
```

### Cache Maintenance
```bash
# Check cache statistics
python scripts/redis/cache_manager.py stats

# Clean temporary data
python scripts/redis/cache_manager.py clear "temp:*" --force

# Clear expired keys (automatic, but can be forced)
python scripts/redis/cache_manager.py clear "*" --pattern-match-ttl 0
```

### Session Management
```bash
# Monitor active sessions
python scripts/redis/session_manager.py analytics

# Clean up specific user
python scripts/redis/session_manager.py destroy-user inactive_user123
```

### Performance Monitoring
```bash
# Continuous monitoring with 10-second intervals
python scripts/redis/redis_health.py --monitor --interval 10

# Output metrics in JSON for external monitoring
python scripts/redis/redis_health.py --json > redis_metrics.json
```

## Troubleshooting

### Connection Issues
```bash
# Test basic connectivity
redis-cli -h $REDIS_HOST -p $REDIS_PORT ping

# Check Redis logs
tail -f /var/log/redis/redis-server.log
```

### Memory Issues
```bash
# Check memory usage
python scripts/redis/redis_health.py --info | grep -A5 "Memory"

# Find large keys
redis-cli --bigkeys

# Clear large datasets
python scripts/redis/cache_manager.py clear "large_dataset:*"
```

### Performance Issues
```bash
# Monitor operations per second
python scripts/redis/redis_health.py --monitor

# Check slow queries
redis-cli slowlog get 10

# Analyze key patterns
python scripts/redis/cache_manager.py stats
```

## Best Practices

### Cache Key Naming
- Use consistent prefixes: `user:123`, `session:abc`, `config:feature`
- Include versions when needed: `api:v1:user:123`
- Keep keys short but descriptive

### TTL Management
- Set appropriate TTLs for all keys
- Use different TTLs based on data volatility
- Monitor keys without TTL

### Security
- Use Redis AUTH when available
- Restrict network access to Redis
- Don't store sensitive data in cache keys
- Encrypt sensitive cache values

### Monitoring
- Set up automated health checks
- Monitor memory usage and hit rates
- Track session analytics
- Alert on connection failures

## Integration with Monitoring

### Prometheus Metrics
The Redis infrastructure exports metrics to Prometheus:
- `brahminykite_cache_operations_total`
- `brahminykite_cache_hit_rate`
- `brahminykite_redis_memory_used_bytes`
- `brahminykite_sessions_active_total`

### Grafana Dashboards
Pre-built dashboards are available in `monitoring/grafana/dashboards/`:
- Redis Overview
- Cache Performance
- Session Analytics

### Alerting Rules
Important alerts in `monitoring/prometheus/alerts/`:
- Redis down
- High memory usage (>90%)
- Low hit rate (<70%)
- Connection spikes