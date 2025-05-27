"""
Redis configuration management.
"""

import os
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
from urllib.parse import urlparse


@dataclass
class RedisConfig:
    """Redis connection configuration."""
    
    # Connection settings
    host: str = "localhost"
    port: int = 6379
    db: int = 0
    password: Optional[str] = None
    username: Optional[str] = None
    
    # Connection pool settings
    max_connections: int = 50
    min_connections: int = 10
    connection_timeout: int = 20
    socket_timeout: int = 20
    socket_connect_timeout: int = 20
    socket_keepalive: bool = True
    socket_keepalive_options: Optional[Dict[int, int]] = None
    
    # SSL settings
    ssl: bool = False
    ssl_keyfile: Optional[str] = None
    ssl_certfile: Optional[str] = None
    ssl_cert_reqs: str = "required"
    ssl_ca_certs: Optional[str] = None
    ssl_check_hostname: bool = False
    
    # Retry settings
    retry_on_timeout: bool = True
    retry_on_error: Optional[List[type]] = None
    max_retry_attempts: int = 3
    
    # Encoding
    decode_responses: bool = True
    encoding: str = "utf-8"
    
    # Health check
    health_check_interval: int = 30
    
    @classmethod
    def from_url(cls, url: str) -> "RedisConfig":
        """Create configuration from Redis URL."""
        parsed = urlparse(url)
        
        config = cls()
        
        if parsed.hostname:
            config.host = parsed.hostname
        
        if parsed.port:
            config.port = parsed.port
        
        if parsed.password:
            config.password = parsed.password
        
        if parsed.username:
            config.username = parsed.username
        
        # Parse path for db number
        if parsed.path and len(parsed.path) > 1:
            try:
                config.db = int(parsed.path[1:])
            except ValueError:
                pass
        
        # Parse query parameters
        if parsed.query:
            import urllib.parse
            params = urllib.parse.parse_qs(parsed.query)
            
            if "max_connections" in params:
                config.max_connections = int(params["max_connections"][0])
            
            if "ssl" in params:
                config.ssl = params["ssl"][0].lower() in ("true", "1", "yes")
        
        return config
    
    @classmethod
    def from_env(cls, prefix: str = "REDIS") -> "RedisConfig":
        """Create configuration from environment variables."""
        # Check for Redis URL first
        redis_url = os.getenv(f"{prefix}_URL")
        if redis_url:
            return cls.from_url(redis_url)
        
        # Otherwise build from individual settings
        config = cls()
        
        # Connection settings
        if host := os.getenv(f"{prefix}_HOST"):
            config.host = host
        
        if port := os.getenv(f"{prefix}_PORT"):
            config.port = int(port)
        
        if db := os.getenv(f"{prefix}_DB"):
            config.db = int(db)
        
        if password := os.getenv(f"{prefix}_PASSWORD"):
            config.password = password
        
        if username := os.getenv(f"{prefix}_USERNAME"):
            config.username = username
        
        # Pool settings
        if max_conn := os.getenv(f"{prefix}_MAX_CONNECTIONS"):
            config.max_connections = int(max_conn)
        
        if min_conn := os.getenv(f"{prefix}_MIN_CONNECTIONS"):
            config.min_connections = int(min_conn)
        
        # SSL settings
        if ssl := os.getenv(f"{prefix}_SSL"):
            config.ssl = ssl.lower() in ("true", "1", "yes")
        
        if ssl_keyfile := os.getenv(f"{prefix}_SSL_KEYFILE"):
            config.ssl_keyfile = ssl_keyfile
        
        if ssl_certfile := os.getenv(f"{prefix}_SSL_CERTFILE"):
            config.ssl_certfile = ssl_certfile
        
        if ssl_ca_certs := os.getenv(f"{prefix}_SSL_CA_CERTS"):
            config.ssl_ca_certs = ssl_ca_certs
        
        return config
    
    @property
    def url(self) -> str:
        """Get Redis URL representation."""
        auth = ""
        if self.username and self.password:
            auth = f"{self.username}:{self.password}@"
        elif self.password:
            auth = f":{self.password}@"
        
        return f"redis{'s' if self.ssl else ''}://{auth}{self.host}:{self.port}/{self.db}"
    
    def to_redis_kwargs(self) -> Dict[str, Any]:
        """Convert to Redis client kwargs."""
        kwargs = {
            "host": self.host,
            "port": self.port,
            "db": self.db,
            "max_connections": self.max_connections,
            "decode_responses": self.decode_responses,
            "encoding": self.encoding,
            "health_check_interval": self.health_check_interval,
            "socket_timeout": self.socket_timeout,
            "socket_connect_timeout": self.socket_connect_timeout,
            "socket_keepalive": self.socket_keepalive,
            "retry_on_timeout": self.retry_on_timeout,
        }
        
        if self.password:
            kwargs["password"] = self.password
        
        if self.username:
            kwargs["username"] = self.username
        
        if self.socket_keepalive_options:
            kwargs["socket_keepalive_options"] = self.socket_keepalive_options
        
        if self.ssl:
            kwargs["ssl"] = True
            if self.ssl_keyfile:
                kwargs["ssl_keyfile"] = self.ssl_keyfile
            if self.ssl_certfile:
                kwargs["ssl_certfile"] = self.ssl_certfile
            if self.ssl_ca_certs:
                kwargs["ssl_ca_certs"] = self.ssl_ca_certs
            kwargs["ssl_cert_reqs"] = self.ssl_cert_reqs
            kwargs["ssl_check_hostname"] = self.ssl_check_hostname
        
        return kwargs


@dataclass
class CacheConfig:
    """Cache behavior configuration."""
    
    # Default TTL in seconds
    default_ttl: int = 3600  # 1 hour
    
    # Serialization
    serializer: str = "json"  # json, pickle, msgpack
    compression: bool = False
    compression_threshold: int = 1024  # bytes
    
    # Key generation
    key_prefix: str = "brahminykite"
    key_separator: str = ":"
    include_version: bool = True
    version: str = "v1"
    
    # Behavior
    raise_on_error: bool = False
    log_errors: bool = True
    
    # Cache warming
    warm_on_startup: bool = False
    warm_keys: List[str] = field(default_factory=list)
    
    # Invalidation
    invalidation_enabled: bool = True
    invalidation_channel: str = "cache_invalidation"
    
    # Monitoring
    track_hit_rate: bool = True
    track_latency: bool = True
    metrics_sample_rate: float = 1.0  # 100%
    
    @classmethod
    def from_env(cls, prefix: str = "CACHE") -> "CacheConfig":
        """Create configuration from environment variables."""
        config = cls()
        
        if ttl := os.getenv(f"{prefix}_DEFAULT_TTL"):
            config.default_ttl = int(ttl)
        
        if serializer := os.getenv(f"{prefix}_SERIALIZER"):
            config.serializer = serializer
        
        if compression := os.getenv(f"{prefix}_COMPRESSION"):
            config.compression = compression.lower() in ("true", "1", "yes")
        
        if key_prefix := os.getenv(f"{prefix}_KEY_PREFIX"):
            config.key_prefix = key_prefix
        
        if version := os.getenv(f"{prefix}_VERSION"):
            config.version = version
        
        if raise_on_error := os.getenv(f"{prefix}_RAISE_ON_ERROR"):
            config.raise_on_error = raise_on_error.lower() in ("true", "1", "yes")
        
        return config


def validate_redis_config(config: RedisConfig) -> None:
    """Validate Redis configuration."""
    if config.port < 1 or config.port > 65535:
        raise ValueError(f"Invalid port: {config.port}")
    
    if config.db < 0:
        raise ValueError(f"Invalid database number: {config.db}")
    
    if config.max_connections < 1:
        raise ValueError(f"Invalid max_connections: {config.max_connections}")
    
    if config.min_connections < 0:
        raise ValueError(f"Invalid min_connections: {config.min_connections}")
    
    if config.min_connections > config.max_connections:
        raise ValueError("min_connections cannot be greater than max_connections")
    
    if config.connection_timeout < 0:
        raise ValueError(f"Invalid connection_timeout: {config.connection_timeout}")
    
    if config.serializer not in ("json", "pickle", "msgpack"):
        raise ValueError(f"Invalid serializer: {config.serializer}")


def validate_cache_config(config: CacheConfig) -> None:
    """Validate cache configuration."""
    if config.default_ttl < 0:
        raise ValueError(f"Invalid default_ttl: {config.default_ttl}")
    
    if config.compression_threshold < 0:
        raise ValueError(f"Invalid compression_threshold: {config.compression_threshold}")
    
    if config.metrics_sample_rate < 0 or config.metrics_sample_rate > 1:
        raise ValueError(f"Invalid metrics_sample_rate: {config.metrics_sample_rate}")
    
    if not config.key_separator:
        raise ValueError("key_separator cannot be empty")