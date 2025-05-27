"""
Database configuration and connection management.

Handles PostgreSQL connection pooling, configuration, and lifecycle management.
"""

import os
from typing import Optional, Dict, Any
from dataclasses import dataclass, field
import logging
from urllib.parse import quote_plus

logger = logging.getLogger(__name__)


@dataclass
class DatabaseConfig:
    """Database configuration settings."""
    
    # Connection settings
    host: str = "localhost"
    port: int = 5432
    database: str = "brahminykite"
    user: str = "brahminykite"
    password: str = ""
    
    # Pool settings
    pool_size: int = 20
    max_overflow: int = 10
    pool_timeout: float = 30.0
    pool_recycle: int = 3600  # 1 hour
    
    # Connection settings
    connect_timeout: int = 10
    command_timeout: int = 30
    
    # SSL settings
    ssl_mode: str = "prefer"  # disable, allow, prefer, require, verify-ca, verify-full
    ssl_cert: Optional[str] = None
    ssl_key: Optional[str] = None
    ssl_rootcert: Optional[str] = None
    
    # Application settings
    echo: bool = False  # SQL query logging
    echo_pool: bool = False  # Connection pool logging
    
    # Schema settings
    schema: str = "public"
    
    # Additional options
    options: Dict[str, Any] = field(default_factory=dict)
    
    @classmethod
    def from_env(cls) -> "DatabaseConfig":
        """Create configuration from environment variables."""
        return cls(
            host=os.getenv("POSTGRES_HOST", "localhost"),
            port=int(os.getenv("POSTGRES_PORT", "5432")),
            database=os.getenv("POSTGRES_DB", "brahminykite"),
            user=os.getenv("POSTGRES_USER", "brahminykite"),
            password=os.getenv("POSTGRES_PASSWORD", ""),
            pool_size=int(os.getenv("POSTGRES_POOL_SIZE", "20")),
            max_overflow=int(os.getenv("POSTGRES_MAX_OVERFLOW", "10")),
            ssl_mode=os.getenv("POSTGRES_SSL_MODE", "prefer"),
            echo=os.getenv("POSTGRES_ECHO", "false").lower() == "true",
        )
    
    @property
    def sync_url(self) -> str:
        """Get synchronous database URL for psycopg2."""
        password = quote_plus(self.password) if self.password else ""
        url = f"postgresql://{self.user}:{password}@{self.host}:{self.port}/{self.database}"
        
        # Add SSL parameters
        params = []
        if self.ssl_mode != "disable":
            params.append(f"sslmode={self.ssl_mode}")
            if self.ssl_cert:
                params.append(f"sslcert={self.ssl_cert}")
            if self.ssl_key:
                params.append(f"sslkey={self.ssl_key}")
            if self.ssl_rootcert:
                params.append(f"sslrootcert={self.ssl_rootcert}")
        
        if params:
            url += "?" + "&".join(params)
        
        return url
    
    @property
    def async_url(self) -> str:
        """Get asynchronous database URL for asyncpg."""
        password = quote_plus(self.password) if self.password else ""
        return f"postgresql+asyncpg://{self.user}:{password}@{self.host}:{self.port}/{self.database}"
    
    def to_asyncpg_kwargs(self) -> Dict[str, Any]:
        """Convert to asyncpg connection kwargs."""
        kwargs = {
            "host": self.host,
            "port": self.port,
            "database": self.database,
            "user": self.user,
            "password": self.password,
            "min_size": self.pool_size,
            "max_size": self.pool_size + self.max_overflow,
            "timeout": self.connect_timeout,
            "command_timeout": self.command_timeout,
        }
        
        # SSL configuration
        if self.ssl_mode != "disable":
            ssl_context = True
            if self.ssl_mode in ["verify-ca", "verify-full"]:
                import ssl
                ssl_context = ssl.create_default_context(ssl.Purpose.SERVER_AUTH)
                if self.ssl_rootcert:
                    ssl_context.load_verify_locations(self.ssl_rootcert)
                if self.ssl_cert and self.ssl_key:
                    ssl_context.load_cert_chain(self.ssl_cert, self.ssl_key)
            kwargs["ssl"] = ssl_context
        
        return kwargs
    
    def to_sqlalchemy_kwargs(self) -> Dict[str, Any]:
        """Convert to SQLAlchemy engine kwargs."""
        return {
            "pool_size": self.pool_size,
            "max_overflow": self.max_overflow,
            "pool_timeout": self.pool_timeout,
            "pool_recycle": self.pool_recycle,
            "pool_pre_ping": True,  # Verify connections before use
            "echo": self.echo,
            "echo_pool": self.echo_pool,
            "connect_args": {
                "connect_timeout": self.connect_timeout,
                "options": f"-c search_path={self.schema}",
                **self.options
            }
        }


class DatabaseConfigError(Exception):
    """Database configuration error."""
    pass


def validate_config(config: DatabaseConfig) -> None:
    """
    Validate database configuration.
    
    Raises:
        DatabaseConfigError: If configuration is invalid
    """
    if not config.host:
        raise DatabaseConfigError("Database host is required")
    
    if not config.database:
        raise DatabaseConfigError("Database name is required")
    
    if not config.user:
        raise DatabaseConfigError("Database user is required")
    
    if config.port < 1 or config.port > 65535:
        raise DatabaseConfigError(f"Invalid port: {config.port}")
    
    if config.pool_size < 1:
        raise DatabaseConfigError(f"Pool size must be at least 1: {config.pool_size}")
    
    if config.max_overflow < 0:
        raise DatabaseConfigError(f"Max overflow cannot be negative: {config.max_overflow}")
    
    valid_ssl_modes = ["disable", "allow", "prefer", "require", "verify-ca", "verify-full"]
    if config.ssl_mode not in valid_ssl_modes:
        raise DatabaseConfigError(f"Invalid SSL mode: {config.ssl_mode}. Must be one of {valid_ssl_modes}")