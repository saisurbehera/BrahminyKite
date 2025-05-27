"""
Database engine and session management.

Provides async database engine creation and session lifecycle management.
"""

import asyncio
import logging
from typing import Optional, AsyncGenerator, Any, Dict
from contextlib import asynccontextmanager
import asyncpg
from sqlalchemy.ext.asyncio import (
    create_async_engine,
    AsyncEngine,
    AsyncSession,
    async_sessionmaker
)
from sqlalchemy.pool import NullPool, AsyncAdaptedQueuePool
from sqlalchemy import event

from .config import DatabaseConfig, validate_config

logger = logging.getLogger(__name__)


class DatabaseEngine:
    """
    Manages database engine and session creation.
    
    Provides both SQLAlchemy and raw asyncpg access for optimal performance.
    """
    
    def __init__(self, config: DatabaseConfig):
        validate_config(config)
        self.config = config
        self._engine: Optional[AsyncEngine] = None
        self._asyncpg_pool: Optional[asyncpg.Pool] = None
        self._session_factory: Optional[async_sessionmaker] = None
        self._initialized = False
    
    async def initialize(self) -> None:
        """Initialize database connections."""
        if self._initialized:
            return
        
        logger.info(f"Initializing database connection to {self.config.host}:{self.config.port}/{self.config.database}")
        
        # Create SQLAlchemy async engine
        self._engine = create_async_engine(
            self.config.async_url,
            **self.config.to_sqlalchemy_kwargs(),
            future=True
        )
        
        # Create session factory
        self._session_factory = async_sessionmaker(
            self._engine,
            class_=AsyncSession,
            expire_on_commit=False
        )
        
        # Create asyncpg pool for raw queries
        self._asyncpg_pool = await asyncpg.create_pool(
            **self.config.to_asyncpg_kwargs()
        )
        
        # Test connection
        try:
            async with self._engine.begin() as conn:
                await conn.execute("SELECT 1")
            logger.info("Database connection established successfully")
        except Exception as e:
            logger.error(f"Failed to connect to database: {e}")
            await self.close()
            raise
        
        self._initialized = True
    
    async def close(self) -> None:
        """Close all database connections."""
        if self._engine:
            await self._engine.dispose()
            self._engine = None
        
        if self._asyncpg_pool:
            await self._asyncpg_pool.close()
            self._asyncpg_pool = None
        
        self._session_factory = None
        self._initialized = False
        logger.info("Database connections closed")
    
    @property
    def engine(self) -> AsyncEngine:
        """Get SQLAlchemy engine."""
        if not self._engine:
            raise RuntimeError("Database engine not initialized")
        return self._engine
    
    @property
    def asyncpg_pool(self) -> asyncpg.Pool:
        """Get asyncpg connection pool."""
        if not self._asyncpg_pool:
            raise RuntimeError("Database pool not initialized")
        return self._asyncpg_pool
    
    @asynccontextmanager
    async def session(self) -> AsyncGenerator[AsyncSession, None]:
        """
        Create a new database session.
        
        Usage:
            async with engine.session() as session:
                # Use session
                await session.execute(...)
        """
        if not self._session_factory:
            raise RuntimeError("Database engine not initialized")
        
        async with self._session_factory() as session:
            try:
                yield session
                await session.commit()
            except Exception:
                await session.rollback()
                raise
            finally:
                await session.close()
    
    @asynccontextmanager
    async def connection(self) -> AsyncGenerator[asyncpg.Connection, None]:
        """
        Get a raw asyncpg connection.
        
        Usage:
            async with engine.connection() as conn:
                # Use connection for raw queries
                await conn.fetch("SELECT * FROM ...")
        """
        if not self._asyncpg_pool:
            raise RuntimeError("Database pool not initialized")
        
        async with self._asyncpg_pool.acquire() as conn:
            yield conn
    
    async def execute(self, query: str, *args, timeout: Optional[float] = None) -> str:
        """Execute a query that doesn't return results."""
        async with self.connection() as conn:
            return await conn.execute(query, *args, timeout=timeout)
    
    async def fetch(self, query: str, *args, timeout: Optional[float] = None) -> list:
        """Fetch multiple rows."""
        async with self.connection() as conn:
            return await conn.fetch(query, *args, timeout=timeout)
    
    async def fetchrow(self, query: str, *args, timeout: Optional[float] = None) -> Optional[asyncpg.Record]:
        """Fetch a single row."""
        async with self.connection() as conn:
            return await conn.fetchrow(query, *args, timeout=timeout)
    
    async def fetchval(self, query: str, *args, column: int = 0, timeout: Optional[float] = None) -> Any:
        """Fetch a single value."""
        async with self.connection() as conn:
            return await conn.fetchval(query, *args, column=column, timeout=timeout)
    
    async def health_check(self) -> Dict[str, Any]:
        """Check database health."""
        try:
            # Check SQLAlchemy engine
            async with self._engine.begin() as conn:
                await conn.execute("SELECT 1")
            
            # Check asyncpg pool
            pool_stats = self._asyncpg_pool.get_stats()
            
            # Get database version
            version = await self.fetchval("SELECT version()")
            
            return {
                "healthy": True,
                "version": version,
                "pool_stats": {
                    "size": pool_stats.get("size", 0),
                    "free": pool_stats.get("free_size", 0),
                    "used": pool_stats.get("size", 0) - pool_stats.get("free_size", 0),
                    "max_size": self.config.pool_size + self.config.max_overflow
                }
            }
        except Exception as e:
            logger.error(f"Database health check failed: {e}")
            return {
                "healthy": False,
                "error": str(e)
            }


# Global engine instance
_engine: Optional[DatabaseEngine] = None
_lock = asyncio.Lock()


async def create_database_engine(config: Optional[DatabaseConfig] = None) -> DatabaseEngine:
    """
    Create or get the global database engine.
    
    Args:
        config: Database configuration. If not provided, loads from environment.
    
    Returns:
        Initialized database engine
    """
    global _engine
    
    async with _lock:
        if _engine is None:
            if config is None:
                config = DatabaseConfig.from_env()
            
            _engine = DatabaseEngine(config)
            await _engine.initialize()
        
        return _engine


async def get_database_engine() -> DatabaseEngine:
    """Get the global database engine."""
    if _engine is None:
        raise RuntimeError("Database engine not initialized. Call create_database_engine first.")
    return _engine


async def close_database_engine() -> None:
    """Close the global database engine."""
    global _engine
    
    async with _lock:
        if _engine:
            await _engine.close()
            _engine = None


# Transaction decorator
def transaction(func):
    """
    Decorator for transactional database operations.
    
    Usage:
        @transaction
        async def my_function(session: AsyncSession, ...):
            # All database operations here are in a transaction
            await session.execute(...)
    """
    async def wrapper(*args, **kwargs):
        engine = await get_database_engine()
        async with engine.session() as session:
            # Inject session as first argument
            return await func(session, *args, **kwargs)
    
    return wrapper