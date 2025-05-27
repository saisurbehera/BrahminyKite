"""
Database monitoring and metrics collection.
"""

import logging
import time
from typing import Dict, Any, Optional, Callable
from functools import wraps
import asyncio

from prometheus_client import Counter, Histogram, Gauge, Summary
from sqlalchemy import event
from sqlalchemy.engine import Engine
from sqlalchemy.pool import Pool
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession

from ..monitoring.metrics.registry import MetricsRegistry

logger = logging.getLogger(__name__)

# Initialize metrics registry
metrics = MetricsRegistry(namespace="brahminykite_db")

# Database connection metrics
db_connections_active = metrics.gauge(
    "connections_active",
    "Number of active database connections"
)

db_connections_idle = metrics.gauge(
    "connections_idle", 
    "Number of idle database connections"
)

db_connection_wait_time = metrics.histogram(
    "connection_wait_seconds",
    "Time spent waiting for a database connection",
    buckets=[0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0]
)

# Query metrics
db_queries_total = metrics.counter(
    "queries_total",
    "Total number of database queries",
    ["operation", "table", "status"]
)

db_query_duration = metrics.histogram(
    "query_duration_seconds",
    "Database query duration",
    ["operation", "table"],
    buckets=[0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0]
)

# Transaction metrics
db_transactions_total = metrics.counter(
    "transactions_total",
    "Total number of database transactions",
    ["status"]
)

db_transaction_duration = metrics.histogram(
    "transaction_duration_seconds",
    "Database transaction duration",
    buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0, 30.0]
)

# Error metrics
db_errors_total = metrics.counter(
    "errors_total",
    "Total number of database errors",
    ["error_type", "operation"]
)

# Pool metrics
db_pool_size = metrics.gauge(
    "pool_size",
    "Database connection pool size"
)

db_pool_overflow = metrics.gauge(
    "pool_overflow",
    "Database connection pool overflow"
)

db_pool_checkedout = metrics.gauge(
    "pool_checkedout",
    "Number of connections checked out from pool"
)


class DatabaseMonitoring:
    """Database monitoring functionality."""
    
    def __init__(self):
        self.query_start_times: Dict[int, float] = {}
        self.transaction_start_times: Dict[int, float] = {}
    
    def setup_engine_monitoring(self, engine: Engine) -> None:
        """Set up monitoring for a database engine."""
        # Monitor connection pool
        event.listen(engine.pool, "connect", self._on_connect)
        event.listen(engine.pool, "checkout", self._on_checkout)
        event.listen(engine.pool, "checkin", self._on_checkin)
        
        # Monitor queries
        event.listen(engine, "before_execute", self._before_execute)
        event.listen(engine, "after_execute", self._after_execute)
        
        # Monitor transactions
        event.listen(engine, "begin", self._on_begin)
        event.listen(engine, "commit", self._on_commit)
        event.listen(engine, "rollback", self._on_rollback)
        
        # Update pool metrics periodically
        asyncio.create_task(self._update_pool_metrics(engine))
    
    def _on_connect(self, dbapi_conn, connection_record):
        """Handle new connection creation."""
        logger.debug("New database connection created")
        db_connections_active.inc()
    
    def _on_checkout(self, dbapi_conn, connection_record, connection_proxy):
        """Handle connection checkout from pool."""
        wait_time = time.time() - connection_record.info.get('checkout_time', time.time())
        db_connection_wait_time.observe(wait_time)
        db_connections_idle.dec()
    
    def _on_checkin(self, dbapi_conn, connection_record):
        """Handle connection checkin to pool."""
        connection_record.info['checkout_time'] = time.time()
        db_connections_idle.inc()
    
    def _before_execute(self, conn, clauseelement, multiparams, params, execution_options):
        """Handle before query execution."""
        self.query_start_times[id(conn)] = time.time()
    
    def _after_execute(self, conn, clauseelement, multiparams, params, execution_options, result):
        """Handle after query execution."""
        duration = time.time() - self.query_start_times.pop(id(conn), time.time())
        
        # Extract operation and table info
        operation = self._extract_operation(clauseelement)
        table = self._extract_table(clauseelement)
        
        db_query_duration.labels(operation=operation, table=table).observe(duration)
        db_queries_total.labels(operation=operation, table=table, status="success").inc()
    
    def _on_begin(self, conn):
        """Handle transaction begin."""
        self.transaction_start_times[id(conn)] = time.time()
    
    def _on_commit(self, conn):
        """Handle transaction commit."""
        duration = time.time() - self.transaction_start_times.pop(id(conn), time.time())
        db_transaction_duration.observe(duration)
        db_transactions_total.labels(status="committed").inc()
    
    def _on_rollback(self, conn):
        """Handle transaction rollback."""
        duration = time.time() - self.transaction_start_times.pop(id(conn), time.time())
        db_transaction_duration.observe(duration)
        db_transactions_total.labels(status="rolled_back").inc()
    
    def _extract_operation(self, clauseelement) -> str:
        """Extract operation type from SQL clause."""
        clause_type = type(clauseelement).__name__
        
        if hasattr(clauseelement, 'is_select') and clauseelement.is_select:
            return "SELECT"
        elif hasattr(clauseelement, 'is_insert') and clauseelement.is_insert:
            return "INSERT"
        elif hasattr(clauseelement, 'is_update') and clauseelement.is_update:
            return "UPDATE"
        elif hasattr(clauseelement, 'is_delete') and clauseelement.is_delete:
            return "DELETE"
        else:
            return clause_type.upper()
    
    def _extract_table(self, clauseelement) -> str:
        """Extract table name from SQL clause."""
        try:
            if hasattr(clauseelement, 'table'):
                return clauseelement.table.name
            elif hasattr(clauseelement, 'froms'):
                froms = list(clauseelement.froms)
                if froms and hasattr(froms[0], 'name'):
                    return froms[0].name
        except:
            pass
        
        return "unknown"
    
    async def _update_pool_metrics(self, engine: Engine) -> None:
        """Periodically update pool metrics."""
        while True:
            try:
                pool = engine.pool
                if hasattr(pool, 'size'):
                    db_pool_size.set(pool.size())
                if hasattr(pool, 'overflow'):
                    db_pool_overflow.set(pool.overflow())
                if hasattr(pool, 'checked_out_connections'):
                    db_pool_checkedout.set(pool.checked_out_connections())
            except Exception as e:
                logger.error(f"Error updating pool metrics: {e}")
            
            await asyncio.sleep(10)  # Update every 10 seconds


# Global monitoring instance
monitoring = DatabaseMonitoring()


def monitor_query(operation: str = "query", table: str = "unknown"):
    """
    Decorator to monitor database queries.
    
    Usage:
        @monitor_query(operation="select", table="users")
        async def get_user(session: AsyncSession, user_id: str):
            ...
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            start_time = time.time()
            status = "success"
            
            try:
                result = await func(*args, **kwargs)
                return result
            except Exception as e:
                status = "error"
                error_type = type(e).__name__
                db_errors_total.labels(error_type=error_type, operation=operation).inc()
                raise
            finally:
                duration = time.time() - start_time
                db_query_duration.labels(operation=operation, table=table).observe(duration)
                db_queries_total.labels(operation=operation, table=table, status=status).inc()
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            start_time = time.time()
            status = "success"
            
            try:
                result = func(*args, **kwargs)
                return result
            except Exception as e:
                status = "error"
                error_type = type(e).__name__
                db_errors_total.labels(error_type=error_type, operation=operation).inc()
                raise
            finally:
                duration = time.time() - start_time
                db_query_duration.labels(operation=operation, table=table).observe(duration)
                db_queries_total.labels(operation=operation, table=table, status=status).inc()
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    
    return decorator


def monitor_transaction():
    """
    Decorator to monitor database transactions.
    
    Usage:
        @monitor_transaction()
        async def process_order(session: AsyncSession):
            ...
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            start_time = time.time()
            status = "committed"
            
            try:
                result = await func(*args, **kwargs)
                return result
            except Exception as e:
                status = "rolled_back"
                error_type = type(e).__name__
                db_errors_total.labels(error_type=error_type, operation="transaction").inc()
                raise
            finally:
                duration = time.time() - start_time
                db_transaction_duration.observe(duration)
                db_transactions_total.labels(status=status).inc()
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            start_time = time.time()
            status = "committed"
            
            try:
                result = func(*args, **kwargs)
                return result
            except Exception as e:
                status = "rolled_back"
                error_type = type(e).__name__
                db_errors_total.labels(error_type=error_type, operation="transaction").inc()
                raise
            finally:
                duration = time.time() - start_time
                db_transaction_duration.observe(duration)
                db_transactions_total.labels(status=status).inc()
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    
    return decorator


class DatabaseHealthCollector:
    """Collects database health metrics."""
    
    def __init__(self):
        self.last_health_check: Optional[Dict[str, Any]] = None
        self.health_check_interval = 30  # seconds
    
    async def collect_health_metrics(self, engine_getter: Callable) -> None:
        """Continuously collect database health metrics."""
        while True:
            try:
                engine = await engine_getter()
                health = await engine.health_check()
                
                self.last_health_check = health
                
                # Update health gauge
                health_gauge = metrics.gauge(
                    "health_status",
                    "Database health status (1=healthy, 0=unhealthy)"
                )
                health_gauge.set(1 if health.get("healthy") else 0)
                
                # Update pool metrics
                pool_stats = health.get("pool_stats", {})
                if pool_stats:
                    db_pool_size.set(pool_stats.get("max_size", 0))
                    db_pool_checkedout.set(pool_stats.get("used", 0))
                    db_connections_idle.set(pool_stats.get("free", 0))
                
            except Exception as e:
                logger.error(f"Error collecting health metrics: {e}")
                health_gauge.set(0)
            
            await asyncio.sleep(self.health_check_interval)
    
    def get_last_health_check(self) -> Optional[Dict[str, Any]]:
        """Get the last health check result."""
        return self.last_health_check


# Global health collector
health_collector = DatabaseHealthCollector()


# Monitoring middleware for repositories
class MonitoringRepositoryMixin:
    """Mixin to add monitoring to repository methods."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Wrap common methods with monitoring
        self._wrap_with_monitoring("get", "select")
        self._wrap_with_monitoring("list", "select")
        self._wrap_with_monitoring("create", "insert")
        self._wrap_with_monitoring("update", "update")
        self._wrap_with_monitoring("delete", "delete")
        self._wrap_with_monitoring("bulk_create", "insert")
        self._wrap_with_monitoring("bulk_update", "update")
        self._wrap_with_monitoring("bulk_delete", "delete")
    
    def _wrap_with_monitoring(self, method_name: str, operation: str) -> None:
        """Wrap a method with monitoring."""
        if hasattr(self, method_name):
            original_method = getattr(self, method_name)
            table_name = self.model.__tablename__ if hasattr(self.model, '__tablename__') else "unknown"
            
            monitored_method = monitor_query(operation=operation, table=table_name)(original_method)
            setattr(self, method_name, monitored_method)


def setup_database_monitoring(engine: AsyncEngine) -> None:
    """
    Set up comprehensive database monitoring.
    
    Args:
        engine: The database engine to monitor
    """
    # Set up engine monitoring
    monitoring.setup_engine_monitoring(engine)
    
    # Start health metric collection
    async def get_engine():
        return engine
    
    asyncio.create_task(health_collector.collect_health_metrics(get_engine))
    
    logger.info("Database monitoring initialized")