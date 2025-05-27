"""
Repository for framework-related database operations.
"""

import logging
from typing import List, Optional, Dict, Any, Tuple
from uuid import UUID
from datetime import datetime, timedelta
import json

from sqlalchemy import select, func, and_, or_, desc
from sqlalchemy.ext.asyncio import AsyncSession

from .base import BaseRepository, RepositoryError
from ..models.framework import (
    FrameworkExecution,
    FrameworkMetrics,
    FrameworkConfig,
    FrameworkState,
    ExecutionStatus
)

logger = logging.getLogger(__name__)


class FrameworkExecutionRepository(BaseRepository[FrameworkExecution]):
    """Repository for framework executions."""
    
    def __init__(self, session: AsyncSession):
        super().__init__(FrameworkExecution, session)
    
    async def get_by_execution_id(self, execution_id: str) -> Optional[FrameworkExecution]:
        """Get execution by its unique execution ID."""
        return await self.get_by(execution_id=execution_id)
    
    async def create_execution(
        self,
        execution_id: str,
        framework_name: str,
        version: str,
        input_data: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None
    ) -> FrameworkExecution:
        """Create a new framework execution."""
        return await self.create(
            execution_id=execution_id,
            framework_name=framework_name,
            version=version,
            input_data=input_data,
            status=ExecutionStatus.QUEUED,
            metadata=metadata or {}
        )
    
    async def start_execution(
        self,
        execution_id: Union[UUID, str]
    ) -> Optional[FrameworkExecution]:
        """Mark execution as started."""
        return await self.update(
            execution_id,
            status=ExecutionStatus.RUNNING,
            start_time=datetime.utcnow()
        )
    
    async def complete_execution(
        self,
        execution_id: Union[UUID, str],
        output_data: Dict[str, Any],
        cpu_usage: Optional[float] = None,
        memory_usage: Optional[float] = None,
        gpu_usage: Optional[float] = None
    ) -> Optional[FrameworkExecution]:
        """Mark execution as completed with results."""
        execution = await self.get(execution_id)
        if not execution:
            return None
        
        end_time = datetime.utcnow()
        execution_time_ms = None
        
        if execution.start_time:
            delta = end_time - execution.start_time
            execution_time_ms = int(delta.total_seconds() * 1000)
        
        return await self.update(
            execution_id,
            status=ExecutionStatus.COMPLETED,
            end_time=end_time,
            execution_time_ms=execution_time_ms,
            output_data=output_data,
            cpu_usage_percent=cpu_usage,
            memory_usage_mb=memory_usage,
            gpu_usage_percent=gpu_usage
        )
    
    async def fail_execution(
        self,
        execution_id: Union[UUID, str],
        error_message: str,
        error_type: str
    ) -> Optional[FrameworkExecution]:
        """Mark execution as failed."""
        execution = await self.get(execution_id)
        if not execution:
            return None
        
        end_time = datetime.utcnow()
        execution_time_ms = None
        
        if execution.start_time:
            delta = end_time - execution.start_time
            execution_time_ms = int(delta.total_seconds() * 1000)
        
        return await self.update(
            execution_id,
            status=ExecutionStatus.FAILED,
            end_time=end_time,
            execution_time_ms=execution_time_ms,
            error_message=error_message,
            error_type=error_type,
            retry_count=execution.retry_count + 1
        )
    
    async def get_running_executions(
        self,
        framework_name: Optional[str] = None
    ) -> List[FrameworkExecution]:
        """Get currently running executions."""
        filters = {"status": ExecutionStatus.RUNNING}
        if framework_name:
            filters["framework_name"] = framework_name
        
        return await self.list(
            filters=filters,
            order_by="start_time"
        )
    
    async def get_framework_statistics(
        self,
        framework_name: str,
        hours: int = 24
    ) -> Dict[str, Any]:
        """Get execution statistics for a framework."""
        since = datetime.utcnow() - timedelta(hours=hours)
        
        # Get status counts
        status_counts = await self.session.execute(
            select(
                self.model.status,
                func.count(self.model.id).label("count")
            )
            .where(
                and_(
                    self.model.framework_name == framework_name,
                    self.model.created_at >= since
                )
            )
            .group_by(self.model.status)
        )
        
        # Get performance metrics
        perf_metrics = await self.session.execute(
            select(
                func.avg(self.model.execution_time_ms).label("avg_time"),
                func.min(self.model.execution_time_ms).label("min_time"),
                func.max(self.model.execution_time_ms).label("max_time"),
                func.avg(self.model.cpu_usage_percent).label("avg_cpu"),
                func.avg(self.model.memory_usage_mb).label("avg_memory")
            )
            .where(
                and_(
                    self.model.framework_name == framework_name,
                    self.model.status == ExecutionStatus.COMPLETED,
                    self.model.created_at >= since
                )
            )
        )
        
        perf = perf_metrics.first()
        
        return {
            "framework_name": framework_name,
            "period_hours": hours,
            "status_distribution": {row.status: row.count for row in status_counts},
            "performance": {
                "avg_execution_time_ms": float(perf.avg_time) if perf.avg_time else 0,
                "min_execution_time_ms": float(perf.min_time) if perf.min_time else 0,
                "max_execution_time_ms": float(perf.max_time) if perf.max_time else 0,
                "avg_cpu_usage": float(perf.avg_cpu) if perf.avg_cpu else 0,
                "avg_memory_mb": float(perf.avg_memory) if perf.avg_memory else 0
            }
        }


class FrameworkMetricsRepository(BaseRepository[FrameworkMetrics]):
    """Repository for framework metrics."""
    
    def __init__(self, session: AsyncSession):
        super().__init__(FrameworkMetrics, session)
    
    async def record_metrics(
        self,
        execution_id: Union[UUID, str],
        accuracy: Optional[float] = None,
        precision: Optional[float] = None,
        recall: Optional[float] = None,
        f1_score: Optional[float] = None,
        throughput_per_second: Optional[float] = None,
        latency_p50_ms: Optional[float] = None,
        latency_p95_ms: Optional[float] = None,
        latency_p99_ms: Optional[float] = None,
        tokens_processed: Optional[int] = None,
        tokens_per_second: Optional[float] = None,
        cost_estimate: Optional[float] = None,
        custom_metrics: Optional[Dict[str, Any]] = None
    ) -> FrameworkMetrics:
        """Record metrics for an execution."""
        return await self.create(
            execution_id=execution_id,
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1_score=f1_score,
            throughput_per_second=throughput_per_second,
            latency_p50_ms=latency_p50_ms,
            latency_p95_ms=latency_p95_ms,
            latency_p99_ms=latency_p99_ms,
            tokens_processed=tokens_processed,
            tokens_per_second=tokens_per_second,
            cost_estimate=cost_estimate,
            custom_metrics=custom_metrics or {}
        )
    
    async def get_framework_performance_trends(
        self,
        framework_name: str,
        days: int = 7,
        metric: str = "accuracy"
    ) -> List[Dict[str, Any]]:
        """Get performance trends for a framework over time."""
        since = datetime.utcnow() - timedelta(days=days)
        
        # Join with executions to filter by framework
        query = select(
            func.date_trunc('day', FrameworkExecution.created_at).label("date"),
            func.avg(getattr(self.model, metric)).label("avg_value"),
            func.min(getattr(self.model, metric)).label("min_value"),
            func.max(getattr(self.model, metric)).label("max_value"),
            func.count(self.model.id).label("count")
        ).join(
            FrameworkExecution,
            self.model.execution_id == FrameworkExecution.id
        ).where(
            and_(
                FrameworkExecution.framework_name == framework_name,
                FrameworkExecution.created_at >= since,
                getattr(self.model, metric).isnot(None)
            )
        ).group_by(
            func.date_trunc('day', FrameworkExecution.created_at)
        ).order_by(
            func.date_trunc('day', FrameworkExecution.created_at)
        )
        
        result = await self.session.execute(query)
        
        return [
            {
                "date": row.date.isoformat(),
                "avg_value": float(row.avg_value),
                "min_value": float(row.min_value),
                "max_value": float(row.max_value),
                "sample_count": row.count
            }
            for row in result
        ]


class FrameworkConfigRepository(BaseRepository[FrameworkConfig]):
    """Repository for framework configurations."""
    
    def __init__(self, session: AsyncSession):
        super().__init__(FrameworkConfig, session)
    
    async def get_active_config(
        self,
        framework_name: str,
        version: str,
        environment: str = "default"
    ) -> Optional[FrameworkConfig]:
        """Get the active configuration for a framework."""
        return await self.get_by(
            framework_name=framework_name,
            version=version,
            environment=environment,
            is_active=True
        )
    
    async def create_config(
        self,
        framework_name: str,
        version: str,
        config_data: Dict[str, Any],
        environment: str = "default",
        activate: bool = True
    ) -> FrameworkConfig:
        """Create a new framework configuration."""
        # Deactivate existing configs if activating this one
        if activate:
            await self.bulk_update(
                filters={
                    "framework_name": framework_name,
                    "version": version,
                    "environment": environment,
                    "is_active": True
                },
                updates={
                    "is_active": False,
                    "deactivated_at": datetime.utcnow()
                }
            )
        
        return await self.create(
            framework_name=framework_name,
            version=version,
            environment=environment,
            config_data=config_data,
            is_active=activate,
            activated_at=datetime.utcnow() if activate else None
        )
    
    async def validate_config(
        self,
        config_id: Union[UUID, str],
        is_valid: bool,
        validation_errors: Optional[List[str]] = None
    ) -> Optional[FrameworkConfig]:
        """Update configuration validation status."""
        return await self.update(
            config_id,
            is_valid=is_valid,
            validation_errors=validation_errors,
            last_validated_at=datetime.utcnow()
        )
    
    async def get_config_history(
        self,
        framework_name: str,
        version: str,
        environment: str = "default",
        limit: int = 10
    ) -> List[FrameworkConfig]:
        """Get configuration history for a framework."""
        return await self.list(
            filters={
                "framework_name": framework_name,
                "version": version,
                "environment": environment
            },
            order_by="created_at",
            order_desc=True,
            limit=limit
        )


class FrameworkStateRepository(BaseRepository[FrameworkState]):
    """Repository for framework state management."""
    
    def __init__(self, session: AsyncSession):
        super().__init__(FrameworkState, session)
    
    async def save_state(
        self,
        framework_name: str,
        state_key: str,
        state_value: Any,
        state_type: str,
        size_bytes: Optional[int] = None,
        checksum: Optional[str] = None,
        ttl_seconds: Optional[int] = None
    ) -> FrameworkState:
        """Save framework state with versioning."""
        # Mark previous versions as not current
        await self.bulk_update(
            filters={
                "framework_name": framework_name,
                "state_key": state_key,
                "is_current": True
            },
            updates={"is_current": False}
        )
        
        # Get next version number
        latest = await self.session.execute(
            select(func.max(self.model.version))
            .where(
                and_(
                    self.model.framework_name == framework_name,
                    self.model.state_key == state_key
                )
            )
        )
        
        next_version = (latest.scalar() or 0) + 1
        
        expires_at = None
        if ttl_seconds:
            expires_at = datetime.utcnow() + timedelta(seconds=ttl_seconds)
        
        return await self.create(
            framework_name=framework_name,
            state_key=state_key,
            state_value=state_value,
            state_type=state_type,
            version=next_version,
            is_current=True,
            size_bytes=size_bytes,
            checksum=checksum,
            expires_at=expires_at
        )
    
    async def get_state(
        self,
        framework_name: str,
        state_key: str,
        version: Optional[int] = None
    ) -> Optional[FrameworkState]:
        """Get framework state, optionally by version."""
        filters = {
            "framework_name": framework_name,
            "state_key": state_key
        }
        
        if version is not None:
            filters["version"] = version
        else:
            filters["is_current"] = True
        
        return await self.get_by(**filters)
    
    async def list_states(
        self,
        framework_name: str,
        state_type: Optional[str] = None,
        include_expired: bool = False
    ) -> List[FrameworkState]:
        """List all current states for a framework."""
        filters = {
            "framework_name": framework_name,
            "is_current": True
        }
        
        if state_type:
            filters["state_type"] = state_type
        
        if not include_expired:
            # Add condition to exclude expired states
            now = datetime.utcnow()
            query = select(self.model).where(
                and_(
                    self.model.framework_name == framework_name,
                    self.model.is_current == True,
                    or_(
                        self.model.expires_at.is_(None),
                        self.model.expires_at > now
                    )
                )
            )
            
            if state_type:
                query = query.where(self.model.state_type == state_type)
            
            result = await self.session.execute(query)
            return list(result.scalars().all())
        
        return await self.list(filters=filters)
    
    async def cleanup_expired_states(self) -> int:
        """Remove expired states."""
        now = datetime.utcnow()
        
        return await self.bulk_delete(
            filters={"expires_at": {"lt": now}}
        )
    
    async def get_state_history(
        self,
        framework_name: str,
        state_key: str,
        limit: int = 10
    ) -> List[FrameworkState]:
        """Get version history for a state key."""
        return await self.list(
            filters={
                "framework_name": framework_name,
                "state_key": state_key
            },
            order_by="version",
            order_desc=True,
            limit=limit
        )


class FrameworkRepository:
    """
    Composite repository for all framework-related operations.
    
    Provides a unified interface for framework data access.
    """
    
    def __init__(self, session: AsyncSession):
        self.session = session
        self.executions = FrameworkExecutionRepository(session)
        self.metrics = FrameworkMetricsRepository(session)
        self.configs = FrameworkConfigRepository(session)
        self.states = FrameworkStateRepository(session)
    
    async def execute_framework(
        self,
        framework_name: str,
        version: str,
        execution_id: str,
        input_data: Dict[str, Any],
        config_override: Optional[Dict[str, Any]] = None
    ) -> Tuple[FrameworkExecution, FrameworkConfig]:
        """
        Execute a framework with configuration.
        
        Returns the execution record and configuration used.
        """
        # Get active configuration
        config = await self.configs.get_active_config(
            framework_name,
            version
        )
        
        if not config:
            # Create default config if none exists
            config = await self.configs.create_config(
                framework_name,
                version,
                config_override or {},
                activate=True
            )
        elif config_override:
            # Merge with override if provided
            merged_config = {**config.config_data, **config_override}
            config.config_data = merged_config
        
        # Create execution record
        execution = await self.executions.create_execution(
            execution_id=execution_id,
            framework_name=framework_name,
            version=version,
            input_data=input_data,
            metadata={"config_id": str(config.id)}
        )
        
        return execution, config
    
    async def record_execution_result(
        self,
        execution_id: str,
        success: bool,
        output_data: Optional[Dict[str, Any]] = None,
        error_message: Optional[str] = None,
        error_type: Optional[str] = None,
        metrics: Optional[Dict[str, Any]] = None,
        resource_usage: Optional[Dict[str, float]] = None
    ) -> Tuple[FrameworkExecution, Optional[FrameworkMetrics]]:
        """
        Record the result of a framework execution.
        
        Returns the updated execution and optionally created metrics.
        """
        if success:
            execution = await self.executions.complete_execution(
                execution_id,
                output_data or {},
                cpu_usage=resource_usage.get("cpu") if resource_usage else None,
                memory_usage=resource_usage.get("memory") if resource_usage else None,
                gpu_usage=resource_usage.get("gpu") if resource_usage else None
            )
        else:
            execution = await self.executions.fail_execution(
                execution_id,
                error_message or "Unknown error",
                error_type or "UnknownError"
            )
        
        # Record metrics if provided
        metrics_record = None
        if metrics and execution:
            metrics_record = await self.metrics.record_metrics(
                execution_id=execution.id,
                **metrics
            )
        
        return execution, metrics_record
    
    async def checkpoint_state(
        self,
        framework_name: str,
        checkpoint_data: Dict[str, Dict[str, Any]],
        ttl_seconds: Optional[int] = None
    ) -> List[FrameworkState]:
        """
        Save multiple state values as a checkpoint.
        
        Args:
            framework_name: Name of the framework
            checkpoint_data: Dict mapping state_key to state data
            ttl_seconds: Optional TTL for all states
        
        Returns:
            List of created state records
        """
        states = []
        
        for state_key, state_info in checkpoint_data.items():
            state = await self.states.save_state(
                framework_name=framework_name,
                state_key=state_key,
                state_value=state_info.get("value"),
                state_type=state_info.get("type", "checkpoint"),
                size_bytes=state_info.get("size_bytes"),
                checksum=state_info.get("checksum"),
                ttl_seconds=ttl_seconds
            )
            states.append(state)
        
        return states
    
    async def restore_checkpoint(
        self,
        framework_name: str,
        state_keys: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Restore framework state from checkpoint.
        
        Args:
            framework_name: Name of the framework
            state_keys: Optional list of specific keys to restore
        
        Returns:
            Dict mapping state_key to state value
        """
        if state_keys:
            # Restore specific keys
            checkpoint = {}
            for key in state_keys:
                state = await self.states.get_state(framework_name, key)
                if state:
                    checkpoint[key] = state.state_value
        else:
            # Restore all current states
            states = await self.states.list_states(framework_name)
            checkpoint = {
                state.state_key: state.state_value
                for state in states
            }
        
        return checkpoint