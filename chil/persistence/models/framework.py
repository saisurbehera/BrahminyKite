"""
Framework execution and state models.
"""

from enum import Enum
from typing import Optional
from datetime import datetime

from sqlalchemy import (
    Column, String, Integer, Float, Boolean, Text, JSON,
    ForeignKey, UniqueConstraint, Index, DateTime
)
from sqlalchemy.dialects.postgresql import UUID, ENUM
from sqlalchemy.orm import relationship

from .base import Base, TimestampMixin, UUIDMixin, get_table_name


class ExecutionStatus(str, Enum):
    """Execution status enumeration."""
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class FrameworkExecution(Base, UUIDMixin, TimestampMixin):
    """Represents a framework execution instance."""
    
    __tablename__ = get_table_name("FrameworkExecution")
    
    # Execution identification
    execution_id = Column(String(255), unique=True, nullable=False, index=True)
    framework_name = Column(String(100), nullable=False, index=True)
    version = Column(String(50), nullable=False)
    
    # Input/Output
    input_data = Column(JSON, nullable=False)
    output_data = Column(JSON, nullable=True)
    
    # Execution details
    status = Column(
        ENUM(ExecutionStatus, name="execution_status"),
        nullable=False,
        default=ExecutionStatus.QUEUED,
        index=True
    )
    
    # Performance metrics
    start_time = Column(DateTime(timezone=True), nullable=True)
    end_time = Column(DateTime(timezone=True), nullable=True)
    execution_time_ms = Column(Integer, nullable=True)
    
    # Resource usage
    cpu_usage_percent = Column(Float, nullable=True)
    memory_usage_mb = Column(Float, nullable=True)
    gpu_usage_percent = Column(Float, nullable=True)
    
    # Error handling
    error_message = Column(Text, nullable=True)
    error_type = Column(String(100), nullable=True)
    retry_count = Column(Integer, nullable=False, default=0)
    
    # Metadata
    metadata = Column(JSON, nullable=True, default=dict)
    
    # Relationships
    metrics = relationship("FrameworkMetrics", back_populates="execution", uselist=False)
    
    __table_args__ = (
        Index("idx_framework_executions_framework_status", "framework_name", "status"),
        Index("idx_framework_executions_created_status", "created_at", "status"),
    )


class FrameworkMetrics(Base, UUIDMixin, TimestampMixin):
    """Metrics for framework executions."""
    
    __tablename__ = get_table_name("FrameworkMetrics")
    
    # Metrics identification
    execution_id = Column(UUID(as_uuid=True), ForeignKey(f"{FrameworkExecution.__tablename__}.id"), unique=True)
    
    # Quality metrics
    accuracy = Column(Float, nullable=True)
    precision = Column(Float, nullable=True)
    recall = Column(Float, nullable=True)
    f1_score = Column(Float, nullable=True)
    
    # Performance metrics
    throughput_per_second = Column(Float, nullable=True)
    latency_p50_ms = Column(Float, nullable=True)
    latency_p95_ms = Column(Float, nullable=True)
    latency_p99_ms = Column(Float, nullable=True)
    
    # Resource efficiency
    tokens_processed = Column(Integer, nullable=True)
    tokens_per_second = Column(Float, nullable=True)
    cost_estimate = Column(Float, nullable=True)
    
    # Custom metrics (framework-specific)
    custom_metrics = Column(JSON, nullable=True, default=dict)
    
    # Relationships
    execution = relationship("FrameworkExecution", back_populates="metrics")
    
    __table_args__ = (
        Index("idx_framework_metrics_accuracy", "accuracy"),
    )


class FrameworkConfig(Base, UUIDMixin, TimestampMixin):
    """Framework configuration management."""
    
    __tablename__ = get_table_name("FrameworkConfig")
    
    # Config identification
    framework_name = Column(String(100), nullable=False)
    version = Column(String(50), nullable=False)
    environment = Column(String(50), nullable=False, default="default")  # default, dev, staging, prod
    
    # Configuration
    config_data = Column(JSON, nullable=False)
    
    # Activation
    is_active = Column(Boolean, nullable=False, default=True)
    activated_at = Column(DateTime(timezone=True), nullable=True)
    deactivated_at = Column(DateTime(timezone=True), nullable=True)
    
    # Validation
    is_valid = Column(Boolean, nullable=False, default=True)
    validation_errors = Column(JSON, nullable=True)
    last_validated_at = Column(DateTime(timezone=True), nullable=True)
    
    __table_args__ = (
        UniqueConstraint("framework_name", "version", "environment", name="uq_framework_config_identity"),
        Index("idx_framework_config_active", "framework_name", "is_active"),
    )


class FrameworkState(Base, UUIDMixin, TimestampMixin):
    """Persistent state for frameworks."""
    
    __tablename__ = get_table_name("FrameworkState")
    
    # State identification
    framework_name = Column(String(100), nullable=False)
    state_key = Column(String(255), nullable=False)
    
    # State data
    state_value = Column(JSON, nullable=False)
    state_type = Column(String(50), nullable=False)  # model, cache, checkpoint, etc.
    
    # Versioning
    version = Column(Integer, nullable=False, default=1)
    is_current = Column(Boolean, nullable=False, default=True)
    
    # Metadata
    size_bytes = Column(Integer, nullable=True)
    checksum = Column(String(64), nullable=True)
    expires_at = Column(DateTime(timezone=True), nullable=True)
    
    __table_args__ = (
        UniqueConstraint("framework_name", "state_key", "version", name="uq_framework_state_identity"),
        Index("idx_framework_state_current", "framework_name", "state_key", "is_current"),
        Index("idx_framework_state_expires", "expires_at"),
    )