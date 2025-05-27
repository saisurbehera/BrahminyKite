"""
Verification-related database models.
"""

from enum import Enum
from typing import Optional
from datetime import datetime

from sqlalchemy import (
    Column, String, Integer, Float, Boolean, Text, JSON,
    ForeignKey, UniqueConstraint, Index, DateTime
)
from sqlalchemy.dialects.postgresql import UUID, ENUM, ARRAY
from sqlalchemy.orm import relationship

from .base import Base, TimestampMixin, UUIDMixin, get_table_name


class VerificationStatus(str, Enum):
    """Verification status enumeration."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"


class VerificationFramework(str, Enum):
    """Verification framework enumeration."""
    CONSISTENCY = "consistency"
    EMPIRICAL = "empirical"
    CONTEXTUAL = "contextual"
    POWER_DYNAMICS = "power_dynamics"
    UTILITY = "utility"
    EVOLUTIONARY = "evolutionary"
    META = "meta"


class VerificationRequest(Base, UUIDMixin, TimestampMixin):
    """Represents a verification request."""
    
    __tablename__ = get_table_name("VerificationRequest")
    
    # Request identification
    request_id = Column(String(255), unique=True, nullable=False, index=True)
    client_id = Column(String(255), nullable=False, index=True)
    
    # Content to verify
    content_type = Column(String(50), nullable=False)  # text, code, action, etc.
    content = Column(Text, nullable=False)
    context = Column(JSON, nullable=True, default=dict)
    
    # Verification settings
    frameworks = Column(
        ARRAY(ENUM(VerificationFramework, name="verification_framework")),
        nullable=False
    )
    priority = Column(Integer, nullable=False, default=5)
    timeout_seconds = Column(Integer, nullable=False, default=300)
    
    # Status
    status = Column(
        ENUM(VerificationStatus, name="verification_status"),
        nullable=False,
        default=VerificationStatus.PENDING,
        index=True
    )
    
    # Timing
    started_at = Column(DateTime(timezone=True), nullable=True)
    completed_at = Column(DateTime(timezone=True), nullable=True)
    
    # Relationships
    results = relationship("VerificationResult", back_populates="request")
    metrics = relationship("VerificationMetrics", back_populates="request", uselist=False)
    
    __table_args__ = (
        Index("idx_verification_requests_status_priority", "status", "priority"),
        Index("idx_verification_requests_client_created", "client_id", "created_at"),
    )


class VerificationResult(Base, UUIDMixin, TimestampMixin):
    """Represents a verification result from a specific framework."""
    
    __tablename__ = get_table_name("VerificationResult")
    
    # Result identification
    request_id = Column(UUID(as_uuid=True), ForeignKey(f"{VerificationRequest.__tablename__}.id"))
    framework = Column(
        ENUM(VerificationFramework, name="verification_framework"),
        nullable=False
    )
    
    # Verification outcome
    score = Column(Float, nullable=False)  # 0.0 to 1.0
    passed = Column(Boolean, nullable=False)
    confidence = Column(Float, nullable=False, default=1.0)
    
    # Details
    findings = Column(JSON, nullable=False, default=list)
    recommendations = Column(JSON, nullable=True, default=list)
    
    # Execution info
    execution_time_ms = Column(Integer, nullable=False)
    error = Column(Text, nullable=True)
    
    # Relationships
    request = relationship("VerificationRequest", back_populates="results")
    
    __table_args__ = (
        UniqueConstraint("request_id", "framework", name="uq_verification_results_request_framework"),
        Index("idx_verification_results_framework_score", "framework", "score"),
    )


class VerificationMetrics(Base, UUIDMixin, TimestampMixin):
    """Aggregated metrics for a verification request."""
    
    __tablename__ = get_table_name("VerificationMetrics")
    
    # Metrics identification
    request_id = Column(UUID(as_uuid=True), ForeignKey(f"{VerificationRequest.__tablename__}.id"), unique=True)
    
    # Aggregate scores
    overall_score = Column(Float, nullable=False)
    min_score = Column(Float, nullable=False)
    max_score = Column(Float, nullable=False)
    avg_score = Column(Float, nullable=False)
    
    # Framework results
    frameworks_passed = Column(Integer, nullable=False, default=0)
    frameworks_failed = Column(Integer, nullable=False, default=0)
    frameworks_total = Column(Integer, nullable=False, default=0)
    
    # Performance metrics
    total_execution_time_ms = Column(Integer, nullable=False)
    avg_execution_time_ms = Column(Integer, nullable=False)
    
    # Risk assessment
    risk_level = Column(String(20), nullable=False, default="low")  # low, medium, high, critical
    risk_factors = Column(JSON, nullable=False, default=list)
    
    # Relationships
    request = relationship("VerificationRequest", back_populates="metrics")
    
    __table_args__ = (
        Index("idx_verification_metrics_risk_score", "risk_level", "overall_score"),
    )


class VerificationAudit(Base, UUIDMixin, TimestampMixin):
    """Audit log for verification activities."""
    
    __tablename__ = get_table_name("VerificationAudit")
    
    # Audit identification
    request_id = Column(UUID(as_uuid=True), ForeignKey(f"{VerificationRequest.__tablename__}.id"), nullable=True)
    
    # Event details
    event_type = Column(String(50), nullable=False)  # request_created, started, completed, failed, etc.
    actor = Column(String(255), nullable=False)  # system, user, framework, etc.
    
    # Event data
    event_data = Column(JSON, nullable=False, default=dict)
    
    # Impact
    severity = Column(String(20), nullable=False, default="info")  # debug, info, warning, error, critical
    
    __table_args__ = (
        Index("idx_verification_audit_request_event", "request_id", "event_type"),
        Index("idx_verification_audit_created_severity", "created_at", "severity"),
    )