"""
Repository for verification-related database operations.
"""

import logging
from typing import List, Optional, Dict, Any
from uuid import UUID
from datetime import datetime, timedelta

from sqlalchemy import select, func, and_, or_
from sqlalchemy.ext.asyncio import AsyncSession

from .base import BaseRepository, RepositoryError
from ..models.verification import (
    VerificationRequest,
    VerificationResult,
    VerificationMetrics,
    VerificationAudit,
    VerificationStatus,
    VerificationFramework
)

logger = logging.getLogger(__name__)


class VerificationRequestRepository(BaseRepository[VerificationRequest]):
    """Repository for verification requests."""
    
    def __init__(self, session: AsyncSession):
        super().__init__(VerificationRequest, session)
    
    async def get_pending_requests(
        self,
        limit: int = 10,
        client_id: Optional[str] = None
    ) -> List[VerificationRequest]:
        """Get pending verification requests."""
        filters = {"status": VerificationStatus.PENDING}
        if client_id:
            filters["client_id"] = client_id
        
        return await self.list(
            limit=limit,
            filters=filters,
            order_by="priority",
            order_desc=True
        )
    
    async def get_active_requests(self) -> List[VerificationRequest]:
        """Get all active (pending or in progress) requests."""
        query = select(self.model).where(
            or_(
                self.model.status == VerificationStatus.PENDING,
                self.model.status == VerificationStatus.IN_PROGRESS
            )
        )
        result = await self.session.execute(query)
        return list(result.scalars().all())
    
    async def get_requests_by_timeframe(
        self,
        start_time: datetime,
        end_time: datetime,
        client_id: Optional[str] = None
    ) -> List[VerificationRequest]:
        """Get requests within a specific timeframe."""
        query = select(self.model).where(
            and_(
                self.model.created_at >= start_time,
                self.model.created_at <= end_time
            )
        )
        
        if client_id:
            query = query.where(self.model.client_id == client_id)
        
        query = query.order_by(self.model.created_at.desc())
        
        result = await self.session.execute(query)
        return list(result.scalars().all())
    
    async def update_status(
        self,
        request_id: Union[UUID, str],
        status: VerificationStatus,
        started_at: Optional[datetime] = None,
        completed_at: Optional[datetime] = None
    ) -> Optional[VerificationRequest]:
        """Update request status with timestamps."""
        updates = {"status": status}
        if started_at:
            updates["started_at"] = started_at
        if completed_at:
            updates["completed_at"] = completed_at
        
        return await self.update(request_id, **updates)
    
    async def get_client_statistics(
        self,
        client_id: str,
        days: int = 30
    ) -> Dict[str, Any]:
        """Get verification statistics for a client."""
        since = datetime.utcnow() - timedelta(days=days)
        
        # Get counts by status
        status_counts = await self.session.execute(
            select(
                self.model.status,
                func.count(self.model.id).label("count")
            )
            .where(
                and_(
                    self.model.client_id == client_id,
                    self.model.created_at >= since
                )
            )
            .group_by(self.model.status)
        )
        
        # Get average completion time
        avg_time = await self.session.execute(
            select(
                func.avg(
                    func.extract("epoch", self.model.completed_at) -
                    func.extract("epoch", self.model.started_at)
                ).label("avg_seconds")
            )
            .where(
                and_(
                    self.model.client_id == client_id,
                    self.model.status == VerificationStatus.COMPLETED,
                    self.model.created_at >= since
                )
            )
        )
        
        return {
            "client_id": client_id,
            "period_days": days,
            "status_counts": {row.status: row.count for row in status_counts},
            "avg_completion_time_seconds": avg_time.scalar()
        }


class VerificationResultRepository(BaseRepository[VerificationResult]):
    """Repository for verification results."""
    
    def __init__(self, session: AsyncSession):
        super().__init__(VerificationResult, session)
    
    async def get_request_results(
        self,
        request_id: Union[UUID, str]
    ) -> List[VerificationResult]:
        """Get all results for a verification request."""
        return await self.list(
            filters={"request_id": request_id},
            load_relationships=["request"]
        )
    
    async def get_framework_performance(
        self,
        framework: VerificationFramework,
        days: int = 7
    ) -> Dict[str, Any]:
        """Get performance metrics for a specific framework."""
        since = datetime.utcnow() - timedelta(days=days)
        
        # Get performance stats
        stats = await self.session.execute(
            select(
                func.count(self.model.id).label("total_verifications"),
                func.avg(self.model.score).label("avg_score"),
                func.avg(self.model.confidence).label("avg_confidence"),
                func.sum(func.cast(self.model.passed, Integer)).label("passed_count"),
                func.avg(self.model.execution_time_ms).label("avg_execution_time")
            )
            .where(
                and_(
                    self.model.framework == framework,
                    self.model.created_at >= since
                )
            )
        )
        
        row = stats.first()
        
        return {
            "framework": framework,
            "period_days": days,
            "total_verifications": row.total_verifications or 0,
            "avg_score": float(row.avg_score) if row.avg_score else 0.0,
            "avg_confidence": float(row.avg_confidence) if row.avg_confidence else 0.0,
            "pass_rate": (row.passed_count / row.total_verifications) if row.total_verifications else 0.0,
            "avg_execution_time_ms": float(row.avg_execution_time) if row.avg_execution_time else 0.0
        }
    
    async def get_failing_patterns(
        self,
        threshold: float = 0.5,
        min_occurrences: int = 5
    ) -> List[Dict[str, Any]]:
        """Identify common failing patterns."""
        # This would analyze findings to identify patterns
        # For now, return a simplified version
        query = select(
            self.model.framework,
            func.count(self.model.id).label("count"),
            func.avg(self.model.score).label("avg_score")
        ).where(
            self.model.score < threshold
        ).group_by(
            self.model.framework
        ).having(
            func.count(self.model.id) >= min_occurrences
        )
        
        result = await self.session.execute(query)
        
        return [
            {
                "framework": row.framework,
                "failure_count": row.count,
                "avg_score": float(row.avg_score)
            }
            for row in result
        ]


class VerificationMetricsRepository(BaseRepository[VerificationMetrics]):
    """Repository for verification metrics."""
    
    def __init__(self, session: AsyncSession):
        super().__init__(VerificationMetrics, session)
    
    async def get_high_risk_requests(
        self,
        risk_levels: List[str] = ["high", "critical"],
        days: int = 7
    ) -> List[VerificationMetrics]:
        """Get metrics for high-risk verification requests."""
        since = datetime.utcnow() - timedelta(days=days)
        
        return await self.list(
            filters={
                "risk_level": risk_levels,
                "created_at": {"gte": since}
            },
            load_relationships=["request"]
        )
    
    async def get_risk_distribution(
        self,
        days: int = 30
    ) -> Dict[str, int]:
        """Get distribution of risk levels."""
        since = datetime.utcnow() - timedelta(days=days)
        
        result = await self.session.execute(
            select(
                self.model.risk_level,
                func.count(self.model.id).label("count")
            )
            .where(self.model.created_at >= since)
            .group_by(self.model.risk_level)
        )
        
        return {row.risk_level: row.count for row in result}


class VerificationAuditRepository(BaseRepository[VerificationAudit]):
    """Repository for verification audit logs."""
    
    def __init__(self, session: AsyncSession):
        super().__init__(VerificationAudit, session)
    
    async def log_event(
        self,
        event_type: str,
        actor: str,
        event_data: Dict[str, Any],
        request_id: Optional[Union[UUID, str]] = None,
        severity: str = "info"
    ) -> VerificationAudit:
        """Log an audit event."""
        return await self.create(
            request_id=request_id,
            event_type=event_type,
            actor=actor,
            event_data=event_data,
            severity=severity
        )
    
    async def get_request_audit_trail(
        self,
        request_id: Union[UUID, str]
    ) -> List[VerificationAudit]:
        """Get complete audit trail for a request."""
        return await self.list(
            filters={"request_id": request_id},
            order_by="created_at",
            order_desc=False
        )
    
    async def get_security_events(
        self,
        severity_levels: List[str] = ["warning", "error", "critical"],
        hours: int = 24
    ) -> List[VerificationAudit]:
        """Get recent security-related events."""
        since = datetime.utcnow() - timedelta(hours=hours)
        
        return await self.list(
            filters={
                "severity": severity_levels,
                "created_at": {"gte": since}
            },
            order_by="created_at",
            order_desc=True
        )


class VerificationRepository:
    """
    Composite repository for all verification-related operations.
    
    Provides a unified interface for verification data access.
    """
    
    def __init__(self, session: AsyncSession):
        self.session = session
        self.requests = VerificationRequestRepository(session)
        self.results = VerificationResultRepository(session)
        self.metrics = VerificationMetricsRepository(session)
        self.audit = VerificationAuditRepository(session)
    
    async def create_verification_request(
        self,
        request_id: str,
        client_id: str,
        content: str,
        content_type: str,
        frameworks: List[VerificationFramework],
        context: Optional[Dict[str, Any]] = None,
        priority: int = 5,
        timeout_seconds: int = 300
    ) -> VerificationRequest:
        """Create a new verification request with audit logging."""
        # Create the request
        request = await self.requests.create(
            request_id=request_id,
            client_id=client_id,
            content=content,
            content_type=content_type,
            frameworks=frameworks,
            context=context or {},
            priority=priority,
            timeout_seconds=timeout_seconds
        )
        
        # Log the creation
        await self.audit.log_event(
            event_type="request_created",
            actor=client_id,
            event_data={
                "request_id": request_id,
                "frameworks": frameworks,
                "content_type": content_type
            },
            request_id=request.id
        )
        
        return request
    
    async def complete_verification(
        self,
        request_id: Union[UUID, str],
        framework_results: List[Dict[str, Any]]
    ) -> VerificationMetrics:
        """
        Complete a verification request with results.
        
        Args:
            request_id: The verification request ID
            framework_results: List of framework results containing:
                - framework: VerificationFramework
                - score: float
                - passed: bool
                - confidence: float
                - findings: List[Dict]
                - recommendations: List[Dict]
                - execution_time_ms: int
                - error: Optional[str]
        """
        # Update request status
        request = await self.requests.update_status(
            request_id,
            VerificationStatus.COMPLETED,
            completed_at=datetime.utcnow()
        )
        
        if not request:
            raise RepositoryError(f"Request {request_id} not found")
        
        # Create results
        for result_data in framework_results:
            await self.results.create(
                request_id=request.id,
                **result_data
            )
        
        # Calculate metrics
        scores = [r["score"] for r in framework_results]
        passed_count = sum(1 for r in framework_results if r["passed"])
        
        # Determine risk level
        avg_score = sum(scores) / len(scores)
        if avg_score < 0.3:
            risk_level = "critical"
        elif avg_score < 0.5:
            risk_level = "high"
        elif avg_score < 0.7:
            risk_level = "medium"
        else:
            risk_level = "low"
        
        # Create metrics
        metrics = await self.metrics.create(
            request_id=request.id,
            overall_score=avg_score,
            min_score=min(scores),
            max_score=max(scores),
            avg_score=avg_score,
            frameworks_passed=passed_count,
            frameworks_failed=len(framework_results) - passed_count,
            frameworks_total=len(framework_results),
            total_execution_time_ms=sum(r["execution_time_ms"] for r in framework_results),
            avg_execution_time_ms=sum(r["execution_time_ms"] for r in framework_results) // len(framework_results),
            risk_level=risk_level,
            risk_factors=[r["framework"] for r in framework_results if r["score"] < 0.5]
        )
        
        # Log completion
        await self.audit.log_event(
            event_type="request_completed",
            actor="system",
            event_data={
                "overall_score": avg_score,
                "risk_level": risk_level,
                "frameworks_completed": len(framework_results)
            },
            request_id=request.id
        )
        
        return metrics