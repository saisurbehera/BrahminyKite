"""Security monitoring and auditing functionality."""

import asyncio
import json
import logging
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Callable
from dataclasses import dataclass, field, asdict
from enum import Enum
from collections import defaultdict, deque

from prometheus_client import Counter, Histogram, Gauge, Summary

logger = logging.getLogger(__name__)


class SecurityEventType(Enum):
    """Types of security events."""
    AUTH_SUCCESS = "auth_success"
    AUTH_FAILURE = "auth_failure"
    CERT_VALIDATION_SUCCESS = "cert_validation_success"
    CERT_VALIDATION_FAILURE = "cert_validation_failure"
    CERT_EXPIRED = "cert_expired"
    CERT_RENEWED = "cert_renewed"
    CERT_ROTATED = "cert_rotated"
    TLS_HANDSHAKE_SUCCESS = "tls_handshake_success"
    TLS_HANDSHAKE_FAILURE = "tls_handshake_failure"
    RATE_LIMIT_EXCEEDED = "rate_limit_exceeded"
    SUSPICIOUS_ACTIVITY = "suspicious_activity"
    SECURITY_VIOLATION = "security_violation"
    ACCESS_DENIED = "access_denied"
    CIPHER_SUITE_WEAK = "cipher_suite_weak"
    PROTOCOL_DOWNGRADE = "protocol_downgrade"


class SecurityEventSeverity(Enum):
    """Security event severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class SecurityEvent:
    """Represents a security event."""
    event_type: SecurityEventType
    severity: SecurityEventSeverity
    timestamp: datetime
    source_ip: Optional[str] = None
    client_dn: Optional[str] = None
    resource: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        data = asdict(self)
        data['event_type'] = self.event_type.value
        data['severity'] = self.severity.value
        data['timestamp'] = self.timestamp.isoformat()
        return data


@dataclass
class SecurityMetrics:
    """Security metrics container."""
    total_requests: int = 0
    authenticated_requests: int = 0
    failed_auth_attempts: int = 0
    rate_limited_requests: int = 0
    certificate_validations: int = 0
    certificate_failures: int = 0
    tls_handshakes: int = 0
    tls_failures: int = 0
    active_connections: int = 0
    unique_clients: Set[str] = field(default_factory=set)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "total_requests": self.total_requests,
            "authenticated_requests": self.authenticated_requests,
            "failed_auth_attempts": self.failed_auth_attempts,
            "rate_limited_requests": self.rate_limited_requests,
            "certificate_validations": self.certificate_validations,
            "certificate_failures": self.certificate_failures,
            "tls_handshakes": self.tls_handshakes,
            "tls_failures": self.tls_failures,
            "active_connections": self.active_connections,
            "unique_clients": len(self.unique_clients)
        }


class SecurityMonitor:
    """Security monitoring system."""
    
    def __init__(
        self,
        metrics_enabled: bool = True,
        audit_log_path: Optional[Path] = None,
        max_events_memory: int = 10000,
        alert_callbacks: Optional[List[Callable[[SecurityEvent], None]]] = None
    ):
        self.metrics_enabled = metrics_enabled
        self.audit_log_path = audit_log_path
        self.max_events_memory = max_events_memory
        self.alert_callbacks = alert_callbacks or []
        
        # Event storage
        self._events: deque[SecurityEvent] = deque(maxlen=max_events_memory)
        self._event_counts: Dict[SecurityEventType, int] = defaultdict(int)
        self._metrics = SecurityMetrics()
        
        # Prometheus metrics
        if metrics_enabled:
            self._setup_prometheus_metrics()
            
        # Audit logging
        if audit_log_path:
            self._setup_audit_logging()
            
        self._lock = asyncio.Lock()
        
    def _setup_prometheus_metrics(self):
        """Setup Prometheus metrics."""
        self.prom_auth_total = Counter(
            'security_auth_total',
            'Total authentication attempts',
            ['result']
        )
        self.prom_cert_validations = Counter(
            'security_cert_validations_total',
            'Total certificate validations',
            ['result']
        )
        self.prom_tls_handshakes = Counter(
            'security_tls_handshakes_total',
            'Total TLS handshakes',
            ['result', 'version']
        )
        self.prom_security_events = Counter(
            'security_events_total',
            'Total security events',
            ['event_type', 'severity']
        )
        self.prom_active_connections = Gauge(
            'security_active_connections',
            'Number of active secure connections'
        )
        self.prom_unique_clients = Gauge(
            'security_unique_clients',
            'Number of unique clients'
        )
        self.prom_response_time = Histogram(
            'security_operation_duration_seconds',
            'Security operation duration',
            ['operation']
        )
        
    def _setup_audit_logging(self):
        """Setup audit logging."""
        self.audit_log_path.mkdir(parents=True, exist_ok=True)
        
        # Rotate logs daily
        log_file = self.audit_log_path / f"security_audit_{datetime.now().strftime('%Y%m%d')}.jsonl"
        
        # Setup dedicated audit logger
        self.audit_logger = logging.getLogger(f"{__name__}.audit")
        handler = logging.FileHandler(log_file)
        handler.setFormatter(logging.Formatter('%(message)s'))
        self.audit_logger.addHandler(handler)
        self.audit_logger.setLevel(logging.INFO)
        
    async def record_event(self, event: SecurityEvent):
        """Record a security event."""
        async with self._lock:
            # Store event
            self._events.append(event)
            self._event_counts[event.event_type] += 1
            
            # Update metrics
            if self.metrics_enabled:
                self.prom_security_events.labels(
                    event_type=event.event_type.value,
                    severity=event.severity.value
                ).inc()
                
            # Log to audit file
            if self.audit_log_path:
                self.audit_logger.info(json.dumps(event.to_dict()))
                
            # Check for alerts
            await self._check_alerts(event)
            
            # Update specific metrics
            await self._update_metrics(event)
            
    async def _update_metrics(self, event: SecurityEvent):
        """Update metrics based on event."""
        if event.event_type == SecurityEventType.AUTH_SUCCESS:
            self._metrics.authenticated_requests += 1
            if self.metrics_enabled:
                self.prom_auth_total.labels(result='success').inc()
                
        elif event.event_type == SecurityEventType.AUTH_FAILURE:
            self._metrics.failed_auth_attempts += 1
            if self.metrics_enabled:
                self.prom_auth_total.labels(result='failure').inc()
                
        elif event.event_type == SecurityEventType.CERT_VALIDATION_SUCCESS:
            self._metrics.certificate_validations += 1
            if self.metrics_enabled:
                self.prom_cert_validations.labels(result='success').inc()
                
        elif event.event_type == SecurityEventType.CERT_VALIDATION_FAILURE:
            self._metrics.certificate_failures += 1
            if self.metrics_enabled:
                self.prom_cert_validations.labels(result='failure').inc()
                
        elif event.event_type == SecurityEventType.TLS_HANDSHAKE_SUCCESS:
            self._metrics.tls_handshakes += 1
            if self.metrics_enabled:
                version = event.details.get('tls_version', 'unknown')
                self.prom_tls_handshakes.labels(result='success', version=version).inc()
                
        elif event.event_type == SecurityEventType.TLS_HANDSHAKE_FAILURE:
            self._metrics.tls_failures += 1
            if self.metrics_enabled:
                version = event.details.get('tls_version', 'unknown')
                self.prom_tls_handshakes.labels(result='failure', version=version).inc()
                
        elif event.event_type == SecurityEventType.RATE_LIMIT_EXCEEDED:
            self._metrics.rate_limited_requests += 1
            
        # Track unique clients
        if event.source_ip:
            self._metrics.unique_clients.add(event.source_ip)
            if self.metrics_enabled:
                self.prom_unique_clients.set(len(self._metrics.unique_clients))
                
    async def _check_alerts(self, event: SecurityEvent):
        """Check if event should trigger alerts."""
        # Alert on critical events
        if event.severity == SecurityEventSeverity.CRITICAL:
            await self._trigger_alerts(event)
            
        # Alert on repeated failures
        if event.event_type in [
            SecurityEventType.AUTH_FAILURE,
            SecurityEventType.CERT_VALIDATION_FAILURE,
            SecurityEventType.TLS_HANDSHAKE_FAILURE
        ]:
            # Count recent failures from same source
            if event.source_ip:
                recent_failures = sum(
                    1 for e in self._events
                    if e.source_ip == event.source_ip
                    and e.event_type == event.event_type
                    and (event.timestamp - e.timestamp) < timedelta(minutes=5)
                )
                
                if recent_failures >= 5:
                    alert_event = SecurityEvent(
                        event_type=SecurityEventType.SUSPICIOUS_ACTIVITY,
                        severity=SecurityEventSeverity.WARNING,
                        timestamp=datetime.utcnow(),
                        source_ip=event.source_ip,
                        details={
                            "reason": f"Multiple {event.event_type.value} from same source",
                            "count": recent_failures
                        }
                    )
                    await self._trigger_alerts(alert_event)
                    
    async def _trigger_alerts(self, event: SecurityEvent):
        """Trigger alert callbacks."""
        for callback in self.alert_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(event)
                else:
                    callback(event)
            except Exception as e:
                logger.error(f"Alert callback failed: {e}")
                
    async def update_connection_count(self, delta: int):
        """Update active connection count."""
        async with self._lock:
            self._metrics.active_connections += delta
            if self.metrics_enabled:
                self.prom_active_connections.set(self._metrics.active_connections)
                
    async def get_metrics(self) -> SecurityMetrics:
        """Get current metrics."""
        async with self._lock:
            return self._metrics
            
    async def get_recent_events(
        self,
        event_type: Optional[SecurityEventType] = None,
        severity: Optional[SecurityEventSeverity] = None,
        limit: int = 100
    ) -> List[SecurityEvent]:
        """Get recent security events."""
        async with self._lock:
            events = list(self._events)
            
            # Filter by type
            if event_type:
                events = [e for e in events if e.event_type == event_type]
                
            # Filter by severity
            if severity:
                events = [e for e in events if e.severity == severity]
                
            # Return most recent
            return events[-limit:]
            
    async def get_event_summary(self) -> Dict[str, Any]:
        """Get event summary statistics."""
        async with self._lock:
            total_events = len(self._events)
            
            # Count by type
            type_counts = dict(self._event_counts)
            
            # Count by severity
            severity_counts = defaultdict(int)
            for event in self._events:
                severity_counts[event.severity] += 1
                
            # Time-based analysis
            now = datetime.utcnow()
            last_hour_events = sum(
                1 for e in self._events
                if (now - e.timestamp) < timedelta(hours=1)
            )
            
            return {
                "total_events": total_events,
                "events_last_hour": last_hour_events,
                "by_type": {k.value: v for k, v in type_counts.items()},
                "by_severity": {k.value: v for k, v in severity_counts.items()},
                "metrics": self._metrics.to_dict()
            }


class SecurityAuditor:
    """Security auditing system."""
    
    def __init__(
        self,
        monitor: SecurityMonitor,
        compliance_checks: Optional[List[Callable]] = None
    ):
        self.monitor = monitor
        self.compliance_checks = compliance_checks or []
        
    async def audit_authentication(
        self,
        success: bool,
        source_ip: str,
        client_dn: Optional[str] = None,
        method: str = "certificate",
        details: Optional[Dict[str, Any]] = None
    ):
        """Audit authentication attempt."""
        event = SecurityEvent(
            event_type=SecurityEventType.AUTH_SUCCESS if success else SecurityEventType.AUTH_FAILURE,
            severity=SecurityEventSeverity.INFO if success else SecurityEventSeverity.WARNING,
            timestamp=datetime.utcnow(),
            source_ip=source_ip,
            client_dn=client_dn,
            details=details or {"method": method}
        )
        await self.monitor.record_event(event)
        
    async def audit_certificate_validation(
        self,
        success: bool,
        source_ip: str,
        cert_dn: str,
        reason: Optional[str] = None
    ):
        """Audit certificate validation."""
        event = SecurityEvent(
            event_type=SecurityEventType.CERT_VALIDATION_SUCCESS if success else SecurityEventType.CERT_VALIDATION_FAILURE,
            severity=SecurityEventSeverity.INFO if success else SecurityEventSeverity.ERROR,
            timestamp=datetime.utcnow(),
            source_ip=source_ip,
            client_dn=cert_dn,
            details={"reason": reason} if reason else {}
        )
        await self.monitor.record_event(event)
        
    async def audit_tls_handshake(
        self,
        success: bool,
        source_ip: str,
        tls_version: str,
        cipher_suite: str,
        error: Optional[str] = None
    ):
        """Audit TLS handshake."""
        # Check for weak ciphers
        weak_ciphers = ["DES", "RC4", "MD5", "EXPORT"]
        is_weak = any(weak in cipher_suite for weak in weak_ciphers)
        
        if is_weak:
            event = SecurityEvent(
                event_type=SecurityEventType.CIPHER_SUITE_WEAK,
                severity=SecurityEventSeverity.WARNING,
                timestamp=datetime.utcnow(),
                source_ip=source_ip,
                details={
                    "cipher_suite": cipher_suite,
                    "tls_version": tls_version
                }
            )
            await self.monitor.record_event(event)
            
        # Record handshake event
        event = SecurityEvent(
            event_type=SecurityEventType.TLS_HANDSHAKE_SUCCESS if success else SecurityEventType.TLS_HANDSHAKE_FAILURE,
            severity=SecurityEventSeverity.INFO if success else SecurityEventSeverity.ERROR,
            timestamp=datetime.utcnow(),
            source_ip=source_ip,
            details={
                "tls_version": tls_version,
                "cipher_suite": cipher_suite,
                "error": error
            }
        )
        await self.monitor.record_event(event)
        
    async def audit_access(
        self,
        allowed: bool,
        source_ip: str,
        resource: str,
        reason: Optional[str] = None
    ):
        """Audit access attempt."""
        event = SecurityEvent(
            event_type=SecurityEventType.ACCESS_DENIED if not allowed else SecurityEventType.AUTH_SUCCESS,
            severity=SecurityEventSeverity.WARNING if not allowed else SecurityEventSeverity.INFO,
            timestamp=datetime.utcnow(),
            source_ip=source_ip,
            resource=resource,
            details={"reason": reason} if reason else {}
        )
        await self.monitor.record_event(event)
        
    async def run_compliance_check(self) -> Dict[str, Any]:
        """Run compliance checks."""
        results = {}
        
        for check in self.compliance_checks:
            try:
                if asyncio.iscoroutinefunction(check):
                    result = await check(self.monitor)
                else:
                    result = check(self.monitor)
                results[check.__name__] = result
            except Exception as e:
                results[check.__name__] = {"error": str(e)}
                
        return results


class SecurityReporter:
    """Generate security reports."""
    
    def __init__(self, monitor: SecurityMonitor):
        self.monitor = monitor
        
    async def generate_report(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """Generate security report."""
        if not end_time:
            end_time = datetime.utcnow()
        if not start_time:
            start_time = end_time - timedelta(days=1)
            
        # Get events in time range
        events = await self.monitor.get_recent_events(limit=10000)
        filtered_events = [
            e for e in events
            if start_time <= e.timestamp <= end_time
        ]
        
        # Generate statistics
        stats = {
            "period": {
                "start": start_time.isoformat(),
                "end": end_time.isoformat()
            },
            "total_events": len(filtered_events),
            "events_by_type": defaultdict(int),
            "events_by_severity": defaultdict(int),
            "top_sources": defaultdict(int),
            "failed_authentications": 0,
            "certificate_failures": 0,
            "suspicious_activities": 0
        }
        
        for event in filtered_events:
            stats["events_by_type"][event.event_type.value] += 1
            stats["events_by_severity"][event.severity.value] += 1
            
            if event.source_ip:
                stats["top_sources"][event.source_ip] += 1
                
            if event.event_type == SecurityEventType.AUTH_FAILURE:
                stats["failed_authentications"] += 1
            elif event.event_type == SecurityEventType.CERT_VALIDATION_FAILURE:
                stats["certificate_failures"] += 1
            elif event.event_type == SecurityEventType.SUSPICIOUS_ACTIVITY:
                stats["suspicious_activities"] += 1
                
        # Get top sources
        stats["top_sources"] = dict(
            sorted(stats["top_sources"].items(), key=lambda x: x[1], reverse=True)[:10]
        )
        
        # Get current metrics
        metrics = await self.monitor.get_metrics()
        stats["current_metrics"] = metrics.to_dict()
        
        return stats