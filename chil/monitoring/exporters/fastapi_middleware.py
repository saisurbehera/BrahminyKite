"""
FastAPI middleware for automatic metrics collection.

Integrates Prometheus metrics into FastAPI applications.
"""

import time
from typing import Callable
from fastapi import FastAPI, Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import PlainTextResponse
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST

from ..metrics.registry import metrics_registry, DEFAULT_LATENCY_BUCKETS

class FastAPIMetricsMiddleware(BaseHTTPMiddleware):
    """
    FastAPI middleware for collecting HTTP metrics.
    
    Automatically tracks request rates, latency, and status codes.
    """
    
    def __init__(self, app: FastAPI, service_name: str = "api"):
        super().__init__(app)
        self.service_name = service_name
        
        # Set service name in metrics registry
        metrics_registry.set_common_labels(service=service_name)
        
        # HTTP metrics
        self.http_requests = metrics_registry.counter(
            'http_requests_total',
            'Total HTTP requests',
            ['method', 'endpoint', 'status_code']
        )
        
        self.http_request_duration = metrics_registry.histogram(
            'http_request_duration_seconds',
            'HTTP request duration in seconds',
            ['method', 'endpoint'],
            buckets=DEFAULT_LATENCY_BUCKETS
        )
        
        self.http_request_size = metrics_registry.histogram(
            'http_request_size_bytes',
            'HTTP request size in bytes',
            ['method', 'endpoint']
        )
        
        self.http_response_size = metrics_registry.histogram(
            'http_response_size_bytes',
            'HTTP response size in bytes',
            ['method', 'endpoint', 'status_code']
        )
        
        # In-flight requests
        self.http_requests_in_flight = metrics_registry.gauge(
            'http_requests_in_flight',
            'Number of HTTP requests currently being processed',
            ['method', 'endpoint']
        )
        
        # Rate limiting metrics
        self.rate_limit_hits = metrics_registry.counter(
            'rate_limit_hits_total',
            'Total rate limit hits',
            ['endpoint', 'limit_type']
        )
        
        # Authentication metrics
        self.auth_attempts = metrics_registry.counter(
            'auth_attempts_total',
            'Total authentication attempts',
            ['endpoint', 'method', 'status']
        )
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request and collect metrics."""
        method = request.method
        path = request.url.path
        
        # Normalize path (remove IDs and parameters)
        normalized_path = self._normalize_path(path)
        
        # Start timing and increment in-flight counter
        start_time = time.time()
        self.http_requests_in_flight.labels(method=method, endpoint=normalized_path).inc()
        
        # Get request size
        request_size = int(request.headers.get('content-length', 0))
        if request_size > 0:
            self.http_request_size.labels(method=method, endpoint=normalized_path).observe(request_size)
        
        try:
            # Process request
            response = await call_next(request)
            status_code = str(response.status_code)
            
            # Record metrics
            self.http_requests.labels(
                method=method,
                endpoint=normalized_path,
                status_code=status_code
            ).inc()
            
            # Get response size if available
            response_size = self._get_response_size(response)
            if response_size > 0:
                self.http_response_size.labels(
                    method=method,
                    endpoint=normalized_path,
                    status_code=status_code
                ).observe(response_size)
            
            return response
            
        except Exception as e:
            # Record error
            self.http_requests.labels(
                method=method,
                endpoint=normalized_path,
                status_code='500'
            ).inc()
            raise
            
        finally:
            # Record duration and decrement in-flight counter
            duration = time.time() - start_time
            self.http_request_duration.labels(method=method, endpoint=normalized_path).observe(duration)
            self.http_requests_in_flight.labels(method=method, endpoint=normalized_path).dec()
    
    def _normalize_path(self, path: str) -> str:
        """
        Normalize URL path to reduce cardinality.
        
        Replaces dynamic segments like IDs with placeholders.
        """
        # Common patterns to normalize
        import re
        
        # Replace UUIDs
        path = re.sub(r'/[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}', '/{uuid}', path)
        
        # Replace numeric IDs
        path = re.sub(r'/\d+', '/{id}', path)
        
        # Replace query parameters
        if '?' in path:
            path = path.split('?')[0]
        
        return path
    
    def _get_response_size(self, response: Response) -> int:
        """Get response size in bytes."""
        if hasattr(response, 'body') and response.body:
            return len(response.body)
        
        # For streaming responses, we can't easily get the size
        content_length = response.headers.get('content-length')
        if content_length:
            return int(content_length)
        
        return 0
    
    def record_rate_limit_hit(self, endpoint: str, limit_type: str) -> None:
        """Record a rate limit hit."""
        self.rate_limit_hits.labels(endpoint=endpoint, limit_type=limit_type).inc()
    
    def record_auth_attempt(self, endpoint: str, method: str, success: bool) -> None:
        """Record an authentication attempt."""
        status = "success" if success else "failure"
        self.auth_attempts.labels(endpoint=endpoint, method=method, status=status).inc()


def add_metrics_endpoint(app: FastAPI, path: str = "/metrics") -> None:
    """
    Add a metrics endpoint to FastAPI application.
    
    Args:
        app: FastAPI application instance
        path: Path for the metrics endpoint
    """
    
    @app.get(path, include_in_schema=False)
    async def metrics():
        """Prometheus metrics endpoint."""
        return PlainTextResponse(
            generate_latest(metrics_registry.registry),
            media_type=CONTENT_TYPE_LATEST
        )


def add_health_endpoint(app: FastAPI, path: str = "/health") -> None:
    """
    Add a health check endpoint to FastAPI application.
    
    Args:
        app: FastAPI application instance
        path: Path for the health endpoint
    """
    
    @app.get(path, include_in_schema=False)
    async def health():
        """Health check endpoint."""
        return {
            "status": "healthy",
            "timestamp": time.time(),
            "service": metrics_registry.common_labels.get('service', 'unknown')
        }


class CustomMetrics:
    """
    Custom metrics for specific API functionality.
    """
    
    def __init__(self):
        # Verification-specific metrics
        self.verification_requests = metrics_registry.counter(
            'verification_requests_total',
            'Total verification requests',
            ['framework', 'claim_type', 'status']
        )
        
        self.verification_duration = metrics_registry.histogram(
            'verification_request_duration_seconds',
            'Time spent processing verification requests',
            ['framework', 'claim_type'],
            buckets=DEFAULT_LATENCY_BUCKETS
        )
        
        self.consensus_requests = metrics_registry.counter(
            'consensus_requests_total',
            'Total consensus requests',
            ['type', 'status']
        )
        
        self.consensus_duration = metrics_registry.histogram(
            'consensus_duration_seconds',
            'Time spent on consensus decisions',
            ['type'],
            buckets=DEFAULT_LATENCY_BUCKETS
        )
        
        # User metrics
        self.active_users = metrics_registry.gauge(
            'active_users',
            'Number of active users',
            ['user_type']
        )
        
        self.user_requests = metrics_registry.counter(
            'user_requests_total',
            'Total requests by user',
            ['user_id', 'endpoint']
        )
    
    def record_verification_request(
        self,
        framework: str,
        claim_type: str,
        status: str,
        duration: float
    ) -> None:
        """Record a verification request."""
        self.verification_requests.labels(
            framework=framework,
            claim_type=claim_type,
            status=status
        ).inc()
        
        self.verification_duration.labels(
            framework=framework,
            claim_type=claim_type
        ).observe(duration)
    
    def record_consensus_request(
        self,
        consensus_type: str,
        status: str,
        duration: float
    ) -> None:
        """Record a consensus request."""
        self.consensus_requests.labels(
            type=consensus_type,
            status=status
        ).inc()
        
        self.consensus_duration.labels(type=consensus_type).observe(duration)
    
    def set_active_users(self, user_type: str, count: int) -> None:
        """Set the number of active users."""
        self.active_users.labels(user_type=user_type).set(count)
    
    def record_user_request(self, user_id: str, endpoint: str) -> None:
        """Record a user request."""
        self.user_requests.labels(user_id=user_id, endpoint=endpoint).inc()


# Global custom metrics instance
custom_metrics = CustomMetrics()