"""
Service-specific metrics for BrahminyKite framework services.

Provides standardized metrics for all gRPC services.
"""

import time
import functools
from typing import Dict, Any, Optional, Callable
from prometheus_client import Counter, Histogram, Gauge, Summary
import grpc
import logging

from .registry import metrics_registry, DEFAULT_LATENCY_BUCKETS

logger = logging.getLogger(__name__)


class ServiceMetrics:
    """
    Metrics collector for individual framework services.
    
    Tracks gRPC method calls, processing times, and service health.
    """
    
    def __init__(self, service_name: str):
        self.service_name = service_name
        
        # gRPC method metrics
        self.grpc_requests = metrics_registry.counter(
            'grpc_requests_total',
            'Total gRPC requests',
            ['service', 'method', 'status']
        )
        
        self.grpc_request_duration = metrics_registry.histogram(
            'grpc_request_duration_seconds',
            'gRPC request duration',
            ['service', 'method'],
            buckets=DEFAULT_LATENCY_BUCKETS
        )
        
        self.grpc_request_size = metrics_registry.histogram(
            'grpc_request_size_bytes',
            'gRPC request message size',
            ['service', 'method']
        )
        
        self.grpc_response_size = metrics_registry.histogram(
            'grpc_response_size_bytes',
            'gRPC response message size',
            ['service', 'method']
        )
        
        # Processing metrics
        self.claims_processed = metrics_registry.counter(
            'claims_processed_total',
            'Total claims processed',
            ['service', 'framework', 'result']
        )
        
        self.verification_duration = metrics_registry.histogram(
            'verification_duration_seconds',
            'Time spent on verification',
            ['service', 'framework', 'claim_type'],
            buckets=DEFAULT_LATENCY_BUCKETS
        )
        
        self.verification_confidence = metrics_registry.histogram(
            'verification_confidence',
            'Confidence scores of verifications',
            ['service', 'framework'],
            buckets=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        )
        
        # Model metrics
        self.model_inference_duration = metrics_registry.histogram(
            'model_inference_duration_seconds',
            'Model inference time',
            ['service', 'model_type'],
            buckets=DEFAULT_LATENCY_BUCKETS
        )
        
        self.model_cache_hits = metrics_registry.counter(
            'model_cache_hits_total',
            'Model cache hits',
            ['service', 'model_type', 'cache_type']
        )
        
        # Connection metrics
        self.active_connections = metrics_registry.gauge(
            'active_connections',
            'Number of active connections',
            ['service', 'connection_type']
        )
        
        # Health metrics
        self.health_status = metrics_registry.gauge(
            'service_health_status',
            'Service health status (1=healthy, 0=unhealthy)',
            ['service']
        )
        
        # Error metrics
        self.errors_total = metrics_registry.counter(
            'errors_total',
            'Total errors by type',
            ['service', 'error_type', 'severity']
        )
    
    def record_grpc_request(
        self,
        method: str,
        status: str,
        duration: float,
        request_size: int = 0,
        response_size: int = 0
    ) -> None:
        """Record gRPC request metrics."""
        self.grpc_requests.labels(
            service=self.service_name,
            method=method,
            status=status
        ).inc()
        
        self.grpc_request_duration.labels(
            service=self.service_name,
            method=method
        ).observe(duration)
        
        if request_size > 0:
            self.grpc_request_size.labels(
                service=self.service_name,
                method=method
            ).observe(request_size)
        
        if response_size > 0:
            self.grpc_response_size.labels(
                service=self.service_name,
                method=method
            ).observe(response_size)
    
    def record_claim_processed(
        self,
        framework: str,
        result: str,
        duration: float,
        confidence: float,
        claim_type: str = "general"
    ) -> None:
        """Record claim processing metrics."""
        self.claims_processed.labels(
            service=self.service_name,
            framework=framework,
            result=result
        ).inc()
        
        self.verification_duration.labels(
            service=self.service_name,
            framework=framework,
            claim_type=claim_type
        ).observe(duration)
        
        self.verification_confidence.labels(
            service=self.service_name,
            framework=framework
        ).observe(confidence)
    
    def record_model_inference(
        self,
        model_type: str,
        duration: float,
        cache_hit: bool = False,
        cache_type: str = "memory"
    ) -> None:
        """Record model inference metrics."""
        self.model_inference_duration.labels(
            service=self.service_name,
            model_type=model_type
        ).observe(duration)
        
        if cache_hit:
            self.model_cache_hits.labels(
                service=self.service_name,
                model_type=model_type,
                cache_type=cache_type
            ).inc()
    
    def set_active_connections(self, connection_type: str, count: int) -> None:
        """Set the number of active connections."""
        self.active_connections.labels(
            service=self.service_name,
            connection_type=connection_type
        ).set(count)
    
    def set_health_status(self, healthy: bool) -> None:
        """Set service health status."""
        self.health_status.labels(service=self.service_name).set(1 if healthy else 0)
    
    def record_error(self, error_type: str, severity: str = "error") -> None:
        """Record an error."""
        self.errors_total.labels(
            service=self.service_name,
            error_type=error_type,
            severity=severity
        ).inc()


def grpc_metrics_interceptor(service_metrics: ServiceMetrics):
    """
    gRPC interceptor for automatic metrics collection.
    """
    
    class MetricsInterceptor(grpc.aio.ServerInterceptor):
        async def intercept_service(self, continuation, handler_call_details):
            method = handler_call_details.method
            start_time = time.time()
            
            try:
                # Continue with the actual service call
                response = await continuation(handler_call_details)
                
                # Record successful request
                duration = time.time() - start_time
                service_metrics.record_grpc_request(
                    method=method,
                    status="success",
                    duration=duration
                )
                
                return response
                
            except grpc.RpcError as e:
                # Record gRPC error
                duration = time.time() - start_time
                service_metrics.record_grpc_request(
                    method=method,
                    status=e.code().name,
                    duration=duration
                )
                service_metrics.record_error("grpc_error", "error")
                raise
                
            except Exception as e:
                # Record general error
                duration = time.time() - start_time
                service_metrics.record_grpc_request(
                    method=method,
                    status="internal_error",
                    duration=duration
                )
                service_metrics.record_error("internal_error", "critical")
                raise
    
    return MetricsInterceptor()


def metrics_decorator(service_metrics: ServiceMetrics, framework: str):
    """
    Decorator for automatic metrics collection on service methods.
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            start_time = time.time()
            
            try:
                result = await func(*args, **kwargs)
                
                # Record successful processing
                duration = time.time() - start_time
                confidence = getattr(result, 'confidence', 0.0) if result else 0.0
                
                service_metrics.record_claim_processed(
                    framework=framework,
                    result="success",
                    duration=duration,
                    confidence=confidence
                )
                
                return result
                
            except Exception as e:
                # Record failed processing
                duration = time.time() - start_time
                service_metrics.record_claim_processed(
                    framework=framework,
                    result="error",
                    duration=duration,
                    confidence=0.0
                )
                raise
        
        return wrapper
    return decorator