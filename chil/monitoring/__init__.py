"""
BrahminyKite Monitoring and Metrics

Comprehensive monitoring solution with Prometheus metrics, Grafana dashboards,
and automated alerting for the BrahminyKite AI verification system.
"""

from .metrics.registry import (
    MetricsRegistry,
    metrics_registry,
    get_counter,
    get_histogram,
    get_gauge,
    get_summary,
    get_info,
    DEFAULT_LATENCY_BUCKETS,
    DEFAULT_SIZE_BUCKETS
)

from .metrics.service_metrics import (
    ServiceMetrics,
    grpc_metrics_interceptor,
    metrics_decorator
)

from .exporters.prometheus_exporter import (
    PrometheusExporter,
    MetricsMiddleware
)

from .exporters.fastapi_middleware import (
    FastAPIMetricsMiddleware,
    add_metrics_endpoint,
    add_health_endpoint,
    CustomMetrics,
    custom_metrics
)

__all__ = [
    # Registry
    'MetricsRegistry',
    'metrics_registry',
    'get_counter',
    'get_histogram', 
    'get_gauge',
    'get_summary',
    'get_info',
    'DEFAULT_LATENCY_BUCKETS',
    'DEFAULT_SIZE_BUCKETS',
    
    # Service metrics
    'ServiceMetrics',
    'grpc_metrics_interceptor',
    'metrics_decorator',
    
    # Exporters
    'PrometheusExporter',
    'MetricsMiddleware',
    
    # FastAPI integration
    'FastAPIMetricsMiddleware',
    'add_metrics_endpoint',
    'add_health_endpoint',
    'CustomMetrics',
    'custom_metrics',
]