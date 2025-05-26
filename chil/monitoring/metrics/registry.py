"""
Metrics Registry for BrahminyKite

Centralized registry for all Prometheus metrics across the system.
"""

import time
import threading
from typing import Dict, List, Optional, Union, Any
from prometheus_client import (
    Counter, Histogram, Gauge, Summary, Info,
    CollectorRegistry, generate_latest, CONTENT_TYPE_LATEST
)
from prometheus_client.core import REGISTRY
import logging

logger = logging.getLogger(__name__)


class MetricsRegistry:
    """
    Centralized metrics registry for BrahminyKite.
    
    Provides a unified interface for creating and managing Prometheus metrics
    across all services and components.
    """
    
    def __init__(self, namespace: str = "brahminykite", registry: Optional[CollectorRegistry] = None):
        self.namespace = namespace
        self.registry = registry or REGISTRY
        self._metrics: Dict[str, Any] = {}
        self._lock = threading.Lock()
        
        # Common labels for all metrics
        self.common_labels = {
            'service': 'unknown',
            'version': '1.0.0',
            'environment': 'development'
        }
        
        # Initialize system metrics
        self._init_system_metrics()
    
    def _init_system_metrics(self) -> None:
        """Initialize common system metrics."""
        self.system_info = Info(
            f'{self.namespace}_system_info',
            'System information',
            registry=self.registry
        )
        
        self.uptime_gauge = Gauge(
            f'{self.namespace}_uptime_seconds',
            'Service uptime in seconds',
            ['service'],
            registry=self.registry
        )
        
        self.start_time = time.time()
    
    def set_common_labels(self, **labels) -> None:
        """Set common labels for all metrics."""
        with self._lock:
            self.common_labels.update(labels)
    
    def counter(
        self,
        name: str,
        description: str,
        labelnames: Optional[List[str]] = None,
        **kwargs
    ) -> Counter:
        """Create or get a Counter metric."""
        full_name = f"{self.namespace}_{name}"
        
        with self._lock:
            if full_name not in self._metrics:
                self._metrics[full_name] = Counter(
                    full_name,
                    description,
                    labelnames or [],
                    registry=self.registry,
                    **kwargs
                )
            return self._metrics[full_name]
    
    def histogram(
        self,
        name: str,
        description: str,
        labelnames: Optional[List[str]] = None,
        buckets: Optional[List[float]] = None,
        **kwargs
    ) -> Histogram:
        """Create or get a Histogram metric."""
        full_name = f"{self.namespace}_{name}"
        
        with self._lock:
            if full_name not in self._metrics:
                self._metrics[full_name] = Histogram(
                    full_name,
                    description,
                    labelnames or [],
                    buckets=buckets,
                    registry=self.registry,
                    **kwargs
                )
            return self._metrics[full_name]
    
    def gauge(
        self,
        name: str,
        description: str,
        labelnames: Optional[List[str]] = None,
        **kwargs
    ) -> Gauge:
        """Create or get a Gauge metric."""
        full_name = f"{self.namespace}_{name}"
        
        with self._lock:
            if full_name not in self._metrics:
                self._metrics[full_name] = Gauge(
                    full_name,
                    description,
                    labelnames or [],
                    registry=self.registry,
                    **kwargs
                )
            return self._metrics[full_name]
    
    def summary(
        self,
        name: str,
        description: str,
        labelnames: Optional[List[str]] = None,
        **kwargs
    ) -> Summary:
        """Create or get a Summary metric."""
        full_name = f"{self.namespace}_{name}"
        
        with self._lock:
            if full_name not in self._metrics:
                self._metrics[full_name] = Summary(
                    full_name,
                    description,
                    labelnames or [],
                    registry=self.registry,
                    **kwargs
                )
            return self._metrics[full_name]
    
    def info(
        self,
        name: str,
        description: str,
        **kwargs
    ) -> Info:
        """Create or get an Info metric."""
        full_name = f"{self.namespace}_{name}"
        
        with self._lock:
            if full_name not in self._metrics:
                self._metrics[full_name] = Info(
                    full_name,
                    description,
                    registry=self.registry,
                    **kwargs
                )
            return self._metrics[full_name]
    
    def update_uptime(self, service_name: str) -> None:
        """Update the uptime metric for a service."""
        uptime = time.time() - self.start_time
        self.uptime_gauge.labels(service=service_name).set(uptime)
    
    def generate_metrics(self) -> str:
        """Generate metrics in Prometheus format."""
        return generate_latest(self.registry).decode('utf-8')
    
    def get_content_type(self) -> str:
        """Get the content type for metrics."""
        return CONTENT_TYPE_LATEST
    
    def clear_metrics(self) -> None:
        """Clear all registered metrics."""
        with self._lock:
            self._metrics.clear()
            # Re-initialize system metrics
            self._init_system_metrics()


# Global metrics registry instance
metrics_registry = MetricsRegistry()


# Convenience functions for common metrics
def get_counter(name: str, description: str, labelnames: Optional[List[str]] = None) -> Counter:
    """Get a counter metric from the global registry."""
    return metrics_registry.counter(name, description, labelnames)


def get_histogram(name: str, description: str, labelnames: Optional[List[str]] = None, 
                 buckets: Optional[List[float]] = None) -> Histogram:
    """Get a histogram metric from the global registry."""
    return metrics_registry.histogram(name, description, labelnames, buckets)


def get_gauge(name: str, description: str, labelnames: Optional[List[str]] = None) -> Gauge:
    """Get a gauge metric from the global registry."""
    return metrics_registry.gauge(name, description, labelnames)


def get_summary(name: str, description: str, labelnames: Optional[List[str]] = None) -> Summary:
    """Get a summary metric from the global registry."""
    return metrics_registry.summary(name, description, labelnames)


def get_info(name: str, description: str) -> Info:
    """Get an info metric from the global registry."""
    return metrics_registry.info(name, description)


# Common metric buckets
DEFAULT_LATENCY_BUCKETS = [0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0]
DEFAULT_SIZE_BUCKETS = [1, 10, 100, 1000, 10000, 100000, 1000000, 10000000]