"""
Prometheus Metrics Exporter

HTTP server that exposes metrics in Prometheus format.
"""

import asyncio
import logging
from typing import Optional
from aiohttp import web, web_response
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST

from ..metrics.registry import metrics_registry

logger = logging.getLogger(__name__)


class PrometheusExporter:
    """
    HTTP server that exposes Prometheus metrics.
    
    Provides /metrics endpoint for Prometheus scraping.
    """
    
    def __init__(
        self,
        port: int = 9090,
        host: str = "0.0.0.0",
        path: str = "/metrics",
        service_name: str = "brahminykite"
    ):
        self.port = port
        self.host = host
        self.path = path
        self.service_name = service_name
        
        self.app: Optional[web.Application] = None
        self.runner: Optional[web.AppRunner] = None
        self.site: Optional[web.TCPSite] = None
        
        # Set service name in metrics registry
        metrics_registry.set_common_labels(service=service_name)
    
    async def start(self) -> None:
        """Start the metrics HTTP server."""
        self.app = web.Application()
        
        # Add metrics endpoint
        self.app.router.add_get(self.path, self._metrics_handler)
        
        # Add health check endpoint
        self.app.router.add_get("/health", self._health_handler)
        
        # Start server
        self.runner = web.AppRunner(self.app)
        await self.runner.setup()
        
        self.site = web.TCPSite(self.runner, self.host, self.port)
        await self.site.start()
        
        logger.info(f"Prometheus exporter started on {self.host}:{self.port}{self.path}")
    
    async def stop(self) -> None:
        """Stop the metrics HTTP server."""
        if self.site:
            await self.site.stop()
        
        if self.runner:
            await self.runner.cleanup()
        
        logger.info("Prometheus exporter stopped")
    
    async def _metrics_handler(self, request: web.Request) -> web.Response:
        """Handle metrics requests."""
        try:
            # Update uptime before generating metrics
            metrics_registry.update_uptime(self.service_name)
            
            # Generate metrics
            metrics_data = generate_latest(metrics_registry.registry)
            
            return web.Response(
                body=metrics_data,
                content_type=CONTENT_TYPE_LATEST
            )
        except Exception as e:
            logger.error(f"Error generating metrics: {e}")
            return web.Response(
                text=f"Error generating metrics: {e}",
                status=500
            )
    
    async def _health_handler(self, request: web.Request) -> web.Response:
        """Handle health check requests."""
        return web.json_response({
            "status": "healthy",
            "service": self.service_name,
            "metrics_endpoint": self.path
        })


class MetricsMiddleware:
    """
    Middleware to automatically collect HTTP metrics.
    """
    
    def __init__(self):
        # HTTP request metrics
        self.http_requests = metrics_registry.counter(
            'http_requests_total',
            'Total HTTP requests',
            ['method', 'endpoint', 'status']
        )
        
        self.http_request_duration = metrics_registry.histogram(
            'http_request_duration_seconds',
            'HTTP request duration',
            ['method', 'endpoint']
        )
        
        self.http_request_size = metrics_registry.histogram(
            'http_request_size_bytes',
            'HTTP request size',
            ['method', 'endpoint']
        )
        
        self.http_response_size = metrics_registry.histogram(
            'http_response_size_bytes',
            'HTTP response size',
            ['method', 'endpoint']
        )
    
    async def __call__(self, request: web.Request, handler):
        """Process request and collect metrics."""
        method = request.method
        path = request.path
        
        # Start timing
        start_time = asyncio.get_event_loop().time()
        
        # Get request size
        request_size = request.content_length or 0
        self.http_request_size.labels(method=method, endpoint=path).observe(request_size)
        
        try:
            # Process request
            response = await handler(request)
            
            # Record success metrics
            status = str(response.status)
            self.http_requests.labels(method=method, endpoint=path, status=status).inc()
            
            # Record response size
            response_size = len(response.body) if hasattr(response, 'body') and response.body else 0
            self.http_response_size.labels(method=method, endpoint=path).observe(response_size)
            
            return response
            
        except Exception as e:
            # Record error metrics
            self.http_requests.labels(method=method, endpoint=path, status='error').inc()
            raise
        
        finally:
            # Record duration
            duration = asyncio.get_event_loop().time() - start_time
            self.http_request_duration.labels(method=method, endpoint=path).observe(duration)