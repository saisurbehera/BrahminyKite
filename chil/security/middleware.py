"""Security middleware and interceptors for various frameworks."""

import asyncio
import logging
import time
from typing import Any, Callable, Dict, List, Optional, Set, Union
from dataclasses import dataclass
from enum import Enum
from functools import wraps

from aiohttp import web
from grpc import aio
import grpc

from .validation import CertificateValidator
from .config import SecurityConfig, ClientAuthMode

logger = logging.getLogger(__name__)


class SecurityLevel(Enum):
    """Security levels for middleware."""
    NONE = "none"
    TLS = "tls"
    MTLS = "mtls"
    MTLS_STRICT = "mtls_strict"


@dataclass
class SecurityContext:
    """Security context for requests."""
    authenticated: bool
    client_cert: Optional[Any] = None
    client_dn: Optional[str] = None
    tls_version: Optional[str] = None
    cipher_suite: Optional[str] = None
    security_level: SecurityLevel = SecurityLevel.NONE
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class SecurityMiddleware:
    """Base security middleware."""
    
    def __init__(
        self,
        config: SecurityConfig,
        validator: Optional[CertificateValidator] = None,
        allowed_dns: Optional[Set[str]] = None,
        required_level: SecurityLevel = SecurityLevel.TLS
    ):
        self.config = config
        self.validator = validator or CertificateValidator()
        self.allowed_dns = allowed_dns or set()
        self.required_level = required_level
        self._metrics = {
            "total_requests": 0,
            "authenticated_requests": 0,
            "rejected_requests": 0,
            "auth_failures": 0
        }
        
    def check_security_level(self, ctx: SecurityContext) -> bool:
        """Check if security level meets requirements."""
        if self.required_level == SecurityLevel.NONE:
            return True
            
        if self.required_level == SecurityLevel.TLS:
            return ctx.security_level in [SecurityLevel.TLS, SecurityLevel.MTLS, SecurityLevel.MTLS_STRICT]
            
        if self.required_level == SecurityLevel.MTLS:
            return ctx.security_level in [SecurityLevel.MTLS, SecurityLevel.MTLS_STRICT]
            
        if self.required_level == SecurityLevel.MTLS_STRICT:
            return ctx.security_level == SecurityLevel.MTLS_STRICT
            
        return False
        
    def check_client_dn(self, dn: str) -> bool:
        """Check if client DN is allowed."""
        if not self.allowed_dns:
            return True
            
        return dn in self.allowed_dns
        
    def get_metrics(self) -> Dict[str, int]:
        """Get middleware metrics."""
        return self._metrics.copy()


class AIOHTTPSecurityMiddleware(SecurityMiddleware):
    """Security middleware for aiohttp."""
    
    @web.middleware
    async def middleware(self, request: web.Request, handler: Callable) -> web.Response:
        """AIOHTTP middleware handler."""
        self._metrics["total_requests"] += 1
        
        # Create security context
        ctx = SecurityContext(authenticated=False)
        
        # Check if connection is secure
        if request.secure:
            ctx.security_level = SecurityLevel.TLS
            
            # Get SSL info
            ssl_info = request.transport.get_extra_info('peercert')
            if ssl_info:
                ctx.client_cert = ssl_info
                ctx.security_level = SecurityLevel.MTLS
                
                # Extract client DN
                if 'subject' in ssl_info:
                    subject = dict(x[0] for x in ssl_info['subject'])
                    ctx.client_dn = f"CN={subject.get('commonName', 'unknown')}"
                    
                # Validate certificate if strict mode
                if self.required_level == SecurityLevel.MTLS_STRICT:
                    try:
                        # Note: In real implementation, convert ssl_info to proper cert object
                        result = self.validator.validate_certificate(ssl_info)
                        if not result.is_valid:
                            self._metrics["auth_failures"] += 1
                            return web.Response(status=401, text="Invalid client certificate")
                    except Exception as e:
                        logger.error(f"Certificate validation error: {e}")
                        self._metrics["auth_failures"] += 1
                        return web.Response(status=401, text="Certificate validation failed")
                        
                # Check DN if required
                if ctx.client_dn and not self.check_client_dn(ctx.client_dn):
                    self._metrics["rejected_requests"] += 1
                    return web.Response(status=403, text="Client not authorized")
                    
                ctx.authenticated = True
                self._metrics["authenticated_requests"] += 1
                
            # Get TLS info
            ssl_object = request.transport.get_extra_info('ssl_object')
            if ssl_object:
                ctx.tls_version = ssl_object.version()
                ctx.cipher_suite = ssl_object.cipher()[0] if ssl_object.cipher() else None
                
        # Check security level
        if not self.check_security_level(ctx):
            self._metrics["rejected_requests"] += 1
            return web.Response(status=403, text=f"Security level {self.required_level.value} required")
            
        # Add context to request
        request['security_context'] = ctx
        
        # Call handler
        return await handler(request)
        
    def setup(self, app: web.Application):
        """Setup middleware on application."""
        app.middlewares.append(self.middleware)


class GRPCSecurityInterceptor(SecurityMiddleware):
    """Security interceptor for gRPC."""
    
    async def intercept_service(
        self,
        continuation: Callable,
        handler_call_details: grpc.HandlerCallDetails
    ) -> Any:
        """Intercept gRPC calls."""
        self._metrics["total_requests"] += 1
        
        # Create security context
        ctx = SecurityContext(authenticated=False)
        
        # Get peer certificate
        context = handler_call_details.invocation_metadata
        if context:
            # Extract TLS info from context
            for key, value in context:
                if key == 'peer_certificate':
                    ctx.client_cert = value
                    ctx.security_level = SecurityLevel.MTLS
                elif key == 'tls_version':
                    ctx.tls_version = value
                elif key == 'cipher_suite':
                    ctx.cipher_suite = value
                    
        # Check security level
        if not self.check_security_level(ctx):
            self._metrics["rejected_requests"] += 1
            context = aio.ServicerContext()
            context.abort(grpc.StatusCode.PERMISSION_DENIED, f"Security level {self.required_level.value} required")
            
        # Validate certificate if required
        if ctx.client_cert and self.required_level == SecurityLevel.MTLS_STRICT:
            try:
                result = self.validator.validate_certificate(ctx.client_cert)
                if not result.is_valid:
                    self._metrics["auth_failures"] += 1
                    context = aio.ServicerContext()
                    context.abort(grpc.StatusCode.UNAUTHENTICATED, "Invalid client certificate")
            except Exception as e:
                logger.error(f"Certificate validation error: {e}")
                self._metrics["auth_failures"] += 1
                context = aio.ServicerContext()
                context.abort(grpc.StatusCode.UNAUTHENTICATED, "Certificate validation failed")
                
        if ctx.security_level in [SecurityLevel.MTLS, SecurityLevel.MTLS_STRICT]:
            ctx.authenticated = True
            self._metrics["authenticated_requests"] += 1
            
        # Continue with handler
        return await continuation(handler_call_details)


class RateLimitingMiddleware:
    """Rate limiting middleware."""
    
    def __init__(
        self,
        max_requests: int = 100,
        window_seconds: int = 60,
        key_func: Optional[Callable] = None
    ):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.key_func = key_func or self._default_key
        self._buckets: Dict[str, List[float]] = {}
        self._lock = asyncio.Lock()
        
    def _default_key(self, request: Any) -> str:
        """Default key extraction."""
        if hasattr(request, 'remote'):
            return request.remote
        elif hasattr(request, 'peer'):
            return request.peer
        return "unknown"
        
    async def check_rate_limit(self, key: str) -> bool:
        """Check if request is within rate limit."""
        async with self._lock:
            now = time.time()
            
            # Initialize bucket if needed
            if key not in self._buckets:
                self._buckets[key] = []
                
            # Clean old entries
            self._buckets[key] = [
                t for t in self._buckets[key]
                if now - t < self.window_seconds
            ]
            
            # Check limit
            if len(self._buckets[key]) >= self.max_requests:
                return False
                
            # Add request
            self._buckets[key].append(now)
            return True
            
    @web.middleware
    async def aiohttp_middleware(self, request: web.Request, handler: Callable) -> web.Response:
        """AIOHTTP rate limiting middleware."""
        key = self.key_func(request)
        
        if not await self.check_rate_limit(key):
            return web.Response(
                status=429,
                text="Rate limit exceeded",
                headers={"Retry-After": str(self.window_seconds)}
            )
            
        return await handler(request)


class IPWhitelistMiddleware:
    """IP whitelisting middleware."""
    
    def __init__(
        self,
        allowed_ips: Set[str],
        allowed_ranges: Optional[List[str]] = None
    ):
        self.allowed_ips = allowed_ips
        self.allowed_ranges = allowed_ranges or []
        
    def is_ip_allowed(self, ip: str) -> bool:
        """Check if IP is allowed."""
        # Direct match
        if ip in self.allowed_ips:
            return True
            
        # Range match (simplified, real implementation would use ipaddress module)
        for range_spec in self.allowed_ranges:
            if ip.startswith(range_spec.rstrip('*')):
                return True
                
        return False
        
    @web.middleware
    async def aiohttp_middleware(self, request: web.Request, handler: Callable) -> web.Response:
        """AIOHTTP IP whitelist middleware."""
        client_ip = request.remote
        
        if not self.is_ip_allowed(client_ip):
            logger.warning(f"Rejected request from unauthorized IP: {client_ip}")
            return web.Response(status=403, text="Access denied")
            
        return await handler(request)


class SecurityHeadersMiddleware:
    """Security headers middleware."""
    
    def __init__(self, headers: Optional[Dict[str, str]] = None):
        self.headers = headers or self._default_headers()
        
    def _default_headers(self) -> Dict[str, str]:
        """Get default security headers."""
        return {
            "Strict-Transport-Security": "max-age=31536000; includeSubDomains",
            "X-Content-Type-Options": "nosniff",
            "X-Frame-Options": "DENY",
            "X-XSS-Protection": "1; mode=block",
            "Content-Security-Policy": "default-src 'self'",
            "Referrer-Policy": "strict-origin-when-cross-origin"
        }
        
    @web.middleware
    async def aiohttp_middleware(self, request: web.Request, handler: Callable) -> web.Response:
        """AIOHTTP security headers middleware."""
        response = await handler(request)
        
        # Add security headers
        for header, value in self.headers.items():
            response.headers[header] = value
            
        return response


class CORSMiddleware:
    """CORS middleware with security considerations."""
    
    def __init__(
        self,
        allowed_origins: Set[str],
        allowed_methods: Set[str] = None,
        allowed_headers: Set[str] = None,
        max_age: int = 86400
    ):
        self.allowed_origins = allowed_origins
        self.allowed_methods = allowed_methods or {"GET", "POST", "PUT", "DELETE", "OPTIONS"}
        self.allowed_headers = allowed_headers or {"Content-Type", "Authorization"}
        self.max_age = max_age
        
    def is_origin_allowed(self, origin: str) -> bool:
        """Check if origin is allowed."""
        return origin in self.allowed_origins or "*" in self.allowed_origins
        
    @web.middleware
    async def aiohttp_middleware(self, request: web.Request, handler: Callable) -> web.Response:
        """AIOHTTP CORS middleware."""
        # Handle preflight
        if request.method == "OPTIONS":
            return self._create_preflight_response(request)
            
        # Handle regular request
        response = await handler(request)
        
        # Add CORS headers
        origin = request.headers.get("Origin")
        if origin and self.is_origin_allowed(origin):
            response.headers["Access-Control-Allow-Origin"] = origin
            response.headers["Access-Control-Allow-Credentials"] = "true"
            
        return response
        
    def _create_preflight_response(self, request: web.Request) -> web.Response:
        """Create preflight response."""
        origin = request.headers.get("Origin")
        
        if not origin or not self.is_origin_allowed(origin):
            return web.Response(status=403)
            
        response = web.Response(status=200)
        response.headers["Access-Control-Allow-Origin"] = origin
        response.headers["Access-Control-Allow-Methods"] = ", ".join(self.allowed_methods)
        response.headers["Access-Control-Allow-Headers"] = ", ".join(self.allowed_headers)
        response.headers["Access-Control-Max-Age"] = str(self.max_age)
        response.headers["Access-Control-Allow-Credentials"] = "true"
        
        return response


def require_auth(level: SecurityLevel = SecurityLevel.TLS):
    """Decorator to require authentication."""
    def decorator(func):
        @wraps(func)
        async def wrapper(request, *args, **kwargs):
            ctx = request.get('security_context')
            if not ctx or ctx.security_level.value < level.value:
                return web.Response(status=401, text="Authentication required")
            return await func(request, *args, **kwargs)
        return wrapper
    return decorator


def require_client_cert(allowed_dns: Optional[Set[str]] = None):
    """Decorator to require client certificate."""
    def decorator(func):
        @wraps(func)
        async def wrapper(request, *args, **kwargs):
            ctx = request.get('security_context')
            if not ctx or not ctx.client_cert:
                return web.Response(status=401, text="Client certificate required")
                
            if allowed_dns and ctx.client_dn not in allowed_dns:
                return web.Response(status=403, text="Client not authorized")
                
            return await func(request, *args, **kwargs)
        return wrapper
    return decorator