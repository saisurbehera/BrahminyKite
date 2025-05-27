"""
Secure TLS/mTLS server implementations.
"""

import ssl
import asyncio
import logging
from typing import Optional, Dict, Any, Callable, List
from pathlib import Path
import socket

from aiohttp import web
import grpc
from grpc import aio

from .config import TLSConfig, ClientAuthMode
from .certificates import Certificate, CertificateManager
from .validation import CertificateValidator

logger = logging.getLogger(__name__)


class SecureServer:
    """Base class for secure servers."""
    
    def __init__(self, tls_config: TLSConfig):
        self.tls_config = tls_config
        self.ssl_context = None
        self._certificate: Optional[Certificate] = None
        self._validator = CertificateValidator()
    
    def create_ssl_context(self) -> ssl.SSLContext:
        """Create SSL context for the server."""
        # Create context with appropriate protocol
        if self.tls_config.min_version == self.tls_config.max_version:
            context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        else:
            context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        
        # Set minimum and maximum TLS versions
        context.minimum_version = getattr(
            ssl.TLSVersion,
            self.tls_config.min_version.value.replace(".", "_")
        )
        context.maximum_version = getattr(
            ssl.TLSVersion,
            self.tls_config.max_version.value.replace(".", "_")
        )
        
        # Set cipher suites
        if self.tls_config.cipher_suites:
            context.set_ciphers(":".join(self.tls_config.cipher_suites))
        
        # Load server certificate and key
        if self.tls_config.certificate.cert_file and self.tls_config.certificate.key_file:
            context.load_cert_chain(
                certfile=self.tls_config.certificate.cert_file,
                keyfile=self.tls_config.certificate.key_file
            )
        else:
            raise ValueError("Server certificate and key must be provided")
        
        # Configure client authentication
        if self.tls_config.client_auth_mode == ClientAuthMode.REQUIRED:
            context.verify_mode = ssl.CERT_REQUIRED
        elif self.tls_config.client_auth_mode == ClientAuthMode.OPTIONAL:
            context.verify_mode = ssl.CERT_OPTIONAL
        else:
            context.verify_mode = ssl.CERT_NONE
        
        # Load CA certificates for client verification
        if self.tls_config.client_auth_mode != ClientAuthMode.NONE:
            if self.tls_config.certificate.ca_file:
                context.load_verify_locations(cafile=self.tls_config.certificate.ca_file)
            elif self.tls_config.certificate.ca_bundle:
                context.load_verify_locations(cafile=self.tls_config.certificate.ca_bundle)
        
        # Set verification depth
        context.verify_flags |= ssl.VERIFY_X509_STRICT
        context.check_hostname = self.tls_config.verify_hostname
        
        # Enable session tickets
        if self.tls_config.enable_session_tickets:
            context.options |= ssl.OP_NO_TICKET
        
        # Prefer server cipher order
        if self.tls_config.prefer_server_ciphers:
            context.options |= ssl.OP_CIPHER_SERVER_PREFERENCE
        
        # Set up SNI callback if enabled
        if self.tls_config.enable_sni:
            context.sni_callback = self._sni_callback
        
        self.ssl_context = context
        return context
    
    def _sni_callback(self, ssl_socket, server_name, ssl_context):
        """SNI (Server Name Indication) callback."""
        logger.debug(f"SNI callback for: {server_name}")
        # In a production environment, this would select the appropriate certificate
        # based on the server name
    
    async def start(self, host: str = "0.0.0.0", port: int = 443):
        """Start the secure server."""
        raise NotImplementedError("Subclasses must implement start()")
    
    async def stop(self):
        """Stop the secure server."""
        raise NotImplementedError("Subclasses must implement stop()")


class TLSServer(SecureServer):
    """Standard TLS server (one-way authentication)."""
    
    def __init__(self, tls_config: TLSConfig, handler: Callable):
        # Ensure client auth is disabled for standard TLS
        tls_config.client_auth_mode = ClientAuthMode.NONE
        super().__init__(tls_config)
        self.handler = handler
        self._server = None
    
    async def handle_client(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
        """Handle client connection."""
        addr = writer.get_extra_info('peername')
        ssl_object = writer.get_extra_info('ssl_object')
        
        logger.info(f"TLS connection from {addr}")
        
        try:
            # Log TLS information
            if ssl_object:
                logger.debug(f"TLS version: {ssl_object.version()}")
                logger.debug(f"Cipher: {ssl_object.cipher()}")
            
            # Call the handler
            await self.handler(reader, writer)
        
        except Exception as e:
            logger.error(f"Error handling client {addr}: {e}")
        
        finally:
            writer.close()
            await writer.wait_closed()
    
    async def start(self, host: str = "0.0.0.0", port: int = 443):
        """Start the TLS server."""
        self.create_ssl_context()
        
        self._server = await asyncio.start_server(
            self.handle_client,
            host,
            port,
            ssl=self.ssl_context
        )
        
        logger.info(f"TLS server listening on {host}:{port}")
        
        async with self._server:
            await self._server.serve_forever()
    
    async def stop(self):
        """Stop the TLS server."""
        if self._server:
            self._server.close()
            await self._server.wait_closed()
            logger.info("TLS server stopped")


class mTLSServer(SecureServer):
    """mTLS server (mutual authentication)."""
    
    def __init__(self, tls_config: TLSConfig, handler: Callable):
        # Ensure client auth is required for mTLS
        tls_config.client_auth_mode = ClientAuthMode.REQUIRED
        super().__init__(tls_config)
        self.handler = handler
        self._server = None
        self._client_validators: List[Callable] = []
    
    def add_client_validator(self, validator: Callable[[Certificate], bool]):
        """Add a client certificate validator."""
        self._client_validators.append(validator)
    
    async def handle_client(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
        """Handle client connection with mTLS."""
        addr = writer.get_extra_info('peername')
        ssl_object = writer.get_extra_info('ssl_object')
        
        logger.info(f"mTLS connection from {addr}")
        
        try:
            # Get client certificate
            client_cert_der = ssl_object.getpeercert_bin()
            if not client_cert_der:
                logger.warning(f"No client certificate from {addr}")
                writer.close()
                await writer.wait_closed()
                return
            
            # Parse client certificate
            from cryptography import x509
            from cryptography.hazmat.backends import default_backend
            
            client_cert = x509.load_der_x509_certificate(
                client_cert_der,
                backend=default_backend()
            )
            
            cert_obj = Certificate(cert=client_cert)
            
            # Log client certificate info
            logger.info(f"Client certificate CN: {cert_obj.common_name}")
            logger.debug(f"Client certificate serial: {cert_obj.serial_number}")
            
            # Validate client certificate
            for validator in self._client_validators:
                if not validator(cert_obj):
                    logger.warning(f"Client certificate validation failed for {addr}")
                    writer.close()
                    await writer.wait_closed()
                    return
            
            # Pass client certificate to handler
            await self.handler(reader, writer, cert_obj)
        
        except Exception as e:
            logger.error(f"Error handling mTLS client {addr}: {e}")
        
        finally:
            writer.close()
            await writer.wait_closed()
    
    async def start(self, host: str = "0.0.0.0", port: int = 443):
        """Start the mTLS server."""
        self.create_ssl_context()
        
        self._server = await asyncio.start_server(
            self.handle_client,
            host,
            port,
            ssl=self.ssl_context
        )
        
        logger.info(f"mTLS server listening on {host}:{port}")
        
        async with self._server:
            await self._server.serve_forever()
    
    async def stop(self):
        """Stop the mTLS server."""
        if self._server:
            self._server.close()
            await self._server.wait_closed()
            logger.info("mTLS server stopped")


class SecureHTTPServer:
    """Secure HTTPS server using aiohttp."""
    
    def __init__(self, tls_config: TLSConfig, app: web.Application):
        self.tls_config = tls_config
        self.app = app
        self.ssl_context = None
        self._runner = None
        self._site = None
    
    def create_ssl_context(self) -> ssl.SSLContext:
        """Create SSL context for HTTPS server."""
        server = SecureServer(self.tls_config)
        return server.create_ssl_context()
    
    def add_security_middleware(self):
        """Add security middleware to the application."""
        from .middleware import SecurityHeadersMiddleware, TLSMiddleware
        
        # Add TLS enforcement middleware
        self.app.middlewares.append(TLSMiddleware(self.tls_config))
        
        # Add security headers middleware
        self.app.middlewares.append(SecurityHeadersMiddleware())
    
    async def start(self, host: str = "0.0.0.0", port: int = 443):
        """Start the HTTPS server."""
        self.ssl_context = self.create_ssl_context()
        self.add_security_middleware()
        
        self._runner = web.AppRunner(self.app)
        await self._runner.setup()
        
        self._site = web.TCPSite(
            self._runner,
            host,
            port,
            ssl_context=self.ssl_context
        )
        
        await self._site.start()
        logger.info(f"Secure HTTPS server listening on {host}:{port}")
    
    async def stop(self):
        """Stop the HTTPS server."""
        if self._site:
            await self._site.stop()
        
        if self._runner:
            await self._runner.cleanup()
        
        logger.info("Secure HTTPS server stopped")


class SecureGRPCServer:
    """Secure gRPC server with TLS/mTLS."""
    
    def __init__(self, tls_config: TLSConfig):
        self.tls_config = tls_config
        self._server: Optional[aio.Server] = None
        self._credentials = None
    
    def create_server_credentials(self) -> grpc.ServerCredentials:
        """Create gRPC server credentials."""
        # Load certificate and key
        with open(self.tls_config.certificate.cert_file, 'rb') as f:
            server_cert = f.read()
        
        with open(self.tls_config.certificate.key_file, 'rb') as f:
            server_key = f.read()
        
        # Load CA certificate for client verification
        root_cert = None
        if self.tls_config.client_auth_mode != ClientAuthMode.NONE:
            if self.tls_config.certificate.ca_file:
                with open(self.tls_config.certificate.ca_file, 'rb') as f:
                    root_cert = f.read()
        
        # Create credentials
        if self.tls_config.client_auth_mode == ClientAuthMode.REQUIRED:
            self._credentials = grpc.ssl_server_credentials(
                [(server_key, server_cert)],
                root_certificates=root_cert,
                require_client_auth=True
            )
        else:
            self._credentials = grpc.ssl_server_credentials(
                [(server_key, server_cert)],
                root_certificates=root_cert,
                require_client_auth=False
            )
        
        return self._credentials
    
    async def start(self, port: int = 50051, max_workers: int = 10):
        """Start the gRPC server."""
        self._server = aio.server()
        
        # Create credentials
        credentials = self.create_server_credentials()
        
        # Add secure port
        address = f"[::]:{port}"
        self._server.add_secure_port(address, credentials)
        
        await self._server.start()
        logger.info(f"Secure gRPC server listening on port {port}")
        
        await self._server.wait_for_termination()
    
    async def stop(self, grace_period: Optional[float] = None):
        """Stop the gRPC server."""
        if self._server:
            await self._server.stop(grace_period)
            logger.info("Secure gRPC server stopped")
    
    def add_service(self, service_adder, service):
        """Add a service to the gRPC server."""
        if not self._server:
            self._server = aio.server()
        
        service_adder(service, self._server)


# Factory functions

def create_secure_server(
    server_type: str,
    tls_config: TLSConfig,
    handler: Optional[Callable] = None,
    app: Optional[web.Application] = None
) -> SecureServer:
    """
    Create a secure server instance.
    
    Args:
        server_type: Type of server ("tls", "mtls", "https", "grpc")
        tls_config: TLS configuration
        handler: Handler function for TLS/mTLS servers
        app: aiohttp application for HTTPS servers
    
    Returns:
        Secure server instance
    """
    if server_type == "tls":
        if not handler:
            raise ValueError("Handler required for TLS server")
        return TLSServer(tls_config, handler)
    
    elif server_type == "mtls":
        if not handler:
            raise ValueError("Handler required for mTLS server")
        return mTLSServer(tls_config, handler)
    
    elif server_type == "https":
        if not app:
            raise ValueError("Application required for HTTPS server")
        return SecureHTTPServer(tls_config, app)
    
    elif server_type == "grpc":
        return SecureGRPCServer(tls_config)
    
    else:
        raise ValueError(f"Unknown server type: {server_type}")