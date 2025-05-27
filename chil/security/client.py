"""
Secure TLS/mTLS client implementations.
"""

import ssl
import asyncio
import logging
from typing import Optional, Dict, Any, Tuple
import aiohttp
import grpc
from grpc import aio

from .config import TLSConfig, ClientAuthMode
from .certificates import Certificate, CertificateManager
from .validation import validate_hostname

logger = logging.getLogger(__name__)


class SecureClient:
    """Base class for secure clients."""
    
    def __init__(self, tls_config: TLSConfig):
        self.tls_config = tls_config
        self.ssl_context = None
        self._certificate: Optional[Certificate] = None
    
    def create_ssl_context(self) -> ssl.SSLContext:
        """Create SSL context for the client."""
        # Create context
        context = ssl.create_default_context(ssl.Purpose.SERVER_AUTH)
        
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
        
        # Load CA certificates
        if self.tls_config.certificate.ca_file:
            context.load_verify_locations(cafile=self.tls_config.certificate.ca_file)
        elif self.tls_config.certificate.ca_bundle:
            context.load_verify_locations(cafile=self.tls_config.certificate.ca_bundle)
        
        # Load client certificate for mTLS
        if (self.tls_config.certificate.cert_file and 
            self.tls_config.certificate.key_file):
            context.load_cert_chain(
                certfile=self.tls_config.certificate.cert_file,
                keyfile=self.tls_config.certificate.key_file
            )
        
        # Set verification mode
        if self.tls_config.verify_hostname:
            context.check_hostname = True
            context.verify_mode = ssl.CERT_REQUIRED
        else:
            context.check_hostname = False
            context.verify_mode = ssl.CERT_NONE
        
        self.ssl_context = context
        return context
    
    async def connect(self, host: str, port: int):
        """Connect to a secure server."""
        raise NotImplementedError("Subclasses must implement connect()")
    
    async def close(self):
        """Close the client connection."""
        raise NotImplementedError("Subclasses must implement close()")


class TLSClient(SecureClient):
    """Standard TLS client."""
    
    def __init__(self, tls_config: TLSConfig):
        super().__init__(tls_config)
        self._reader: Optional[asyncio.StreamReader] = None
        self._writer: Optional[asyncio.StreamWriter] = None
    
    async def connect(self, host: str, port: int) -> Tuple[asyncio.StreamReader, asyncio.StreamWriter]:
        """Connect to a TLS server."""
        self.create_ssl_context()
        
        self._reader, self._writer = await asyncio.open_connection(
            host,
            port,
            ssl=self.ssl_context,
            server_hostname=host if self.tls_config.verify_hostname else None
        )
        
        # Get SSL object for inspection
        ssl_object = self._writer.get_extra_info('ssl_object')
        if ssl_object:
            logger.info(f"Connected to {host}:{port}")
            logger.debug(f"TLS version: {ssl_object.version()}")
            logger.debug(f"Cipher: {ssl_object.cipher()}")
            
            # Get server certificate
            server_cert_der = ssl_object.getpeercert_bin()
            if server_cert_der:
                from cryptography import x509
                from cryptography.hazmat.backends import default_backend
                
                server_cert = x509.load_der_x509_certificate(
                    server_cert_der,
                    backend=default_backend()
                )
                
                cert_obj = Certificate(cert=server_cert)
                logger.info(f"Server certificate CN: {cert_obj.common_name}")
                
                # Validate hostname
                if self.tls_config.verify_hostname:
                    errors = validate_hostname(cert_obj, host)
                    if errors:
                        logger.warning(f"Hostname validation errors: {errors}")
        
        return self._reader, self._writer
    
    async def send(self, data: bytes):
        """Send data to the server."""
        if not self._writer:
            raise RuntimeError("Not connected")
        
        self._writer.write(data)
        await self._writer.drain()
    
    async def receive(self, size: int = 4096) -> bytes:
        """Receive data from the server."""
        if not self._reader:
            raise RuntimeError("Not connected")
        
        return await self._reader.read(size)
    
    async def close(self):
        """Close the connection."""
        if self._writer:
            self._writer.close()
            await self._writer.wait_closed()
            self._writer = None
            self._reader = None
            logger.info("TLS connection closed")


class mTLSClient(SecureClient):
    """mTLS client with client certificate authentication."""
    
    def __init__(self, tls_config: TLSConfig):
        # Ensure client certificate is provided
        if not (tls_config.certificate.cert_file and tls_config.certificate.key_file):
            raise ValueError("Client certificate and key required for mTLS")
        
        super().__init__(tls_config)
        self._reader: Optional[asyncio.StreamReader] = None
        self._writer: Optional[asyncio.StreamWriter] = None
    
    async def connect(self, host: str, port: int) -> Tuple[asyncio.StreamReader, asyncio.StreamWriter]:
        """Connect to an mTLS server."""
        self.create_ssl_context()
        
        self._reader, self._writer = await asyncio.open_connection(
            host,
            port,
            ssl=self.ssl_context,
            server_hostname=host if self.tls_config.verify_hostname else None
        )
        
        # Get SSL object for inspection
        ssl_object = self._writer.get_extra_info('ssl_object')
        if ssl_object:
            logger.info(f"mTLS connection established to {host}:{port}")
            logger.debug(f"TLS version: {ssl_object.version()}")
            logger.debug(f"Cipher: {ssl_object.cipher()}")
            
            # Log client certificate info
            if self._certificate:
                logger.info(f"Using client certificate CN: {self._certificate.common_name}")
        
        return self._reader, self._writer
    
    async def send(self, data: bytes):
        """Send data to the server."""
        if not self._writer:
            raise RuntimeError("Not connected")
        
        self._writer.write(data)
        await self._writer.drain()
    
    async def receive(self, size: int = 4096) -> bytes:
        """Receive data from the server."""
        if not self._reader:
            raise RuntimeError("Not connected")
        
        return await self._reader.read(size)
    
    async def close(self):
        """Close the connection."""
        if self._writer:
            self._writer.close()
            await self._writer.wait_closed()
            self._writer = None
            self._reader = None
            logger.info("mTLS connection closed")


class SecureHTTPClient:
    """Secure HTTPS client using aiohttp."""
    
    def __init__(self, tls_config: TLSConfig):
        self.tls_config = tls_config
        self.ssl_context = None
        self._session: Optional[aiohttp.ClientSession] = None
        self._connector = None
    
    def create_ssl_context(self) -> ssl.SSLContext:
        """Create SSL context for HTTPS client."""
        client = SecureClient(self.tls_config)
        return client.create_ssl_context()
    
    async def __aenter__(self):
        """Async context manager entry."""
        await self.start()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()
    
    async def start(self):
        """Start the HTTP client session."""
        self.ssl_context = self.create_ssl_context()
        
        # Create connector with SSL context
        self._connector = aiohttp.TCPConnector(
            ssl=self.ssl_context,
            limit=100,
            limit_per_host=30
        )
        
        # Create session
        self._session = aiohttp.ClientSession(
            connector=self._connector,
            timeout=aiohttp.ClientTimeout(total=30)
        )
        
        logger.info("Secure HTTP client session created")
    
    async def close(self):
        """Close the HTTP client session."""
        if self._session:
            await self._session.close()
            self._session = None
        
        if self._connector:
            await self._connector.close()
            self._connector = None
        
        logger.info("Secure HTTP client session closed")
    
    async def get(self, url: str, **kwargs) -> aiohttp.ClientResponse:
        """Perform GET request."""
        if not self._session:
            raise RuntimeError("Client not started")
        
        return await self._session.get(url, **kwargs)
    
    async def post(self, url: str, **kwargs) -> aiohttp.ClientResponse:
        """Perform POST request."""
        if not self._session:
            raise RuntimeError("Client not started")
        
        return await self._session.post(url, **kwargs)
    
    async def put(self, url: str, **kwargs) -> aiohttp.ClientResponse:
        """Perform PUT request."""
        if not self._session:
            raise RuntimeError("Client not started")
        
        return await self._session.put(url, **kwargs)
    
    async def delete(self, url: str, **kwargs) -> aiohttp.ClientResponse:
        """Perform DELETE request."""
        if not self._session:
            raise RuntimeError("Client not started")
        
        return await self._session.delete(url, **kwargs)
    
    async def request(self, method: str, url: str, **kwargs) -> aiohttp.ClientResponse:
        """Perform generic HTTP request."""
        if not self._session:
            raise RuntimeError("Client not started")
        
        return await self._session.request(method, url, **kwargs)


class SecureGRPCClient:
    """Secure gRPC client with TLS/mTLS."""
    
    def __init__(self, tls_config: TLSConfig):
        self.tls_config = tls_config
        self._channel: Optional[aio.Channel] = None
        self._credentials = None
    
    def create_channel_credentials(self) -> grpc.ChannelCredentials:
        """Create gRPC channel credentials."""
        # Load CA certificate
        root_cert = None
        if self.tls_config.certificate.ca_file:
            with open(self.tls_config.certificate.ca_file, 'rb') as f:
                root_cert = f.read()
        
        # Load client certificate for mTLS
        client_cert = None
        client_key = None
        if (self.tls_config.certificate.cert_file and 
            self.tls_config.certificate.key_file):
            with open(self.tls_config.certificate.cert_file, 'rb') as f:
                client_cert = f.read()
            
            with open(self.tls_config.certificate.key_file, 'rb') as f:
                client_key = f.read()
        
        # Create credentials
        if client_cert and client_key:
            # mTLS
            self._credentials = grpc.ssl_channel_credentials(
                root_certificates=root_cert,
                private_key=client_key,
                certificate_chain=client_cert
            )
        else:
            # Standard TLS
            self._credentials = grpc.ssl_channel_credentials(
                root_certificates=root_cert
            )
        
        return self._credentials
    
    async def connect(self, target: str) -> aio.Channel:
        """Connect to a gRPC server."""
        credentials = self.create_channel_credentials()
        
        # Create channel options
        options = [
            ('grpc.ssl_target_name_override', target.split(':')[0])
        ] if not self.tls_config.verify_hostname else []
        
        # Create secure channel
        self._channel = aio.secure_channel(
            target,
            credentials,
            options=options
        )
        
        logger.info(f"Secure gRPC channel created to {target}")
        
        return self._channel
    
    async def close(self):
        """Close the gRPC channel."""
        if self._channel:
            await self._channel.close()
            self._channel = None
            logger.info("Secure gRPC channel closed")
    
    def get_channel(self) -> aio.Channel:
        """Get the gRPC channel."""
        if not self._channel:
            raise RuntimeError("Not connected")
        
        return self._channel


# Factory functions

def create_secure_client(
    client_type: str,
    tls_config: TLSConfig
) -> SecureClient:
    """
    Create a secure client instance.
    
    Args:
        client_type: Type of client ("tls", "mtls", "https", "grpc")
        tls_config: TLS configuration
    
    Returns:
        Secure client instance
    """
    if client_type == "tls":
        return TLSClient(tls_config)
    
    elif client_type == "mtls":
        return mTLSClient(tls_config)
    
    elif client_type == "https":
        return SecureHTTPClient(tls_config)
    
    elif client_type == "grpc":
        return SecureGRPCClient(tls_config)
    
    else:
        raise ValueError(f"Unknown client type: {client_type}")