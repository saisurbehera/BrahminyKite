"""
TLS/mTLS configuration management.
"""

import os
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from enum import Enum
from pathlib import Path


class TLSVersion(str, Enum):
    """Supported TLS versions."""
    TLS_1_2 = "TLSv1.2"
    TLS_1_3 = "TLSv1.3"


class CipherSuite(str, Enum):
    """Recommended cipher suites."""
    # TLS 1.3 cipher suites
    TLS_AES_128_GCM_SHA256 = "TLS_AES_128_GCM_SHA256"
    TLS_AES_256_GCM_SHA384 = "TLS_AES_256_GCM_SHA384"
    TLS_CHACHA20_POLY1305_SHA256 = "TLS_CHACHA20_POLY1305_SHA256"
    
    # TLS 1.2 cipher suites (ECDHE for forward secrecy)
    ECDHE_RSA_AES128_GCM_SHA256 = "ECDHE-RSA-AES128-GCM-SHA256"
    ECDHE_RSA_AES256_GCM_SHA384 = "ECDHE-RSA-AES256-GCM-SHA384"
    ECDHE_ECDSA_AES128_GCM_SHA256 = "ECDHE-ECDSA-AES128-GCM-SHA256"
    ECDHE_ECDSA_AES256_GCM_SHA384 = "ECDHE-ECDSA-AES256-GCM-SHA384"


class ClientAuthMode(str, Enum):
    """Client authentication modes."""
    NONE = "none"           # No client auth (standard TLS)
    OPTIONAL = "optional"   # Client cert optional
    REQUIRED = "required"   # Client cert required (mTLS)


@dataclass
class CertificateConfig:
    """Certificate configuration."""
    
    # Certificate paths
    cert_file: Optional[str] = None
    key_file: Optional[str] = None
    ca_file: Optional[str] = None
    ca_bundle: Optional[str] = None
    
    # Certificate generation
    common_name: Optional[str] = None
    organization: Optional[str] = None
    organizational_unit: Optional[str] = None
    country: Optional[str] = None
    state: Optional[str] = None
    locality: Optional[str] = None
    
    # Certificate options
    key_size: int = 2048
    days_valid: int = 365
    is_ca: bool = False
    
    # Subject Alternative Names
    san_dns: List[str] = field(default_factory=list)
    san_ip: List[str] = field(default_factory=list)
    
    @classmethod
    def from_env(cls, prefix: str = "TLS") -> "CertificateConfig":
        """Create configuration from environment variables."""
        config = cls()
        
        # Certificate paths
        if cert_file := os.getenv(f"{prefix}_CERT_FILE"):
            config.cert_file = cert_file
        
        if key_file := os.getenv(f"{prefix}_KEY_FILE"):
            config.key_file = key_file
        
        if ca_file := os.getenv(f"{prefix}_CA_FILE"):
            config.ca_file = ca_file
        
        if ca_bundle := os.getenv(f"{prefix}_CA_BUNDLE"):
            config.ca_bundle = ca_bundle
        
        # Certificate details
        if common_name := os.getenv(f"{prefix}_COMMON_NAME"):
            config.common_name = common_name
        
        if organization := os.getenv(f"{prefix}_ORGANIZATION"):
            config.organization = organization
        
        # SAN entries
        if san_dns := os.getenv(f"{prefix}_SAN_DNS"):
            config.san_dns = san_dns.split(",")
        
        if san_ip := os.getenv(f"{prefix}_SAN_IP"):
            config.san_ip = san_ip.split(",")
        
        return config
    
    def validate(self) -> List[str]:
        """Validate certificate configuration."""
        errors = []
        
        # Check certificate files exist if specified
        if self.cert_file and not Path(self.cert_file).exists():
            errors.append(f"Certificate file not found: {self.cert_file}")
        
        if self.key_file and not Path(self.key_file).exists():
            errors.append(f"Key file not found: {self.key_file}")
        
        if self.ca_file and not Path(self.ca_file).exists():
            errors.append(f"CA file not found: {self.ca_file}")
        
        # Validate key size
        if self.key_size not in [2048, 3072, 4096]:
            errors.append(f"Invalid key size: {self.key_size}")
        
        return errors


@dataclass
class TLSConfig:
    """TLS configuration."""
    
    # TLS version
    min_version: TLSVersion = TLSVersion.TLS_1_2
    max_version: TLSVersion = TLSVersion.TLS_1_3
    
    # Cipher suites
    cipher_suites: List[str] = field(default_factory=lambda: [
        CipherSuite.TLS_AES_128_GCM_SHA256.value,
        CipherSuite.TLS_AES_256_GCM_SHA384.value,
        CipherSuite.ECDHE_RSA_AES128_GCM_SHA256.value,
        CipherSuite.ECDHE_RSA_AES256_GCM_SHA384.value,
    ])
    
    # Client authentication
    client_auth_mode: ClientAuthMode = ClientAuthMode.NONE
    
    # Certificate configuration
    certificate: CertificateConfig = field(default_factory=CertificateConfig)
    
    # Validation
    verify_hostname: bool = True
    verify_depth: int = 10
    
    # Session
    session_timeout: int = 86400  # 24 hours
    session_cache_size: int = 20480
    
    # OCSP (Online Certificate Status Protocol)
    enable_ocsp_stapling: bool = True
    ocsp_responder_url: Optional[str] = None
    
    # Additional options
    prefer_server_ciphers: bool = True
    enable_session_tickets: bool = True
    enable_sni: bool = True  # Server Name Indication
    
    @classmethod
    def from_env(cls, prefix: str = "TLS") -> "TLSConfig":
        """Create configuration from environment variables."""
        config = cls()
        
        # TLS versions
        if min_version := os.getenv(f"{prefix}_MIN_VERSION"):
            config.min_version = TLSVersion(min_version)
        
        if max_version := os.getenv(f"{prefix}_MAX_VERSION"):
            config.max_version = TLSVersion(max_version)
        
        # Client auth
        if client_auth := os.getenv(f"{prefix}_CLIENT_AUTH"):
            config.client_auth_mode = ClientAuthMode(client_auth)
        
        # Validation
        if verify_hostname := os.getenv(f"{prefix}_VERIFY_HOSTNAME"):
            config.verify_hostname = verify_hostname.lower() in ("true", "1", "yes")
        
        # Certificate config
        config.certificate = CertificateConfig.from_env(prefix)
        
        return config
    
    def to_ssl_context_kwargs(self) -> Dict[str, Any]:
        """Convert to SSL context kwargs."""
        import ssl
        
        kwargs = {
            "minimum_version": getattr(ssl.TLSVersion, self.min_version.value.replace(".", "_")),
            "maximum_version": getattr(ssl.TLSVersion, self.max_version.value.replace(".", "_")),
        }
        
        # Set cipher suites
        if self.cipher_suites:
            kwargs["ciphers"] = ":".join(self.cipher_suites)
        
        return kwargs


@dataclass
class SecurityConfig:
    """Overall security configuration."""
    
    # TLS configuration
    tls: TLSConfig = field(default_factory=TLSConfig)
    
    # Authentication
    enable_auth: bool = True
    auth_token_ttl: int = 3600  # 1 hour
    max_failed_auth_attempts: int = 5
    auth_lockout_duration: int = 300  # 5 minutes
    
    # Rate limiting
    enable_rate_limiting: bool = True
    rate_limit_requests: int = 100
    rate_limit_window: int = 60  # 1 minute
    
    # Security headers
    enable_security_headers: bool = True
    hsts_max_age: int = 31536000  # 1 year
    content_security_policy: str = "default-src 'self'"
    
    # Audit logging
    enable_audit_logging: bool = True
    audit_log_file: Optional[str] = None
    audit_log_level: str = "INFO"
    
    # IP filtering
    enable_ip_filtering: bool = False
    allowed_ips: List[str] = field(default_factory=list)
    blocked_ips: List[str] = field(default_factory=list)
    
    # Certificate validation
    check_certificate_revocation: bool = True
    certificate_transparency: bool = True
    
    # Session security
    secure_cookies: bool = True
    same_site_cookies: str = "strict"
    
    @classmethod
    def from_env(cls, prefix: str = "SECURITY") -> "SecurityConfig":
        """Create configuration from environment variables."""
        config = cls()
        
        # TLS config
        config.tls = TLSConfig.from_env()
        
        # Authentication
        if enable_auth := os.getenv(f"{prefix}_ENABLE_AUTH"):
            config.enable_auth = enable_auth.lower() in ("true", "1", "yes")
        
        if auth_token_ttl := os.getenv(f"{prefix}_AUTH_TOKEN_TTL"):
            config.auth_token_ttl = int(auth_token_ttl)
        
        # Rate limiting
        if rate_limit := os.getenv(f"{prefix}_RATE_LIMIT"):
            config.rate_limit_requests = int(rate_limit)
        
        # IP filtering
        if allowed_ips := os.getenv(f"{prefix}_ALLOWED_IPS"):
            config.allowed_ips = allowed_ips.split(",")
        
        if blocked_ips := os.getenv(f"{prefix}_BLOCKED_IPS"):
            config.blocked_ips = blocked_ips.split(",")
        
        # Audit logging
        if audit_log := os.getenv(f"{prefix}_AUDIT_LOG"):
            config.audit_log_file = audit_log
        
        return config


def validate_tls_config(config: TLSConfig) -> List[str]:
    """Validate TLS configuration."""
    errors = []
    
    # Validate version compatibility
    if config.min_version == TLSVersion.TLS_1_3 and config.max_version == TLSVersion.TLS_1_2:
        errors.append("Minimum TLS version cannot be higher than maximum version")
    
    # Validate certificate config
    cert_errors = config.certificate.validate()
    errors.extend(cert_errors)
    
    # Check client auth requirements
    if config.client_auth_mode in [ClientAuthMode.OPTIONAL, ClientAuthMode.REQUIRED]:
        if not config.certificate.ca_file and not config.certificate.ca_bundle:
            errors.append("Client authentication requires CA certificate")
    
    # Validate cipher suites
    if not config.cipher_suites:
        errors.append("At least one cipher suite must be specified")
    
    return errors


def get_secure_defaults() -> TLSConfig:
    """Get secure default TLS configuration."""
    return TLSConfig(
        min_version=TLSVersion.TLS_1_2,
        max_version=TLSVersion.TLS_1_3,
        cipher_suites=[
            # TLS 1.3 suites
            CipherSuite.TLS_AES_256_GCM_SHA384.value,
            CipherSuite.TLS_CHACHA20_POLY1305_SHA256.value,
            CipherSuite.TLS_AES_128_GCM_SHA256.value,
            # TLS 1.2 suites with forward secrecy
            CipherSuite.ECDHE_ECDSA_AES256_GCM_SHA384.value,
            CipherSuite.ECDHE_RSA_AES256_GCM_SHA384.value,
            CipherSuite.ECDHE_ECDSA_AES128_GCM_SHA256.value,
            CipherSuite.ECDHE_RSA_AES128_GCM_SHA256.value,
        ],
        verify_hostname=True,
        verify_depth=5,
        enable_ocsp_stapling=True,
        prefer_server_ciphers=True,
        enable_session_tickets=True,
        enable_sni=True
    )


def get_development_config() -> TLSConfig:
    """Get development TLS configuration (less strict)."""
    config = get_secure_defaults()
    config.verify_hostname = False
    config.client_auth_mode = ClientAuthMode.NONE
    return config


def get_production_config() -> TLSConfig:
    """Get production TLS configuration (most strict)."""
    config = get_secure_defaults()
    config.min_version = TLSVersion.TLS_1_3  # TLS 1.3 only
    config.client_auth_mode = ClientAuthMode.REQUIRED  # mTLS required
    config.verify_depth = 3  # Stricter chain validation
    return config