"""
Certificate management and generation.
"""

import os
import logging
from typing import Optional, List, Tuple, Dict, Any
from datetime import datetime, timedelta
from pathlib import Path
import ipaddress
import subprocess
from dataclasses import dataclass

from cryptography import x509
from cryptography.x509.oid import NameOID, ExtensionOID, ExtendedKeyUsageOID
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.backends import default_backend

from .config import CertificateConfig

logger = logging.getLogger(__name__)


@dataclass
class Certificate:
    """Certificate data structure."""
    
    cert: x509.Certificate
    key: Optional[rsa.RSAPrivateKey] = None
    cert_pem: Optional[str] = None
    key_pem: Optional[str] = None
    chain_pem: Optional[str] = None
    
    @property
    def common_name(self) -> str:
        """Get certificate common name."""
        for attribute in self.cert.subject:
            if attribute.oid == NameOID.COMMON_NAME:
                return attribute.value
        return ""
    
    @property
    def serial_number(self) -> int:
        """Get certificate serial number."""
        return self.cert.serial_number
    
    @property
    def not_valid_before(self) -> datetime:
        """Get certificate start validity."""
        return self.cert.not_valid_before
    
    @property
    def not_valid_after(self) -> datetime:
        """Get certificate end validity."""
        return self.cert.not_valid_after
    
    @property
    def is_expired(self) -> bool:
        """Check if certificate is expired."""
        return datetime.utcnow() > self.not_valid_after
    
    @property
    def days_until_expiry(self) -> int:
        """Get days until certificate expires."""
        delta = self.not_valid_after - datetime.utcnow()
        return delta.days
    
    @property
    def is_ca(self) -> bool:
        """Check if certificate is a CA."""
        try:
            basic_constraints = self.cert.extensions.get_extension_for_oid(
                ExtensionOID.BASIC_CONSTRAINTS
            ).value
            return basic_constraints.ca
        except x509.ExtensionNotFound:
            return False
    
    @property
    def san_dns_names(self) -> List[str]:
        """Get SAN DNS names."""
        try:
            san = self.cert.extensions.get_extension_for_oid(
                ExtensionOID.SUBJECT_ALTERNATIVE_NAME
            ).value
            return [name.value for name in san if isinstance(name, x509.DNSName)]
        except x509.ExtensionNotFound:
            return []
    
    @property
    def san_ip_addresses(self) -> List[str]:
        """Get SAN IP addresses."""
        try:
            san = self.cert.extensions.get_extension_for_oid(
                ExtensionOID.SUBJECT_ALTERNATIVE_NAME
            ).value
            return [str(ip.value) for ip in san if isinstance(ip, x509.IPAddress)]
        except x509.ExtensionNotFound:
            return []
    
    def to_pem(self) -> str:
        """Export certificate to PEM format."""
        if self.cert_pem:
            return self.cert_pem
        
        return self.cert.public_bytes(
            encoding=serialization.Encoding.PEM
        ).decode('utf-8')
    
    def key_to_pem(self, password: Optional[bytes] = None) -> str:
        """Export private key to PEM format."""
        if not self.key:
            raise ValueError("No private key available")
        
        if self.key_pem and not password:
            return self.key_pem
        
        encryption = serialization.NoEncryption()
        if password:
            encryption = serialization.BestAvailableEncryption(password)
        
        return self.key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.TraditionalOpenSSL,
            encryption_algorithm=encryption
        ).decode('utf-8')
    
    def save(self, cert_path: str, key_path: Optional[str] = None, password: Optional[bytes] = None):
        """Save certificate and key to files."""
        # Save certificate
        with open(cert_path, 'w') as f:
            f.write(self.to_pem())
        
        # Save key if provided
        if key_path and self.key:
            with open(key_path, 'w') as f:
                f.write(self.key_to_pem(password))
            
            # Set secure permissions on key file
            os.chmod(key_path, 0o600)


class CertificateManager:
    """Manages certificates for the application."""
    
    def __init__(self, base_path: str = "certs"):
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        
        # Certificate storage
        self._certificates: Dict[str, Certificate] = {}
        self._ca_certificates: Dict[str, Certificate] = {}
    
    def generate_private_key(self, key_size: int = 2048) -> rsa.RSAPrivateKey:
        """Generate a new RSA private key."""
        return rsa.generate_private_key(
            public_exponent=65537,
            key_size=key_size,
            backend=default_backend()
        )
    
    def generate_self_signed_certificate(
        self,
        config: CertificateConfig,
        key: Optional[rsa.RSAPrivateKey] = None
    ) -> Certificate:
        """Generate a self-signed certificate."""
        if not key:
            key = self.generate_private_key(config.key_size)
        
        # Build subject
        subject_components = []
        if config.country:
            subject_components.append(x509.NameAttribute(NameOID.COUNTRY_NAME, config.country))
        if config.state:
            subject_components.append(x509.NameAttribute(NameOID.STATE_OR_PROVINCE_NAME, config.state))
        if config.locality:
            subject_components.append(x509.NameAttribute(NameOID.LOCALITY_NAME, config.locality))
        if config.organization:
            subject_components.append(x509.NameAttribute(NameOID.ORGANIZATION_NAME, config.organization))
        if config.organizational_unit:
            subject_components.append(x509.NameAttribute(NameOID.ORGANIZATIONAL_UNIT_NAME, config.organizational_unit))
        
        subject_components.append(x509.NameAttribute(
            NameOID.COMMON_NAME, 
            config.common_name or "BrahminyKite"
        ))
        
        subject = x509.Name(subject_components)
        
        # Create certificate builder
        builder = x509.CertificateBuilder()
        builder = builder.subject_name(subject)
        builder = builder.issuer_name(subject)  # Self-signed
        builder = builder.public_key(key.public_key())
        builder = builder.serial_number(x509.random_serial_number())
        builder = builder.not_valid_before(datetime.utcnow())
        builder = builder.not_valid_after(datetime.utcnow() + timedelta(days=config.days_valid))
        
        # Add extensions
        builder = builder.add_extension(
            x509.BasicConstraints(ca=config.is_ca, path_length=None),
            critical=True,
        )
        
        # Key usage
        if config.is_ca:
            builder = builder.add_extension(
                x509.KeyUsage(
                    digital_signature=True,
                    content_commitment=False,
                    key_encipherment=True,
                    data_encipherment=False,
                    key_agreement=False,
                    key_cert_sign=True,
                    crl_sign=True,
                    encipher_only=False,
                    decipher_only=False,
                ),
                critical=True,
            )
        else:
            builder = builder.add_extension(
                x509.KeyUsage(
                    digital_signature=True,
                    content_commitment=False,
                    key_encipherment=True,
                    data_encipherment=False,
                    key_agreement=False,
                    key_cert_sign=False,
                    crl_sign=False,
                    encipher_only=False,
                    decipher_only=False,
                ),
                critical=True,
            )
        
        # Extended key usage
        if not config.is_ca:
            builder = builder.add_extension(
                x509.ExtendedKeyUsage([
                    ExtendedKeyUsageOID.SERVER_AUTH,
                    ExtendedKeyUsageOID.CLIENT_AUTH,
                ]),
                critical=True,
            )
        
        # Subject Alternative Names
        san_list = []
        
        # Add DNS names
        for dns_name in config.san_dns:
            san_list.append(x509.DNSName(dns_name))
        
        # Add IP addresses
        for ip_str in config.san_ip:
            try:
                ip = ipaddress.ip_address(ip_str)
                san_list.append(x509.IPAddress(ip))
            except ValueError:
                logger.warning(f"Invalid IP address in SAN: {ip_str}")
        
        # Add common name to SAN if not already present
        if config.common_name and config.common_name not in config.san_dns:
            san_list.append(x509.DNSName(config.common_name))
        
        if san_list:
            builder = builder.add_extension(
                x509.SubjectAlternativeName(san_list),
                critical=False,
            )
        
        # Subject Key Identifier
        builder = builder.add_extension(
            x509.SubjectKeyIdentifier.from_public_key(key.public_key()),
            critical=False,
        )
        
        # Sign the certificate
        certificate = builder.sign(key, hashes.SHA256(), backend=default_backend())
        
        return Certificate(
            cert=certificate,
            key=key,
            cert_pem=certificate.public_bytes(serialization.Encoding.PEM).decode('utf-8'),
            key_pem=key.private_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PrivateFormat.TraditionalOpenSSL,
                encryption_algorithm=serialization.NoEncryption()
            ).decode('utf-8')
        )
    
    def generate_ca_certificate(self, config: CertificateConfig) -> Certificate:
        """Generate a CA certificate."""
        config.is_ca = True
        ca_cert = self.generate_self_signed_certificate(config)
        
        # Store CA certificate
        ca_name = config.common_name or "BrahminyKite-CA"
        self._ca_certificates[ca_name] = ca_cert
        
        # Save CA certificate
        ca_path = self.base_path / "ca"
        ca_path.mkdir(exist_ok=True)
        
        ca_cert.save(
            str(ca_path / f"{ca_name}.crt"),
            str(ca_path / f"{ca_name}.key")
        )
        
        logger.info(f"Generated CA certificate: {ca_name}")
        return ca_cert
    
    def generate_signed_certificate(
        self,
        config: CertificateConfig,
        ca_cert: Certificate,
        key: Optional[rsa.RSAPrivateKey] = None
    ) -> Certificate:
        """Generate a certificate signed by a CA."""
        if not ca_cert.key:
            raise ValueError("CA certificate must have private key")
        
        if not key:
            key = self.generate_private_key(config.key_size)
        
        # Build subject
        subject_components = []
        if config.country:
            subject_components.append(x509.NameAttribute(NameOID.COUNTRY_NAME, config.country))
        if config.state:
            subject_components.append(x509.NameAttribute(NameOID.STATE_OR_PROVINCE_NAME, config.state))
        if config.locality:
            subject_components.append(x509.NameAttribute(NameOID.LOCALITY_NAME, config.locality))
        if config.organization:
            subject_components.append(x509.NameAttribute(NameOID.ORGANIZATION_NAME, config.organization))
        if config.organizational_unit:
            subject_components.append(x509.NameAttribute(NameOID.ORGANIZATIONAL_UNIT_NAME, config.organizational_unit))
        
        subject_components.append(x509.NameAttribute(
            NameOID.COMMON_NAME,
            config.common_name or "BrahminyKite-Server"
        ))
        
        subject = x509.Name(subject_components)
        
        # Create certificate builder
        builder = x509.CertificateBuilder()
        builder = builder.subject_name(subject)
        builder = builder.issuer_name(ca_cert.cert.subject)
        builder = builder.public_key(key.public_key())
        builder = builder.serial_number(x509.random_serial_number())
        builder = builder.not_valid_before(datetime.utcnow())
        builder = builder.not_valid_after(datetime.utcnow() + timedelta(days=config.days_valid))
        
        # Add extensions
        builder = builder.add_extension(
            x509.BasicConstraints(ca=False, path_length=None),
            critical=True,
        )
        
        # Key usage
        builder = builder.add_extension(
            x509.KeyUsage(
                digital_signature=True,
                content_commitment=False,
                key_encipherment=True,
                data_encipherment=False,
                key_agreement=False,
                key_cert_sign=False,
                crl_sign=False,
                encipher_only=False,
                decipher_only=False,
            ),
            critical=True,
        )
        
        # Extended key usage
        builder = builder.add_extension(
            x509.ExtendedKeyUsage([
                ExtendedKeyUsageOID.SERVER_AUTH,
                ExtendedKeyUsageOID.CLIENT_AUTH,
            ]),
            critical=True,
        )
        
        # Subject Alternative Names
        san_list = []
        
        # Add DNS names
        for dns_name in config.san_dns:
            san_list.append(x509.DNSName(dns_name))
        
        # Add IP addresses
        for ip_str in config.san_ip:
            try:
                ip = ipaddress.ip_address(ip_str)
                san_list.append(x509.IPAddress(ip))
            except ValueError:
                logger.warning(f"Invalid IP address in SAN: {ip_str}")
        
        # Add common name to SAN if not already present
        if config.common_name and config.common_name not in config.san_dns:
            san_list.append(x509.DNSName(config.common_name))
        
        if san_list:
            builder = builder.add_extension(
                x509.SubjectAlternativeName(san_list),
                critical=False,
            )
        
        # Authority Key Identifier
        builder = builder.add_extension(
            x509.AuthorityKeyIdentifier.from_issuer_public_key(ca_cert.cert.public_key()),
            critical=False,
        )
        
        # Sign the certificate
        certificate = builder.sign(ca_cert.key, hashes.SHA256(), backend=default_backend())
        
        return Certificate(
            cert=certificate,
            key=key,
            cert_pem=certificate.public_bytes(serialization.Encoding.PEM).decode('utf-8'),
            key_pem=key.private_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PrivateFormat.TraditionalOpenSSL,
                encryption_algorithm=serialization.NoEncryption()
            ).decode('utf-8')
        )
    
    def load_certificate(self, cert_path: str, key_path: Optional[str] = None) -> Certificate:
        """Load certificate from file."""
        with open(cert_path, 'rb') as f:
            cert_data = f.read()
        
        cert = x509.load_pem_x509_certificate(cert_data, backend=default_backend())
        
        key = None
        key_pem = None
        if key_path:
            with open(key_path, 'rb') as f:
                key_data = f.read()
            
            key = serialization.load_pem_private_key(
                key_data,
                password=None,
                backend=default_backend()
            )
            key_pem = key_data.decode('utf-8')
        
        return Certificate(
            cert=cert,
            key=key,
            cert_pem=cert_data.decode('utf-8'),
            key_pem=key_pem
        )
    
    def store_certificate(self, name: str, certificate: Certificate):
        """Store certificate in manager."""
        self._certificates[name] = certificate
        logger.info(f"Stored certificate: {name}")
    
    def get_certificate(self, name: str) -> Optional[Certificate]:
        """Get certificate by name."""
        return self._certificates.get(name)
    
    def list_certificates(self) -> List[str]:
        """List all stored certificate names."""
        return list(self._certificates.keys())
    
    def check_expiry(self, days_threshold: int = 30) -> List[Tuple[str, Certificate, int]]:
        """Check for certificates expiring soon."""
        expiring = []
        
        for name, cert in self._certificates.items():
            days_left = cert.days_until_expiry
            if days_left <= days_threshold:
                expiring.append((name, cert, days_left))
        
        return expiring


class CertificateAuthority:
    """Simple certificate authority for development/testing."""
    
    def __init__(self, ca_cert: Certificate, base_path: str = "ca"):
        if not ca_cert.is_ca:
            raise ValueError("Certificate is not a CA certificate")
        
        self.ca_cert = ca_cert
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        
        # Track issued certificates
        self._issued_certificates: Dict[str, Certificate] = {}
    
    def issue_certificate(
        self,
        common_name: str,
        san_dns: Optional[List[str]] = None,
        san_ip: Optional[List[str]] = None,
        days_valid: int = 365,
        key_size: int = 2048
    ) -> Certificate:
        """Issue a new certificate."""
        config = CertificateConfig(
            common_name=common_name,
            organization=self.ca_cert.cert.subject.get_attributes_for_oid(NameOID.ORGANIZATION_NAME)[0].value,
            san_dns=san_dns or [],
            san_ip=san_ip or [],
            days_valid=days_valid,
            key_size=key_size
        )
        
        # Generate key
        key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=key_size,
            backend=default_backend()
        )
        
        # Generate certificate
        manager = CertificateManager()
        cert = manager.generate_signed_certificate(config, self.ca_cert, key)
        
        # Store certificate
        self._issued_certificates[common_name] = cert
        
        # Save certificate
        cert_path = self.base_path / "issued"
        cert_path.mkdir(exist_ok=True)
        
        cert.save(
            str(cert_path / f"{common_name}.crt"),
            str(cert_path / f"{common_name}.key")
        )
        
        logger.info(f"Issued certificate for: {common_name}")
        return cert
    
    def revoke_certificate(self, common_name: str, reason: str = "unspecified"):
        """Revoke a certificate (basic implementation)."""
        if common_name not in self._issued_certificates:
            raise ValueError(f"Certificate not found: {common_name}")
        
        # In a real implementation, this would update a CRL
        # For now, just remove from issued certificates
        del self._issued_certificates[common_name]
        
        logger.info(f"Revoked certificate: {common_name} (reason: {reason})")
    
    def get_issued_certificates(self) -> List[Tuple[str, Certificate]]:
        """Get all issued certificates."""
        return list(self._issued_certificates.items())


# Convenience functions

def generate_self_signed_cert(
    common_name: str,
    san_dns: Optional[List[str]] = None,
    san_ip: Optional[List[str]] = None,
    days_valid: int = 365,
    key_size: int = 2048
) -> Certificate:
    """Generate a self-signed certificate."""
    config = CertificateConfig(
        common_name=common_name,
        organization="BrahminyKite",
        san_dns=san_dns or [],
        san_ip=san_ip or [],
        days_valid=days_valid,
        key_size=key_size
    )
    
    manager = CertificateManager()
    return manager.generate_self_signed_certificate(config)


def load_certificate(cert_path: str, key_path: Optional[str] = None) -> Certificate:
    """Load certificate from file."""
    manager = CertificateManager()
    return manager.load_certificate(cert_path, key_path)


def validate_certificate(cert: Certificate, hostname: Optional[str] = None) -> List[str]:
    """Validate certificate."""
    errors = []
    
    # Check expiry
    if cert.is_expired:
        errors.append("Certificate is expired")
    elif cert.days_until_expiry < 30:
        errors.append(f"Certificate expires in {cert.days_until_expiry} days")
    
    # Check hostname if provided
    if hostname:
        valid_names = [cert.common_name] + cert.san_dns_names
        if hostname not in valid_names:
            # Check wildcard certificates
            wildcard_match = False
            for name in valid_names:
                if name.startswith("*.") and hostname.endswith(name[2:]):
                    wildcard_match = True
                    break
            
            if not wildcard_match:
                errors.append(f"Hostname {hostname} not in certificate")
    
    return errors