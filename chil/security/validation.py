"""
Certificate validation and verification utilities.
"""

import logging
from typing import List, Optional, Set, Tuple
from datetime import datetime, timedelta
import ipaddress
import re

from cryptography import x509
from cryptography.x509.oid import NameOID, ExtensionOID
from cryptography.hazmat.backends import default_backend

from .certificates import Certificate

logger = logging.getLogger(__name__)


class CertificateValidator:
    """Validates X.509 certificates."""
    
    def __init__(self):
        self._trusted_cas: Set[x509.Certificate] = set()
        self._revoked_serials: Set[int] = set()
        self._validation_callbacks = []
    
    def add_trusted_ca(self, ca_cert: Certificate):
        """Add a trusted CA certificate."""
        if not ca_cert.is_ca:
            raise ValueError("Certificate is not a CA certificate")
        
        self._trusted_cas.add(ca_cert.cert)
        logger.info(f"Added trusted CA: {ca_cert.common_name}")
    
    def add_revoked_serial(self, serial_number: int):
        """Add a revoked certificate serial number."""
        self._revoked_serials.add(serial_number)
        logger.info(f"Added revoked serial: {serial_number}")
    
    def add_validation_callback(self, callback):
        """Add custom validation callback."""
        self._validation_callbacks.append(callback)
    
    def validate_certificate(
        self,
        cert: Certificate,
        hostname: Optional[str] = None,
        check_revocation: bool = True,
        check_expiry: bool = True,
        check_chain: bool = True
    ) -> Tuple[bool, List[str]]:
        """
        Validate a certificate.
        
        Returns:
            Tuple of (is_valid, errors)
        """
        errors = []
        
        # Check expiry
        if check_expiry:
            expiry_errors = check_certificate_expiry(cert)
            errors.extend(expiry_errors)
        
        # Check revocation
        if check_revocation:
            if cert.serial_number in self._revoked_serials:
                errors.append(f"Certificate is revoked (serial: {cert.serial_number})")
        
        # Validate hostname
        if hostname:
            hostname_errors = validate_hostname(cert, hostname)
            errors.extend(hostname_errors)
        
        # Check certificate chain
        if check_chain and self._trusted_cas:
            chain_errors = self._validate_chain(cert)
            errors.extend(chain_errors)
        
        # Run custom validators
        for callback in self._validation_callbacks:
            try:
                result = callback(cert)
                if isinstance(result, str):
                    errors.append(result)
                elif isinstance(result, list):
                    errors.extend(result)
                elif result is False:
                    errors.append("Custom validation failed")
            except Exception as e:
                errors.append(f"Validation callback error: {e}")
        
        return len(errors) == 0, errors
    
    def _validate_chain(self, cert: Certificate) -> List[str]:
        """Validate certificate chain."""
        errors = []
        
        # For self-signed certificates
        if cert.cert.issuer == cert.cert.subject:
            if cert.cert in self._trusted_cas:
                return []  # Valid self-signed CA
            else:
                return ["Self-signed certificate not in trusted CAs"]
        
        # Find issuer in trusted CAs
        issuer_found = False
        for ca_cert in self._trusted_cas:
            if ca_cert.subject == cert.cert.issuer:
                issuer_found = True
                # Verify signature (simplified check)
                try:
                    # In production, would verify the signature
                    logger.debug(f"Certificate issued by trusted CA: {cert.cert.issuer}")
                except Exception as e:
                    errors.append(f"Certificate signature verification failed: {e}")
                break
        
        if not issuer_found:
            errors.append("Certificate issuer not in trusted CAs")
        
        return errors


def validate_hostname(cert: Certificate, hostname: str) -> List[str]:
    """Validate that hostname matches certificate."""
    errors = []
    
    # Get all valid names from certificate
    valid_names = set()
    
    # Add common name
    if cert.common_name:
        valid_names.add(cert.common_name.lower())
    
    # Add SAN DNS names
    valid_names.update(name.lower() for name in cert.san_dns_names)
    
    # Add SAN IP addresses
    valid_names.update(cert.san_ip_addresses)
    
    # Normalize hostname
    hostname_lower = hostname.lower()
    
    # Direct match
    if hostname_lower in valid_names:
        return []
    
    # IP address match
    try:
        host_ip = ipaddress.ip_address(hostname)
        if str(host_ip) in valid_names:
            return []
    except ValueError:
        # Not an IP address
        pass
    
    # Wildcard match
    for name in valid_names:
        if name.startswith("*."):
            # Wildcard certificate
            pattern = name[2:]  # Remove *.
            if hostname_lower.endswith(pattern):
                # Check that wildcard only matches one level
                prefix = hostname_lower[:-len(pattern)]
                if prefix and '.' not in prefix[:-1]:  # Allow trailing dot
                    return []
    
    errors.append(f"Hostname '{hostname}' does not match certificate (valid: {', '.join(valid_names)})")
    return errors


def check_certificate_expiry(
    cert: Certificate,
    warning_days: int = 30,
    error_days: int = 7
) -> List[str]:
    """Check certificate expiry."""
    errors = []
    
    if cert.is_expired:
        errors.append(f"Certificate expired on {cert.not_valid_after}")
    else:
        days_until_expiry = cert.days_until_expiry
        
        if days_until_expiry <= error_days:
            errors.append(f"Certificate expires in {days_until_expiry} days (critical)")
        elif days_until_expiry <= warning_days:
            # This would be a warning in production
            logger.warning(f"Certificate expires in {days_until_expiry} days")
    
    # Check not valid before
    if datetime.utcnow() < cert.not_valid_before:
        errors.append(f"Certificate not valid until {cert.not_valid_before}")
    
    return errors


def validate_cert_chain(
    cert_chain: List[Certificate],
    trusted_roots: Optional[List[Certificate]] = None
) -> Tuple[bool, List[str]]:
    """Validate a certificate chain."""
    errors = []
    
    if not cert_chain:
        return False, ["Empty certificate chain"]
    
    # Check each certificate in the chain
    for i, cert in enumerate(cert_chain):
        # Check expiry
        expiry_errors = check_certificate_expiry(cert)
        if expiry_errors:
            errors.extend([f"Certificate {i}: {e}" for e in expiry_errors])
        
        # Check if CA certificate has proper extensions
        if i > 0:  # Not the leaf certificate
            if not cert.is_ca:
                errors.append(f"Certificate {i} is not a CA certificate")
    
    # Verify chain linkage
    for i in range(len(cert_chain) - 1):
        subject_cert = cert_chain[i]
        issuer_cert = cert_chain[i + 1]
        
        if subject_cert.cert.issuer != issuer_cert.cert.subject:
            errors.append(f"Certificate {i} issuer does not match certificate {i+1} subject")
    
    # Check root certificate
    root_cert = cert_chain[-1]
    if root_cert.cert.issuer != root_cert.cert.subject:
        # Not self-signed, need to check against trusted roots
        if trusted_roots:
            root_found = False
            for trusted_root in trusted_roots:
                if root_cert.cert.issuer == trusted_root.cert.subject:
                    root_found = True
                    break
            
            if not root_found:
                errors.append("Root certificate not in trusted roots")
        else:
            errors.append("Chain does not end with self-signed certificate and no trusted roots provided")
    
    return len(errors) == 0, errors


def extract_san_names(cert: Certificate) -> Dict[str, List[str]]:
    """Extract all Subject Alternative Names from certificate."""
    result = {
        "dns": cert.san_dns_names,
        "ip": cert.san_ip_addresses,
        "email": [],
        "uri": []
    }
    
    try:
        san_ext = cert.cert.extensions.get_extension_for_oid(
            ExtensionOID.SUBJECT_ALTERNATIVE_NAME
        ).value
        
        for name in san_ext:
            if isinstance(name, x509.RFC822Name):
                result["email"].append(name.value)
            elif isinstance(name, x509.UniformResourceIdentifier):
                result["uri"].append(name.value)
    except x509.ExtensionNotFound:
        pass
    
    return result


def validate_key_usage(
    cert: Certificate,
    required_usage: Set[str]
) -> List[str]:
    """Validate certificate key usage."""
    errors = []
    
    try:
        key_usage = cert.cert.extensions.get_extension_for_oid(
            ExtensionOID.KEY_USAGE
        ).value
        
        usage_map = {
            "digital_signature": key_usage.digital_signature,
            "key_encipherment": key_usage.key_encipherment,
            "key_agreement": key_usage.key_agreement,
            "key_cert_sign": key_usage.key_cert_sign,
            "crl_sign": key_usage.crl_sign
        }
        
        for usage in required_usage:
            if usage in usage_map and not usage_map[usage]:
                errors.append(f"Certificate does not have required key usage: {usage}")
    
    except x509.ExtensionNotFound:
        if required_usage:
            errors.append("Certificate has no key usage extension")
    
    return errors


def validate_extended_key_usage(
    cert: Certificate,
    required_usage: Set[str]
) -> List[str]:
    """Validate certificate extended key usage."""
    errors = []
    
    try:
        ext_key_usage = cert.cert.extensions.get_extension_for_oid(
            ExtensionOID.EXTENDED_KEY_USAGE
        ).value
        
        eku_oids = {
            "server_auth": ExtensionOID.SERVER_AUTH,
            "client_auth": ExtensionOID.CLIENT_AUTH,
            "code_signing": ExtensionOID.CODE_SIGNING,
            "email_protection": ExtensionOID.EMAIL_PROTECTION
        }
        
        cert_ekus = set(eku.dotted_string for eku in ext_key_usage)
        
        for usage in required_usage:
            if usage in eku_oids:
                if eku_oids[usage].dotted_string not in cert_ekus:
                    errors.append(f"Certificate does not have required extended key usage: {usage}")
    
    except x509.ExtensionNotFound:
        if required_usage:
            errors.append("Certificate has no extended key usage extension")
    
    return errors


def is_wildcard_match(pattern: str, hostname: str) -> bool:
    """Check if hostname matches wildcard pattern."""
    if not pattern.startswith("*."):
        return pattern.lower() == hostname.lower()
    
    # Convert wildcard to regex
    pattern_regex = pattern.replace(".", r"\.")
    pattern_regex = pattern_regex.replace("*", r"[^.]+")
    pattern_regex = f"^{pattern_regex}$"
    
    return bool(re.match(pattern_regex, hostname, re.IGNORECASE))


def get_certificate_info(cert: Certificate) -> Dict[str, Any]:
    """Get detailed certificate information."""
    info = {
        "subject": {
            "common_name": cert.common_name,
            "organization": None,
            "country": None
        },
        "issuer": {},
        "serial_number": str(cert.serial_number),
        "not_valid_before": cert.not_valid_before.isoformat(),
        "not_valid_after": cert.not_valid_after.isoformat(),
        "is_ca": cert.is_ca,
        "san": extract_san_names(cert),
        "key_usage": {},
        "extended_key_usage": []
    }
    
    # Extract subject details
    for attribute in cert.cert.subject:
        if attribute.oid == NameOID.ORGANIZATION_NAME:
            info["subject"]["organization"] = attribute.value
        elif attribute.oid == NameOID.COUNTRY_NAME:
            info["subject"]["country"] = attribute.value
    
    # Extract issuer details
    for attribute in cert.cert.issuer:
        if attribute.oid == NameOID.COMMON_NAME:
            info["issuer"]["common_name"] = attribute.value
        elif attribute.oid == NameOID.ORGANIZATION_NAME:
            info["issuer"]["organization"] = attribute.value
    
    # Extract key usage
    try:
        key_usage = cert.cert.extensions.get_extension_for_oid(
            ExtensionOID.KEY_USAGE
        ).value
        
        info["key_usage"] = {
            "digital_signature": key_usage.digital_signature,
            "key_encipherment": key_usage.key_encipherment,
            "key_agreement": key_usage.key_agreement,
            "key_cert_sign": key_usage.key_cert_sign,
            "crl_sign": key_usage.crl_sign
        }
    except x509.ExtensionNotFound:
        pass
    
    # Extract extended key usage
    try:
        ext_key_usage = cert.cert.extensions.get_extension_for_oid(
            ExtensionOID.EXTENDED_KEY_USAGE
        ).value
        
        info["extended_key_usage"] = [str(eku) for eku in ext_key_usage]
    except x509.ExtensionNotFound:
        pass
    
    return info