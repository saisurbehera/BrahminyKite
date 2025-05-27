"""Certificate rotation and renewal functionality."""

import asyncio
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Set, Callable, Any
from dataclasses import dataclass, field
from enum import Enum

from cryptography import x509
from cryptography.x509.oid import NameOID
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa

from .config import CertificateConfig
from .certificates import CertificateManager, Certificate
from .validation import CertificateValidator

logger = logging.getLogger(__name__)


class RotationStrategy(Enum):
    """Certificate rotation strategies."""
    TIME_BASED = "time_based"
    ON_DEMAND = "on_demand"
    EVENT_DRIVEN = "event_driven"
    HYBRID = "hybrid"


class RenewalTrigger(Enum):
    """Certificate renewal triggers."""
    EXPIRY_THRESHOLD = "expiry_threshold"
    MANUAL = "manual"
    SECURITY_EVENT = "security_event"
    COMPLIANCE = "compliance"


@dataclass
class RotationConfig:
    """Certificate rotation configuration."""
    strategy: RotationStrategy = RotationStrategy.TIME_BASED
    check_interval: timedelta = timedelta(hours=1)
    renewal_threshold: timedelta = timedelta(days=30)
    grace_period: timedelta = timedelta(days=7)
    max_versions: int = 5
    archive_old_certs: bool = True
    archive_path: Optional[Path] = None
    notification_callbacks: List[Callable[[str, Certificate], None]] = field(default_factory=list)


@dataclass
class CertificateVersion:
    """Represents a version of a certificate."""
    certificate: Certificate
    private_key: rsa.RSAPrivateKey
    version: int
    created_at: datetime
    expires_at: datetime
    is_active: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)


class CertificateRotationManager:
    """Manages certificate rotation and renewal."""
    
    def __init__(
        self,
        cert_manager: CertificateManager,
        validator: CertificateValidator,
        config: RotationConfig
    ):
        self.cert_manager = cert_manager
        self.validator = validator
        self.config = config
        self._certificates: Dict[str, List[CertificateVersion]] = {}
        self._rotation_tasks: Dict[str, asyncio.Task] = {}
        self._lock = asyncio.Lock()
        self._shutdown = False
        
    async def start(self):
        """Start rotation management."""
        logger.info("Starting certificate rotation manager")
        self._shutdown = False
        
        if self.config.strategy in [RotationStrategy.TIME_BASED, RotationStrategy.HYBRID]:
            asyncio.create_task(self._rotation_loop())
            
    async def stop(self):
        """Stop rotation management."""
        logger.info("Stopping certificate rotation manager")
        self._shutdown = True
        
        for task in self._rotation_tasks.values():
            task.cancel()
            
        await asyncio.gather(*self._rotation_tasks.values(), return_exceptions=True)
        
    async def register_certificate(
        self,
        name: str,
        certificate: Certificate,
        private_key: rsa.RSAPrivateKey,
        auto_renew: bool = True
    ) -> CertificateVersion:
        """Register a certificate for rotation management."""
        async with self._lock:
            if name not in self._certificates:
                self._certificates[name] = []
                
            version = len(self._certificates[name]) + 1
            expires_at = certificate.not_valid_after
            
            cert_version = CertificateVersion(
                certificate=certificate,
                private_key=private_key,
                version=version,
                created_at=datetime.utcnow(),
                expires_at=expires_at,
                metadata={"auto_renew": auto_renew}
            )
            
            self._certificates[name].append(cert_version)
            
            if auto_renew and self.config.strategy == RotationStrategy.EVENT_DRIVEN:
                await self._schedule_renewal(name, cert_version)
                
            logger.info(f"Registered certificate '{name}' version {version}")
            return cert_version
            
    async def get_active_certificate(self, name: str) -> Optional[CertificateVersion]:
        """Get the active certificate version."""
        async with self._lock:
            if name not in self._certificates:
                return None
                
            for cert in reversed(self._certificates[name]):
                if cert.is_active and self._is_valid(cert):
                    return cert
                    
            return None
            
    async def rotate_certificate(
        self,
        name: str,
        trigger: RenewalTrigger = RenewalTrigger.MANUAL,
        config: Optional[CertificateConfig] = None
    ) -> CertificateVersion:
        """Rotate a certificate."""
        async with self._lock:
            current = await self.get_active_certificate(name)
            if not current:
                raise ValueError(f"No active certificate found for '{name}'")
                
            # Generate new certificate
            if not config:
                # Create config from existing certificate
                config = self._create_config_from_certificate(current.certificate)
                
            new_key = self.cert_manager.generate_private_key(config.key_size)
            new_cert = self.cert_manager.generate_self_signed_certificate(config, new_key)
            
            # Register new version
            new_version = await self.register_certificate(
                name,
                new_cert,
                new_key,
                auto_renew=current.metadata.get("auto_renew", True)
            )
            
            # Start grace period for old certificate
            await self._start_grace_period(name, current, new_version)
            
            # Notify callbacks
            for callback in self.config.notification_callbacks:
                try:
                    callback(name, new_cert)
                except Exception as e:
                    logger.error(f"Notification callback failed: {e}")
                    
            logger.info(f"Rotated certificate '{name}' from version {current.version} to {new_version.version}")
            return new_version
            
    async def _rotation_loop(self):
        """Main rotation check loop."""
        while not self._shutdown:
            try:
                await self._check_certificates()
                await asyncio.sleep(self.config.check_interval.total_seconds())
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Rotation loop error: {e}")
                await asyncio.sleep(60)  # Wait before retry
                
    async def _check_certificates(self):
        """Check certificates for renewal needs."""
        async with self._lock:
            for name, versions in self._certificates.items():
                active = None
                for cert in reversed(versions):
                    if cert.is_active:
                        active = cert
                        break
                        
                if not active:
                    continue
                    
                if not active.metadata.get("auto_renew", True):
                    continue
                    
                # Check if renewal needed
                time_until_expiry = active.expires_at - datetime.utcnow()
                if time_until_expiry <= self.config.renewal_threshold:
                    logger.info(f"Certificate '{name}' needs renewal (expires in {time_until_expiry.days} days)")
                    
                    try:
                        await self.rotate_certificate(name, RenewalTrigger.EXPIRY_THRESHOLD)
                    except Exception as e:
                        logger.error(f"Failed to rotate certificate '{name}': {e}")
                        
    async def _schedule_renewal(self, name: str, cert_version: CertificateVersion):
        """Schedule certificate renewal."""
        time_until_renewal = (cert_version.expires_at - datetime.utcnow()) - self.config.renewal_threshold
        
        if time_until_renewal.total_seconds() > 0:
            async def renew_task():
                await asyncio.sleep(time_until_renewal.total_seconds())
                if not self._shutdown:
                    try:
                        await self.rotate_certificate(name, RenewalTrigger.EXPIRY_THRESHOLD)
                    except Exception as e:
                        logger.error(f"Scheduled renewal failed for '{name}': {e}")
                        
            task = asyncio.create_task(renew_task())
            self._rotation_tasks[f"{name}_{cert_version.version}"] = task
            
    async def _start_grace_period(self, name: str, old_version: CertificateVersion, new_version: CertificateVersion):
        """Start grace period for old certificate."""
        async def grace_period_task():
            await asyncio.sleep(self.config.grace_period.total_seconds())
            
            async with self._lock:
                old_version.is_active = False
                
                # Archive if configured
                if self.config.archive_old_certs:
                    await self._archive_certificate(name, old_version)
                    
                # Clean up old versions
                await self._cleanup_old_versions(name)
                
        asyncio.create_task(grace_period_task())
        
    async def _archive_certificate(self, name: str, cert_version: CertificateVersion):
        """Archive an old certificate."""
        if not self.config.archive_path:
            return
            
        archive_dir = self.config.archive_path / name
        archive_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = cert_version.created_at.strftime("%Y%m%d_%H%M%S")
        cert_file = archive_dir / f"cert_v{cert_version.version}_{timestamp}.pem"
        key_file = archive_dir / f"key_v{cert_version.version}_{timestamp}.pem"
        
        # Save certificate
        cert_pem = cert_version.certificate.public_bytes(serialization.Encoding.PEM)
        cert_file.write_bytes(cert_pem)
        
        # Save private key (encrypted)
        key_pem = cert_version.private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.TraditionalOpenSSL,
            encryption_algorithm=serialization.BestAvailableEncryption(b"archive_password")
        )
        key_file.write_bytes(key_pem)
        
        logger.info(f"Archived certificate '{name}' version {cert_version.version}")
        
    async def _cleanup_old_versions(self, name: str):
        """Clean up old certificate versions."""
        if name not in self._certificates:
            return
            
        versions = self._certificates[name]
        if len(versions) <= self.config.max_versions:
            return
            
        # Keep only the most recent versions
        versions_to_remove = len(versions) - self.config.max_versions
        for _ in range(versions_to_remove):
            old_version = versions.pop(0)
            logger.info(f"Removed old certificate version '{name}' v{old_version.version}")
            
    def _is_valid(self, cert_version: CertificateVersion) -> bool:
        """Check if certificate version is valid."""
        try:
            # Check expiry
            if datetime.utcnow() >= cert_version.expires_at:
                return False
                
            # Additional validation
            result = self.validator.validate_certificate(
                cert_version.certificate,
                check_hostname=False  # Skip hostname for stored certs
            )
            
            return result.is_valid
        except Exception as e:
            logger.error(f"Certificate validation error: {e}")
            return False
            
    def _create_config_from_certificate(self, cert: Certificate) -> CertificateConfig:
        """Create certificate config from existing certificate."""
        subject = cert.subject
        
        # Extract subject components
        common_name = None
        organization = None
        country = None
        
        for attribute in subject:
            if attribute.oid == NameOID.COMMON_NAME:
                common_name = attribute.value
            elif attribute.oid == NameOID.ORGANIZATION_NAME:
                organization = attribute.value
            elif attribute.oid == NameOID.COUNTRY_NAME:
                country = attribute.value
                
        # Extract SANs
        san_dns_names = []
        san_ip_addresses = []
        
        try:
            san_ext = cert.extensions.get_extension_for_oid(x509.oid.ExtensionOID.SUBJECT_ALTERNATIVE_NAME)
            for san in san_ext.value:
                if isinstance(san, x509.DNSName):
                    san_dns_names.append(san.value)
                elif isinstance(san, x509.IPAddress):
                    san_ip_addresses.append(san.value)
        except x509.ExtensionNotFound:
            pass
            
        return CertificateConfig(
            common_name=common_name or "localhost",
            organization=organization,
            country=country,
            san_dns_names=san_dns_names,
            san_ip_addresses=san_ip_addresses,
            validity_days=365,  # Default for renewal
            key_size=4096 if cert.public_key().key_size >= 4096 else 2048
        )
        
    async def get_certificate_info(self, name: str) -> Dict[str, Any]:
        """Get information about a certificate."""
        async with self._lock:
            if name not in self._certificates:
                return {"error": f"Certificate '{name}' not found"}
                
            versions = self._certificates[name]
            active = await self.get_active_certificate(name)
            
            return {
                "name": name,
                "total_versions": len(versions),
                "active_version": active.version if active else None,
                "versions": [
                    {
                        "version": v.version,
                        "created_at": v.created_at.isoformat(),
                        "expires_at": v.expires_at.isoformat(),
                        "is_active": v.is_active,
                        "is_valid": self._is_valid(v),
                        "days_until_expiry": (v.expires_at - datetime.utcnow()).days
                    }
                    for v in versions
                ]
            }
            
    async def force_renewal(self, name: str) -> CertificateVersion:
        """Force immediate certificate renewal."""
        return await self.rotate_certificate(name, RenewalTrigger.MANUAL)
        
    async def pause_auto_renewal(self, name: str):
        """Pause automatic renewal for a certificate."""
        async with self._lock:
            if name in self._certificates:
                for cert in self._certificates[name]:
                    cert.metadata["auto_renew"] = False
                    
    async def resume_auto_renewal(self, name: str):
        """Resume automatic renewal for a certificate."""
        async with self._lock:
            if name in self._certificates:
                for cert in self._certificates[name]:
                    cert.metadata["auto_renew"] = True


class CertificateRenewalService:
    """High-level certificate renewal service."""
    
    def __init__(self, rotation_manager: CertificateRotationManager):
        self.rotation_manager = rotation_manager
        self._monitors: Dict[str, asyncio.Task] = {}
        
    async def start_monitoring(self, certificates: Dict[str, tuple[Certificate, rsa.RSAPrivateKey]]):
        """Start monitoring certificates."""
        for name, (cert, key) in certificates.items():
            await self.rotation_manager.register_certificate(name, cert, key)
            
        await self.rotation_manager.start()
        
    async def stop_monitoring(self):
        """Stop monitoring certificates."""
        await self.rotation_manager.stop()
        
        for task in self._monitors.values():
            task.cancel()
            
        await asyncio.gather(*self._monitors.values(), return_exceptions=True)
        
    async def add_renewal_callback(self, callback: Callable[[str, Certificate], None]):
        """Add a callback for certificate renewals."""
        self.rotation_manager.config.notification_callbacks.append(callback)
        
    async def get_status(self) -> Dict[str, Any]:
        """Get renewal service status."""
        certs = {}
        
        for name in self.rotation_manager._certificates:
            certs[name] = await self.rotation_manager.get_certificate_info(name)
            
        return {
            "strategy": self.rotation_manager.config.strategy.value,
            "check_interval": str(self.rotation_manager.config.check_interval),
            "renewal_threshold": str(self.rotation_manager.config.renewal_threshold),
            "certificates": certs
        }