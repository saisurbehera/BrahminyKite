"""Certificate store for secure storage and retrieval."""

import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
from dataclasses import dataclass, asdict
from enum import Enum

from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend

from .certificates import Certificate

logger = logging.getLogger(__name__)


class StoreType(Enum):
    """Certificate store types."""
    FILE_SYSTEM = "file_system"
    MEMORY = "memory"
    DATABASE = "database"
    VAULT = "vault"


@dataclass
class StoredCertificate:
    """Represents a stored certificate."""
    name: str
    certificate_pem: bytes
    private_key_pem: bytes
    metadata: Dict[str, Any]
    created_at: datetime
    updated_at: datetime
    fingerprint: str


class CertificateStore:
    """Base certificate store interface."""
    
    async def store(
        self,
        name: str,
        certificate: Certificate,
        private_key: rsa.RSAPrivateKey,
        metadata: Optional[Dict[str, Any]] = None
    ) -> StoredCertificate:
        """Store a certificate and private key."""
        raise NotImplementedError
        
    async def retrieve(self, name: str) -> Optional[Tuple[Certificate, rsa.RSAPrivateKey]]:
        """Retrieve a certificate and private key."""
        raise NotImplementedError
        
    async def list(self) -> List[str]:
        """List all stored certificate names."""
        raise NotImplementedError
        
    async def delete(self, name: str) -> bool:
        """Delete a stored certificate."""
        raise NotImplementedError
        
    async def exists(self, name: str) -> bool:
        """Check if a certificate exists."""
        raise NotImplementedError


class FileSystemStore(CertificateStore):
    """File system based certificate store."""
    
    def __init__(
        self,
        base_path: Path,
        encrypt_keys: bool = True,
        key_password: Optional[bytes] = None
    ):
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        self.encrypt_keys = encrypt_keys
        self.key_password = key_password or os.urandom(32)
        self._metadata_file = self.base_path / "metadata.json"
        self._load_metadata()
        
    def _load_metadata(self):
        """Load store metadata."""
        if self._metadata_file.exists():
            self._metadata = json.loads(self._metadata_file.read_text())
        else:
            self._metadata = {"certificates": {}, "version": "1.0"}
            self._save_metadata()
            
    def _save_metadata(self):
        """Save store metadata."""
        self._metadata_file.write_text(json.dumps(self._metadata, indent=2, default=str))
        
    async def store(
        self,
        name: str,
        certificate: Certificate,
        private_key: rsa.RSAPrivateKey,
        metadata: Optional[Dict[str, Any]] = None
    ) -> StoredCertificate:
        """Store a certificate and private key."""
        cert_dir = self.base_path / name
        cert_dir.mkdir(parents=True, exist_ok=True)
        
        # Serialize certificate
        cert_pem = certificate.public_bytes(serialization.Encoding.PEM)
        cert_file = cert_dir / "certificate.pem"
        cert_file.write_bytes(cert_pem)
        
        # Serialize private key
        if self.encrypt_keys:
            key_pem = private_key.private_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PrivateFormat.TraditionalOpenSSL,
                encryption_algorithm=serialization.BestAvailableEncryption(self.key_password)
            )
        else:
            key_pem = private_key.private_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PrivateFormat.TraditionalOpenSSL,
                encryption_algorithm=serialization.NoEncryption()
            )
            
        key_file = cert_dir / "private_key.pem"
        key_file.write_bytes(key_pem)
        os.chmod(key_file, 0o600)  # Restrict permissions
        
        # Calculate fingerprint
        fingerprint = certificate.fingerprint(hashes.SHA256()).hex()
        
        # Store metadata
        stored_cert = StoredCertificate(
            name=name,
            certificate_pem=cert_pem,
            private_key_pem=key_pem,
            metadata=metadata or {},
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
            fingerprint=fingerprint
        )
        
        self._metadata["certificates"][name] = {
            "fingerprint": fingerprint,
            "created_at": stored_cert.created_at.isoformat(),
            "updated_at": stored_cert.updated_at.isoformat(),
            "metadata": metadata or {}
        }
        self._save_metadata()
        
        logger.info(f"Stored certificate '{name}' with fingerprint {fingerprint[:16]}...")
        return stored_cert
        
    async def retrieve(self, name: str) -> Optional[Tuple[Certificate, rsa.RSAPrivateKey]]:
        """Retrieve a certificate and private key."""
        cert_dir = self.base_path / name
        if not cert_dir.exists():
            return None
            
        cert_file = cert_dir / "certificate.pem"
        key_file = cert_dir / "private_key.pem"
        
        if not cert_file.exists() or not key_file.exists():
            return None
            
        # Load certificate
        cert_pem = cert_file.read_bytes()
        certificate = x509.load_pem_x509_certificate(cert_pem, default_backend())
        
        # Load private key
        key_pem = key_file.read_bytes()
        if self.encrypt_keys:
            private_key = serialization.load_pem_private_key(
                key_pem,
                password=self.key_password,
                backend=default_backend()
            )
        else:
            private_key = serialization.load_pem_private_key(
                key_pem,
                password=None,
                backend=default_backend()
            )
            
        logger.info(f"Retrieved certificate '{name}'")
        return certificate, private_key
        
    async def list(self) -> List[str]:
        """List all stored certificate names."""
        return list(self._metadata["certificates"].keys())
        
    async def delete(self, name: str) -> bool:
        """Delete a stored certificate."""
        cert_dir = self.base_path / name
        if not cert_dir.exists():
            return False
            
        # Remove files
        for file in cert_dir.iterdir():
            file.unlink()
        cert_dir.rmdir()
        
        # Update metadata
        if name in self._metadata["certificates"]:
            del self._metadata["certificates"][name]
            self._save_metadata()
            
        logger.info(f"Deleted certificate '{name}'")
        return True
        
    async def exists(self, name: str) -> bool:
        """Check if a certificate exists."""
        return name in self._metadata["certificates"]
        
    async def get_metadata(self, name: str) -> Optional[Dict[str, Any]]:
        """Get certificate metadata."""
        if name in self._metadata["certificates"]:
            return self._metadata["certificates"][name]
        return None


class MemoryStore(CertificateStore):
    """In-memory certificate store."""
    
    def __init__(self):
        self._store: Dict[str, StoredCertificate] = {}
        
    async def store(
        self,
        name: str,
        certificate: Certificate,
        private_key: rsa.RSAPrivateKey,
        metadata: Optional[Dict[str, Any]] = None
    ) -> StoredCertificate:
        """Store a certificate and private key."""
        # Serialize to PEM format
        cert_pem = certificate.public_bytes(serialization.Encoding.PEM)
        key_pem = private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.TraditionalOpenSSL,
            encryption_algorithm=serialization.NoEncryption()
        )
        
        # Calculate fingerprint
        fingerprint = certificate.fingerprint(hashes.SHA256()).hex()
        
        stored_cert = StoredCertificate(
            name=name,
            certificate_pem=cert_pem,
            private_key_pem=key_pem,
            metadata=metadata or {},
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
            fingerprint=fingerprint
        )
        
        self._store[name] = stored_cert
        logger.info(f"Stored certificate '{name}' in memory")
        return stored_cert
        
    async def retrieve(self, name: str) -> Optional[Tuple[Certificate, rsa.RSAPrivateKey]]:
        """Retrieve a certificate and private key."""
        if name not in self._store:
            return None
            
        stored = self._store[name]
        
        # Load from PEM
        certificate = x509.load_pem_x509_certificate(
            stored.certificate_pem,
            default_backend()
        )
        private_key = serialization.load_pem_private_key(
            stored.private_key_pem,
            password=None,
            backend=default_backend()
        )
        
        return certificate, private_key
        
    async def list(self) -> List[str]:
        """List all stored certificate names."""
        return list(self._store.keys())
        
    async def delete(self, name: str) -> bool:
        """Delete a stored certificate."""
        if name in self._store:
            del self._store[name]
            logger.info(f"Deleted certificate '{name}' from memory")
            return True
        return False
        
    async def exists(self, name: str) -> bool:
        """Check if a certificate exists."""
        return name in self._store


class SecureFileStore(FileSystemStore):
    """Enhanced file store with additional security features."""
    
    def __init__(
        self,
        base_path: Path,
        master_key: Optional[bytes] = None,
        use_hardware_security: bool = False
    ):
        super().__init__(base_path, encrypt_keys=True)
        self.master_key = master_key or os.urandom(32)
        self.use_hardware_security = use_hardware_security
        self._init_security()
        
    def _init_security(self):
        """Initialize security features."""
        # Set restrictive permissions on base directory
        os.chmod(self.base_path, 0o700)
        
        # Create .htaccess to prevent web access
        htaccess = self.base_path / ".htaccess"
        htaccess.write_text("Deny from all")
        
    def _derive_key(self, salt: bytes, info: bytes) -> bytes:
        """Derive encryption key using PBKDF2."""
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
            backend=default_backend()
        )
        return kdf.derive(self.master_key + info)
        
    def _encrypt_data(self, data: bytes, associated_data: bytes) -> Tuple[bytes, bytes, bytes]:
        """Encrypt data using AES-GCM."""
        salt = os.urandom(16)
        nonce = os.urandom(12)
        
        key = self._derive_key(salt, associated_data)
        cipher = Cipher(
            algorithms.AES(key),
            modes.GCM(nonce),
            backend=default_backend()
        )
        encryptor = cipher.encryptor()
        encryptor.authenticate_additional_data(associated_data)
        
        ciphertext = encryptor.update(data) + encryptor.finalize()
        
        return salt, nonce + encryptor.tag, ciphertext
        
    def _decrypt_data(self, salt: bytes, nonce_tag: bytes, ciphertext: bytes, associated_data: bytes) -> bytes:
        """Decrypt data using AES-GCM."""
        nonce = nonce_tag[:12]
        tag = nonce_tag[12:]
        
        key = self._derive_key(salt, associated_data)
        cipher = Cipher(
            algorithms.AES(key),
            modes.GCM(nonce, tag),
            backend=default_backend()
        )
        decryptor = cipher.decryptor()
        decryptor.authenticate_additional_data(associated_data)
        
        return decryptor.update(ciphertext) + decryptor.finalize()
        
    async def store(
        self,
        name: str,
        certificate: Certificate,
        private_key: rsa.RSAPrivateKey,
        metadata: Optional[Dict[str, Any]] = None
    ) -> StoredCertificate:
        """Store with additional encryption."""
        cert_dir = self.base_path / name
        cert_dir.mkdir(parents=True, exist_ok=True)
        os.chmod(cert_dir, 0o700)
        
        # Store certificate (public, no extra encryption needed)
        cert_pem = certificate.public_bytes(serialization.Encoding.PEM)
        cert_file = cert_dir / "certificate.pem"
        cert_file.write_bytes(cert_pem)
        
        # Encrypt private key with additional layer
        key_pem = private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.TraditionalOpenSSL,
            encryption_algorithm=serialization.NoEncryption()
        )
        
        # Additional encryption
        salt, nonce_tag, ciphertext = self._encrypt_data(
            key_pem,
            name.encode('utf-8')
        )
        
        # Store encrypted components
        key_file = cert_dir / "private_key.enc"
        key_data = {
            "salt": salt.hex(),
            "nonce_tag": nonce_tag.hex(),
            "ciphertext": ciphertext.hex()
        }
        key_file.write_text(json.dumps(key_data))
        os.chmod(key_file, 0o600)
        
        # Create stored certificate record
        fingerprint = certificate.fingerprint(hashes.SHA256()).hex()
        stored_cert = StoredCertificate(
            name=name,
            certificate_pem=cert_pem,
            private_key_pem=b"",  # Not storing plaintext
            metadata=metadata or {},
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
            fingerprint=fingerprint
        )
        
        # Update metadata
        self._metadata["certificates"][name] = {
            "fingerprint": fingerprint,
            "created_at": stored_cert.created_at.isoformat(),
            "updated_at": stored_cert.updated_at.isoformat(),
            "metadata": metadata or {},
            "encryption": "AES-GCM"
        }
        self._save_metadata()
        
        logger.info(f"Securely stored certificate '{name}'")
        return stored_cert
        
    async def retrieve(self, name: str) -> Optional[Tuple[Certificate, rsa.RSAPrivateKey]]:
        """Retrieve with decryption."""
        cert_dir = self.base_path / name
        if not cert_dir.exists():
            return None
            
        # Load certificate
        cert_file = cert_dir / "certificate.pem"
        if not cert_file.exists():
            return None
            
        cert_pem = cert_file.read_bytes()
        certificate = x509.load_pem_x509_certificate(cert_pem, default_backend())
        
        # Load and decrypt private key
        key_file = cert_dir / "private_key.enc"
        if not key_file.exists():
            return None
            
        key_data = json.loads(key_file.read_text())
        
        # Decrypt
        key_pem = self._decrypt_data(
            bytes.fromhex(key_data["salt"]),
            bytes.fromhex(key_data["nonce_tag"]),
            bytes.fromhex(key_data["ciphertext"]),
            name.encode('utf-8')
        )
        
        private_key = serialization.load_pem_private_key(
            key_pem,
            password=None,
            backend=default_backend()
        )
        
        logger.info(f"Retrieved certificate '{name}'")
        return certificate, private_key


class CertificateStoreFactory:
    """Factory for creating certificate stores."""
    
    @staticmethod
    def create(
        store_type: StoreType,
        **kwargs
    ) -> CertificateStore:
        """Create a certificate store."""
        if store_type == StoreType.FILE_SYSTEM:
            return FileSystemStore(**kwargs)
        elif store_type == StoreType.MEMORY:
            return MemoryStore(**kwargs)
        else:
            raise ValueError(f"Unsupported store type: {store_type}")
            
    @staticmethod
    def create_secure_store(base_path: Path, **kwargs) -> SecureFileStore:
        """Create a secure file store."""
        return SecureFileStore(base_path, **kwargs)