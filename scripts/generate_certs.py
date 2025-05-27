#!/usr/bin/env python3
"""Generate TLS certificates for development and testing."""

import argparse
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from chil.security.config import CertificateConfig
from chil.security.certificates import CertificateManager, CertificateAuthority
from chil.security.store import FileSystemStore, StoreType


def main():
    parser = argparse.ArgumentParser(description="Generate TLS certificates")
    parser.add_argument("--type", choices=["self-signed", "ca", "client", "server"], required=True,
                        help="Type of certificate to generate")
    parser.add_argument("--name", required=True, help="Certificate name")
    parser.add_argument("--output-dir", type=Path, default=Path("./certs"),
                        help="Output directory for certificates")
    parser.add_argument("--ca-name", help="CA name (for client/server certs)")
    parser.add_argument("--common-name", help="Certificate common name")
    parser.add_argument("--organization", default="BrahminyKite", help="Organization name")
    parser.add_argument("--country", default="US", help="Country code")
    parser.add_argument("--validity-days", type=int, default=365, help="Certificate validity in days")
    parser.add_argument("--key-size", type=int, default=2048, help="RSA key size")
    parser.add_argument("--san-dns", nargs="+", help="Subject Alternative Names (DNS)")
    parser.add_argument("--san-ip", nargs="+", help="Subject Alternative Names (IP)")
    parser.add_argument("--encrypt-key", action="store_true", help="Encrypt private key")
    parser.add_argument("--key-password", help="Password for key encryption")
    
    args = parser.parse_args()
    
    # Create certificate manager
    cert_manager = CertificateManager()
    
    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create certificate config
    config = CertificateConfig(
        common_name=args.common_name or args.name,
        organization=args.organization,
        country=args.country,
        validity_days=args.validity_days,
        key_size=args.key_size,
        san_dns_names=args.san_dns or [],
        san_ip_addresses=args.san_ip or []
    )
    
    # Generate certificate based on type
    if args.type == "self-signed":
        print(f"Generating self-signed certificate for {args.name}...")
        
        # Generate key and certificate
        private_key = cert_manager.generate_private_key(config.key_size)
        certificate = cert_manager.generate_self_signed_certificate(config, private_key)
        
        # Save to files
        cert_path = args.output_dir / f"{args.name}_cert.pem"
        key_path = args.output_dir / f"{args.name}_key.pem"
        
        cert_manager.save_certificate(certificate, cert_path)
        cert_manager.save_private_key(
            private_key,
            key_path,
            password=args.key_password.encode() if args.key_password else None
        )
        
        print(f"Certificate saved to: {cert_path}")
        print(f"Private key saved to: {key_path}")
        
    elif args.type == "ca":
        print(f"Generating CA certificate for {args.name}...")
        
        # Create CA
        ca = CertificateAuthority(args.name, config)
        ca_cert, ca_key = ca.generate_ca_certificate()
        
        # Save CA
        ca_dir = args.output_dir / "ca" / args.name
        ca_dir.mkdir(parents=True, exist_ok=True)
        
        cert_path = ca_dir / "ca_cert.pem"
        key_path = ca_dir / "ca_key.pem"
        
        cert_manager.save_certificate(ca_cert, cert_path)
        cert_manager.save_private_key(
            ca_key,
            key_path,
            password=args.key_password.encode() if args.key_password else None
        )
        
        print(f"CA certificate saved to: {cert_path}")
        print(f"CA private key saved to: {key_path}")
        
    elif args.type in ["client", "server"]:
        if not args.ca_name:
            print("Error: --ca-name required for client/server certificates")
            sys.exit(1)
            
        print(f"Generating {args.type} certificate for {args.name}...")
        
        # Load CA
        ca_dir = args.output_dir / "ca" / args.ca_name
        ca_cert_path = ca_dir / "ca_cert.pem"
        ca_key_path = ca_dir / "ca_key.pem"
        
        if not ca_cert_path.exists() or not ca_key_path.exists():
            print(f"Error: CA '{args.ca_name}' not found. Generate it first.")
            sys.exit(1)
            
        ca_cert = cert_manager.load_certificate(ca_cert_path)
        ca_key = cert_manager.load_private_key(
            ca_key_path,
            password=args.key_password.encode() if args.key_password else None
        )
        
        # Create CA instance
        ca = CertificateAuthority(args.ca_name, config)
        ca.ca_certificate = ca_cert
        ca.ca_private_key = ca_key
        
        # Generate certificate
        if args.type == "client":
            certificate, private_key = ca.issue_client_certificate(config)
        else:
            certificate, private_key = ca.issue_server_certificate(config)
            
        # Save certificate
        cert_dir = args.output_dir / args.type / args.name
        cert_dir.mkdir(parents=True, exist_ok=True)
        
        cert_path = cert_dir / f"{args.name}_cert.pem"
        key_path = cert_dir / f"{args.name}_key.pem"
        ca_cert_path_copy = cert_dir / "ca_cert.pem"
        
        cert_manager.save_certificate(certificate, cert_path)
        cert_manager.save_private_key(
            private_key,
            key_path,
            password=args.key_password.encode() if args.key_password else None
        )
        
        # Copy CA certificate for verification
        import shutil
        shutil.copy(ca_cert_path, ca_cert_path_copy)
        
        print(f"{args.type.capitalize()} certificate saved to: {cert_path}")
        print(f"Private key saved to: {key_path}")
        print(f"CA certificate copied to: {ca_cert_path_copy}")
        
    # Display certificate info
    print("\nCertificate Information:")
    print(f"  Subject: {config.common_name}")
    print(f"  Organization: {config.organization}")
    print(f"  Validity: {config.validity_days} days")
    print(f"  Key Size: {config.key_size} bits")
    
    if config.san_dns_names:
        print(f"  DNS SANs: {', '.join(config.san_dns_names)}")
    if config.san_ip_addresses:
        print(f"  IP SANs: {', '.join(config.san_ip_addresses)}")


if __name__ == "__main__":
    main()