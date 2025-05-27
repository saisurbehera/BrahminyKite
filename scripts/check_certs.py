#!/usr/bin/env python3
"""Check and validate TLS certificates."""

import argparse
import sys
from pathlib import Path
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from chil.security.certificates import CertificateManager
from chil.security.validation import CertificateValidator


def main():
    parser = argparse.ArgumentParser(description="Check and validate TLS certificates")
    parser.add_argument("cert_path", type=Path, help="Path to certificate file")
    parser.add_argument("--ca-path", type=Path, help="Path to CA certificate for verification")
    parser.add_argument("--hostname", help="Hostname to verify")
    parser.add_argument("--check-expiry", action="store_true", help="Check certificate expiry")
    parser.add_argument("--check-chain", action="store_true", help="Validate certificate chain")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    # Load certificate
    cert_manager = CertificateManager()
    validator = CertificateValidator()
    
    try:
        certificate = cert_manager.load_certificate(args.cert_path)
    except Exception as e:
        print(f"Error loading certificate: {e}")
        sys.exit(1)
        
    # Display certificate info
    print("Certificate Information:")
    print(f"  Subject: {certificate.subject.rfc4514_string()}")
    print(f"  Issuer: {certificate.issuer.rfc4514_string()}")
    print(f"  Serial Number: {certificate.serial_number}")
    print(f"  Not Valid Before: {certificate.not_valid_before}")
    print(f"  Not Valid After: {certificate.not_valid_after}")
    
    # Check expiry
    if args.check_expiry:
        days_until_expiry = (certificate.not_valid_after - datetime.utcnow()).days
        print(f"\nExpiry Check:")
        
        if days_until_expiry < 0:
            print(f"  ❌ Certificate EXPIRED {-days_until_expiry} days ago")
            sys.exit(1)
        elif days_until_expiry < 30:
            print(f"  ⚠️  Certificate expires in {days_until_expiry} days")
        else:
            print(f"  ✅ Certificate valid for {days_until_expiry} more days")
            
    # Validate certificate
    print("\nValidation:")
    
    # Basic validation
    result = validator.validate_certificate(
        certificate,
        check_hostname=args.hostname is not None,
        hostname=args.hostname
    )
    
    if result.is_valid:
        print("  ✅ Certificate validation PASSED")
    else:
        print("  ❌ Certificate validation FAILED")
        for error in result.errors:
            print(f"     - {error}")
            
    # Check against CA if provided
    if args.ca_path:
        print("\nCA Verification:")
        try:
            ca_cert = cert_manager.load_certificate(args.ca_path)
            
            # Verify signature
            from cryptography.x509.verification import PolicyBuilder, VerificationError
            from cryptography.x509.oid import NameOID
            
            try:
                # Build verification policy
                builder = PolicyBuilder().build_server_verifier(
                    trust_store=[ca_cert]
                )
                
                # Verify
                chain = builder.verify(certificate, [])
                print("  ✅ Certificate signed by provided CA")
                
            except VerificationError as e:
                print(f"  ❌ Certificate NOT signed by provided CA: {e}")
                
        except Exception as e:
            print(f"  ❌ Error loading CA certificate: {e}")
            
    # Check hostname if provided
    if args.hostname:
        print(f"\nHostname Verification for '{args.hostname}':")
        
        if validator.validate_hostname(certificate, args.hostname):
            print("  ✅ Hostname matches certificate")
        else:
            print("  ❌ Hostname does NOT match certificate")
            
    # Display SANs
    if args.verbose:
        print("\nSubject Alternative Names:")
        try:
            from cryptography import x509
            san_ext = certificate.extensions.get_extension_for_oid(
                x509.oid.ExtensionOID.SUBJECT_ALTERNATIVE_NAME
            )
            for san in san_ext.value:
                if isinstance(san, x509.DNSName):
                    print(f"  DNS: {san.value}")
                elif isinstance(san, x509.IPAddress):
                    print(f"  IP: {san.value}")
        except x509.ExtensionNotFound:
            print("  None")
            
        # Display key usage
        print("\nKey Usage:")
        try:
            key_usage_ext = certificate.extensions.get_extension_for_oid(
                x509.oid.ExtensionOID.KEY_USAGE
            )
            key_usage = key_usage_ext.value
            
            usages = []
            if key_usage.digital_signature:
                usages.append("Digital Signature")
            if key_usage.key_encipherment:
                usages.append("Key Encipherment")
            if key_usage.key_agreement:
                usages.append("Key Agreement")
            if key_usage.key_cert_sign:
                usages.append("Certificate Sign")
                
            print(f"  {', '.join(usages)}")
        except x509.ExtensionNotFound:
            print("  Not specified")
            
        # Display extended key usage
        print("\nExtended Key Usage:")
        try:
            eku_ext = certificate.extensions.get_extension_for_oid(
                x509.oid.ExtensionOID.EXTENDED_KEY_USAGE
            )
            
            eku_names = {
                x509.oid.ExtensionOID.SERVER_AUTH: "TLS Server Authentication",
                x509.oid.ExtensionOID.CLIENT_AUTH: "TLS Client Authentication",
                x509.oid.ExtensionOID.CODE_SIGNING: "Code Signing",
                x509.oid.ExtensionOID.EMAIL_PROTECTION: "Email Protection"
            }
            
            for usage in eku_ext.value:
                name = eku_names.get(usage, str(usage))
                print(f"  {name}")
        except x509.ExtensionNotFound:
            print("  Not specified")


if __name__ == "__main__":
    main()