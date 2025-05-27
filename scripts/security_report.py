#!/usr/bin/env python3
"""Generate security reports from audit logs."""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime, timedelta
from collections import defaultdict

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def parse_audit_logs(log_dir: Path, start_date: datetime, end_date: datetime):
    """Parse audit logs in date range."""
    events = []
    
    # Find log files in date range
    for log_file in log_dir.glob("security_audit_*.jsonl"):
        # Extract date from filename
        try:
            file_date_str = log_file.stem.replace("security_audit_", "")
            file_date = datetime.strptime(file_date_str, "%Y%m%d")
            
            # Check if file is in date range
            if start_date.date() <= file_date.date() <= end_date.date():
                with open(log_file, 'r') as f:
                    for line in f:
                        if line.strip():
                            event = json.loads(line)
                            event_time = datetime.fromisoformat(event['timestamp'])
                            
                            # Check if event is in time range
                            if start_date <= event_time <= end_date:
                                events.append(event)
        except Exception as e:
            print(f"Warning: Error parsing {log_file}: {e}")
            
    return events


def generate_report(events):
    """Generate security report from events."""
    report = {
        "summary": {
            "total_events": len(events),
            "event_types": defaultdict(int),
            "severity_levels": defaultdict(int),
            "top_sources": defaultdict(int),
            "hourly_distribution": defaultdict(int)
        },
        "authentication": {
            "total_attempts": 0,
            "successful": 0,
            "failed": 0,
            "failure_reasons": defaultdict(int)
        },
        "certificates": {
            "validations": 0,
            "validation_failures": 0,
            "renewals": 0,
            "expirations": 0
        },
        "tls": {
            "handshakes": 0,
            "handshake_failures": 0,
            "protocol_versions": defaultdict(int),
            "cipher_suites": defaultdict(int)
        },
        "security_issues": {
            "suspicious_activities": 0,
            "rate_limit_violations": 0,
            "weak_ciphers": 0,
            "access_denials": 0
        }
    }
    
    # Process events
    for event in events:
        event_type = event.get('event_type', 'unknown')
        severity = event.get('severity', 'unknown')
        
        # Update summary
        report['summary']['event_types'][event_type] += 1
        report['summary']['severity_levels'][severity] += 1
        
        # Track source IPs
        if 'source_ip' in event and event['source_ip']:
            report['summary']['top_sources'][event['source_ip']] += 1
            
        # Hourly distribution
        event_time = datetime.fromisoformat(event['timestamp'])
        hour = event_time.hour
        report['summary']['hourly_distribution'][hour] += 1
        
        # Process specific event types
        details = event.get('details', {})
        
        if event_type == 'auth_success':
            report['authentication']['total_attempts'] += 1
            report['authentication']['successful'] += 1
            
        elif event_type == 'auth_failure':
            report['authentication']['total_attempts'] += 1
            report['authentication']['failed'] += 1
            reason = details.get('reason', 'unknown')
            report['authentication']['failure_reasons'][reason] += 1
            
        elif event_type == 'cert_validation_success':
            report['certificates']['validations'] += 1
            
        elif event_type == 'cert_validation_failure':
            report['certificates']['validation_failures'] += 1
            
        elif event_type == 'cert_renewed':
            report['certificates']['renewals'] += 1
            
        elif event_type == 'cert_expired':
            report['certificates']['expirations'] += 1
            
        elif event_type == 'tls_handshake_success':
            report['tls']['handshakes'] += 1
            if 'tls_version' in details:
                report['tls']['protocol_versions'][details['tls_version']] += 1
            if 'cipher_suite' in details:
                report['tls']['cipher_suites'][details['cipher_suite']] += 1
                
        elif event_type == 'tls_handshake_failure':
            report['tls']['handshake_failures'] += 1
            
        elif event_type == 'suspicious_activity':
            report['security_issues']['suspicious_activities'] += 1
            
        elif event_type == 'rate_limit_exceeded':
            report['security_issues']['rate_limit_violations'] += 1
            
        elif event_type == 'cipher_suite_weak':
            report['security_issues']['weak_ciphers'] += 1
            
        elif event_type == 'access_denied':
            report['security_issues']['access_denials'] += 1
            
    # Sort top sources
    report['summary']['top_sources'] = dict(
        sorted(report['summary']['top_sources'].items(), 
               key=lambda x: x[1], reverse=True)[:10]
    )
    
    return report


def print_report(report, start_date, end_date, output_format):
    """Print the security report."""
    if output_format == 'json':
        print(json.dumps(report, indent=2))
        return
        
    # Text format
    print("=" * 80)
    print(f"SECURITY REPORT")
    print(f"Period: {start_date.strftime('%Y-%m-%d %H:%M')} to {end_date.strftime('%Y-%m-%d %H:%M')}")
    print("=" * 80)
    
    # Summary
    print("\nEVENT SUMMARY:")
    print(f"  Total Events: {report['summary']['total_events']}")
    print("\n  Event Types:")
    for event_type, count in sorted(report['summary']['event_types'].items(), 
                                   key=lambda x: x[1], reverse=True):
        print(f"    {event_type}: {count}")
        
    print("\n  Severity Levels:")
    for severity, count in report['summary']['severity_levels'].items():
        print(f"    {severity}: {count}")
        
    # Authentication
    total_auth = report['authentication']['total_attempts']
    if total_auth > 0:
        success_rate = (report['authentication']['successful'] / total_auth) * 100
        print(f"\nAUTHENTICATION:")
        print(f"  Total Attempts: {total_auth}")
        print(f"  Successful: {report['authentication']['successful']} ({success_rate:.1f}%)")
        print(f"  Failed: {report['authentication']['failed']} ({100-success_rate:.1f}%)")
        
        if report['authentication']['failure_reasons']:
            print("  Failure Reasons:")
            for reason, count in report['authentication']['failure_reasons'].items():
                print(f"    {reason}: {count}")
                
    # Certificates
    print(f"\nCERTIFICATES:")
    print(f"  Validations: {report['certificates']['validations']}")
    print(f"  Validation Failures: {report['certificates']['validation_failures']}")
    print(f"  Renewals: {report['certificates']['renewals']}")
    print(f"  Expirations: {report['certificates']['expirations']}")
    
    # TLS
    print(f"\nTLS:")
    print(f"  Handshakes: {report['tls']['handshakes']}")
    print(f"  Handshake Failures: {report['tls']['handshake_failures']}")
    
    if report['tls']['protocol_versions']:
        print("  Protocol Versions:")
        for version, count in sorted(report['tls']['protocol_versions'].items()):
            print(f"    {version}: {count}")
            
    # Security Issues
    issues = report['security_issues']
    total_issues = sum(issues.values())
    if total_issues > 0:
        print(f"\nSECURITY ISSUES:")
        print(f"  Total Issues: {total_issues}")
        if issues['suspicious_activities'] > 0:
            print(f"  Suspicious Activities: {issues['suspicious_activities']}")
        if issues['rate_limit_violations'] > 0:
            print(f"  Rate Limit Violations: {issues['rate_limit_violations']}")
        if issues['weak_ciphers'] > 0:
            print(f"  Weak Cipher Usage: {issues['weak_ciphers']}")
        if issues['access_denials'] > 0:
            print(f"  Access Denials: {issues['access_denials']}")
            
    # Top Sources
    if report['summary']['top_sources']:
        print(f"\nTOP SOURCE IPS:")
        for ip, count in list(report['summary']['top_sources'].items())[:10]:
            print(f"  {ip}: {count} events")
            
    # Hourly Distribution
    print(f"\nHOURLY DISTRIBUTION:")
    for hour in range(24):
        count = report['summary']['hourly_distribution'].get(hour, 0)
        bar = '█' * (count // 10) if count > 0 else ''
        print(f"  {hour:02d}:00 [{count:4d}] {bar}")
        
    print("\n" + "=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Generate security reports from audit logs")
    parser.add_argument("--log-dir", type=Path, default=Path("./logs/security"),
                        help="Directory containing audit logs")
    parser.add_argument("--days", type=int, default=1,
                        help="Number of days to include in report")
    parser.add_argument("--start-date", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end-date", help="End date (YYYY-MM-DD)")
    parser.add_argument("--format", choices=["text", "json"], default="text",
                        help="Output format")
    parser.add_argument("--output", type=Path, help="Output file (default: stdout)")
    
    args = parser.parse_args()
    
    # Determine date range
    if args.start_date and args.end_date:
        start_date = datetime.strptime(args.start_date, "%Y-%m-%d")
        end_date = datetime.strptime(args.end_date + " 23:59:59", "%Y-%m-%d %H:%M:%S")
    else:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=args.days)
        
    # Check if log directory exists
    if not args.log_dir.exists():
        print(f"Error: Log directory '{args.log_dir}' does not exist")
        sys.exit(1)
        
    # Parse logs
    print(f"Parsing logs from {start_date} to {end_date}...", file=sys.stderr)
    events = parse_audit_logs(args.log_dir, start_date, end_date)
    
    if not events:
        print("No events found in the specified time range")
        sys.exit(0)
        
    # Generate report
    report = generate_report(events)
    
    # Output report
    if args.output:
        with open(args.output, 'w') as f:
            if args.format == 'json':
                json.dump(report, f, indent=2)
            else:
                # Redirect stdout temporarily
                old_stdout = sys.stdout
                sys.stdout = f
                print_report(report, start_date, end_date, args.format)
                sys.stdout = old_stdout
        print(f"Report saved to: {args.output}")
    else:
        print_report(report, start_date, end_date, args.format)


if __name__ == "__main__":
    main()