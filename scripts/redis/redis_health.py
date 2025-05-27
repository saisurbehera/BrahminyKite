#!/usr/bin/env python3
"""
Redis health monitoring and diagnostics script.
"""

import asyncio
import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Any

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.layout import Layout
from rich.live import Live

from chil.caching import RedisConfig, create_redis_client
from chil.caching.monitoring import CacheHealthCheck

console = Console()


async def check_redis_health(config: RedisConfig) -> Dict[str, Any]:
    """Perform comprehensive Redis health check."""
    try:
        client = await create_redis_client(config)
        health_checker = CacheHealthCheck(client)
        return await health_checker.check_health()
    except Exception as e:
        return {
            "healthy": False,
            "error": str(e),
            "checks": {}
        }


async def get_redis_info(config: RedisConfig) -> Dict[str, Any]:
    """Get detailed Redis information."""
    try:
        client = await create_redis_client(config)
        
        # Get various info sections
        info = {}
        sections = ["server", "memory", "stats", "replication", "clients", "persistence"]
        
        for section in sections:
            try:
                info[section] = await client.client.info(section)
            except:
                info[section] = {}
        
        # Get additional metrics
        info["keyspace"] = await client.client.info("keyspace")
        
        return info
    except Exception as e:
        return {"error": str(e)}


def create_health_display(health_data: Dict[str, Any]) -> Layout:
    """Create a rich display for health data."""
    layout = Layout()
    
    # Overall status
    status = "🟢 HEALTHY" if health_data.get("healthy") else "🔴 UNHEALTHY"
    status_panel = Panel(f"[bold]{status}[/bold]", title="Redis Status")
    
    # Checks table
    table = Table(title="Health Checks")
    table.add_column("Check", style="cyan")
    table.add_column("Status", style="white")
    table.add_column("Details", style="dim")
    
    for check_name, check_data in health_data.get("checks", {}).items():
        status = check_data.get("status", "unknown")
        status_icon = {
            "pass": "✅",
            "warn": "⚠️", 
            "fail": "❌"
        }.get(status, "❓")
        
        details = []
        if "response_time_ms" in check_data:
            details.append(f"Response: {check_data['response_time_ms']:.2f}ms")
        if "usage_percent" in check_data:
            details.append(f"Usage: {check_data['usage_percent']:.1f}%")
        if "count" in check_data:
            details.append(f"Count: {check_data['count']}")
        if "error" in check_data:
            details.append(f"Error: {check_data['error']}")
        
        table.add_row(
            check_name.replace("_", " ").title(),
            f"{status_icon} {status.upper()}",
            " | ".join(details)
        )
    
    layout.split_column(
        Layout(status_panel, size=3),
        Layout(table)
    )
    
    return layout


def create_info_display(info_data: Dict[str, Any]) -> Layout:
    """Create display for Redis info."""
    layout = Layout()
    
    if "error" in info_data:
        error_panel = Panel(f"[red]Error: {info_data['error']}[/red]", title="Error")
        layout.add_split(error_panel)
        return layout
    
    # Server info
    server_info = info_data.get("server", {})
    server_text = f"""
Version: {server_info.get('redis_version', 'Unknown')}
Mode: {server_info.get('redis_mode', 'Unknown')}
OS: {server_info.get('os', 'Unknown')}
Arch: {server_info.get('arch_bits', 'Unknown')} bit
Uptime: {server_info.get('uptime_in_days', 0)} days
"""
    
    # Memory info
    memory_info = info_data.get("memory", {})
    used_memory = memory_info.get("used_memory_human", "0B")
    peak_memory = memory_info.get("used_memory_peak_human", "0B")
    memory_text = f"""
Used Memory: {used_memory}
Peak Memory: {peak_memory}
Fragmentation: {memory_info.get('mem_fragmentation_ratio', 0):.2f}
"""
    
    # Client info
    clients_info = info_data.get("clients", {})
    stats_info = info_data.get("stats", {})
    client_text = f"""
Connected: {clients_info.get('connected_clients', 0)}
Blocked: {clients_info.get('blocked_clients', 0)}
Total Commands: {stats_info.get('total_commands_processed', 0)}
Commands/sec: {stats_info.get('instantaneous_ops_per_sec', 0)}
"""
    
    # Keyspace info
    keyspace_info = info_data.get("keyspace", {})
    keyspace_text = "No databases\n"
    if keyspace_info:
        db_lines = []
        for db, stats in keyspace_info.items():
            if isinstance(stats, dict):
                keys = stats.get('keys', 0)
                expires = stats.get('expires', 0)
                db_lines.append(f"{db}: {keys} keys, {expires} expires")
        if db_lines:
            keyspace_text = "\n".join(db_lines)
    
    layout.split(
        Layout(Panel(server_text.strip(), title="Server"), name="server"),
        Layout(Panel(memory_text.strip(), title="Memory"), name="memory")
    )
    
    layout["server"].split_row(
        Layout(Panel(client_text.strip(), title="Clients")),
        Layout(Panel(keyspace_text.strip(), title="Keyspace"))
    )
    
    return layout


async def monitor_redis(config: RedisConfig, interval: int = 5):
    """Continuously monitor Redis."""
    with Live(auto_refresh=False) as live:
        while True:
            try:
                health_data = await check_redis_health(config)
                display = create_health_display(health_data)
                live.update(display)
                live.refresh()
                
                await asyncio.sleep(interval)
            except KeyboardInterrupt:
                break
            except Exception as e:
                console.print(f"[red]Error: {e}[/red]")
                await asyncio.sleep(interval)


async def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Redis health monitoring")
    parser.add_argument(
        "--monitor",
        action="store_true",
        help="Continuous monitoring mode"
    )
    parser.add_argument(
        "--info",
        action="store_true",
        help="Show detailed Redis information"
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=5,
        help="Monitoring interval in seconds (default: 5)"
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output as JSON"
    )
    
    args = parser.parse_args()
    
    # Load Redis configuration
    config = RedisConfig.from_env()
    
    try:
        if args.monitor:
            console.print(f"[bold]Monitoring Redis: {config.host}:{config.port}[/bold]")
            console.print("Press Ctrl+C to stop\n")
            await monitor_redis(config, args.interval)
        
        elif args.info:
            info_data = await get_redis_info(config)
            
            if args.json:
                print(json.dumps(info_data, indent=2, default=str))
            else:
                display = create_info_display(info_data)
                console.print(display)
        
        else:
            # One-time health check
            health_data = await check_redis_health(config)
            
            if args.json:
                print(json.dumps(health_data, indent=2))
            else:
                display = create_health_display(health_data)
                console.print(display)
        
        return 0
    
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        return 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))