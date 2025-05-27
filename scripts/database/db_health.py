#!/usr/bin/env python3
"""
Database health check and monitoring script.

This script provides real-time database health monitoring and diagnostics.
"""

import asyncio
import argparse
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any
import json

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import asyncpg
from rich.console import Console
from rich.table import Table
from rich.live import Live
from rich.panel import Panel
from rich.layout import Layout
from rich.progress import Progress, SpinnerColumn, TextColumn

from chil.persistence.core.config import DatabaseConfig
from chil.persistence.core.database import create_database_engine

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
console = Console()


async def check_connection_health(config: DatabaseConfig) -> Dict[str, Any]:
    """Check database connection health."""
    try:
        start_time = datetime.now()
        
        conn = await asyncpg.connect(
            host=config.host,
            port=config.port,
            user=config.username,
            password=config.password,
            database=config.database,
            timeout=5
        )
        
        connection_time = (datetime.now() - start_time).total_seconds()
        
        # Test query
        query_start = datetime.now()
        await conn.fetchval("SELECT 1")
        query_time = (datetime.now() - query_start).total_seconds()
        
        await conn.close()
        
        return {
            "status": "healthy",
            "connection_time": connection_time,
            "query_time": query_time
        }
    
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e)
        }


async def get_database_stats(config: DatabaseConfig) -> Dict[str, Any]:
    """Get comprehensive database statistics."""
    conn = await asyncpg.connect(
        host=config.host,
        port=config.port,
        user=config.username,
        password=config.password,
        database=config.database
    )
    
    try:
        stats = {}
        
        # Database size
        db_size = await conn.fetchval("""
            SELECT pg_database_size(current_database())
        """)
        stats["database_size_mb"] = db_size / 1024 / 1024
        
        # Table count and sizes
        tables = await conn.fetch("""
            SELECT 
                schemaname,
                tablename,
                pg_total_relation_size(schemaname||'.'||tablename) as total_size,
                n_live_tup as row_count
            FROM pg_stat_user_tables
            ORDER BY total_size DESC
        """)
        
        stats["table_count"] = len(tables)
        stats["tables"] = [
            {
                "name": f"{t['schemaname']}.{t['tablename']}",
                "size_mb": t["total_size"] / 1024 / 1024,
                "row_count": t["row_count"]
            }
            for t in tables[:10]  # Top 10 tables
        ]
        
        # Connection stats
        connections = await conn.fetch("""
            SELECT 
                state,
                COUNT(*) as count
            FROM pg_stat_activity
            WHERE datname = current_database()
            GROUP BY state
        """)
        
        stats["connections"] = {
            c["state"] or "idle": c["count"]
            for c in connections
        }
        
        # Active queries
        active_queries = await conn.fetch("""
            SELECT 
                pid,
                usename,
                state,
                query,
                EXTRACT(EPOCH FROM (now() - query_start)) as duration
            FROM pg_stat_activity
            WHERE state != 'idle'
                AND datname = current_database()
            ORDER BY duration DESC
            LIMIT 5
        """)
        
        stats["active_queries"] = [
            {
                "pid": q["pid"],
                "user": q["usename"],
                "state": q["state"],
                "duration": q["duration"],
                "query": q["query"][:100] + "..." if len(q["query"]) > 100 else q["query"]
            }
            for q in active_queries
        ]
        
        # Lock statistics
        locks = await conn.fetch("""
            SELECT 
                mode,
                COUNT(*) as count
            FROM pg_locks
            WHERE database = (SELECT oid FROM pg_database WHERE datname = current_database())
            GROUP BY mode
        """)
        
        stats["locks"] = {
            l["mode"]: l["count"]
            for l in locks
        }
        
        # Cache hit ratio
        cache_stats = await conn.fetchrow("""
            SELECT 
                sum(heap_blks_hit) / NULLIF(sum(heap_blks_hit) + sum(heap_blks_read), 0) as cache_hit_ratio
            FROM pg_statio_user_tables
        """)
        
        stats["cache_hit_ratio"] = float(cache_stats["cache_hit_ratio"] or 0)
        
        # Index usage
        index_stats = await conn.fetchrow("""
            SELECT 
                sum(idx_scan) / NULLIF(sum(seq_scan + idx_scan), 0) as index_usage_ratio
            FROM pg_stat_user_tables
        """)
        
        stats["index_usage_ratio"] = float(index_stats["index_usage_ratio"] or 0)
        
        return stats
    
    finally:
        await conn.close()


async def check_replication_status(config: DatabaseConfig) -> Dict[str, Any]:
    """Check replication status if configured."""
    conn = await asyncpg.connect(
        host=config.host,
        port=config.port,
        user=config.username,
        password=config.password,
        database=config.database
    )
    
    try:
        # Check if this is a primary or replica
        is_replica = await conn.fetchval("SELECT pg_is_in_recovery()")
        
        if is_replica:
            # Get replication lag
            lag = await conn.fetchrow("""
                SELECT 
                    EXTRACT(EPOCH FROM (now() - pg_last_xact_replay_timestamp())) as lag_seconds,
                    pg_last_wal_receive_lsn() as receive_lsn,
                    pg_last_wal_replay_lsn() as replay_lsn
            """)
            
            return {
                "role": "replica",
                "lag_seconds": lag["lag_seconds"],
                "receive_lsn": str(lag["receive_lsn"]),
                "replay_lsn": str(lag["replay_lsn"])
            }
        else:
            # Get replication slots
            slots = await conn.fetch("""
                SELECT 
                    slot_name,
                    active,
                    restart_lsn
                FROM pg_replication_slots
            """)
            
            return {
                "role": "primary",
                "replication_slots": [
                    {
                        "name": s["slot_name"],
                        "active": s["active"],
                        "restart_lsn": str(s["restart_lsn"])
                    }
                    for s in slots
                ]
            }
    
    finally:
        await conn.close()


def create_health_dashboard(health_data: Dict[str, Any]) -> Layout:
    """Create a rich dashboard layout for health data."""
    layout = Layout()
    
    # Connection health panel
    conn_health = health_data.get("connection_health", {})
    if conn_health.get("status") == "healthy":
        conn_status = f"[green]✓ Healthy[/green]\nConnection: {conn_health['connection_time']:.3f}s\nQuery: {conn_health['query_time']:.3f}s"
    else:
        conn_status = f"[red]✗ Unhealthy[/red]\n{conn_health.get('error', 'Unknown error')}"
    
    # Database stats
    stats = health_data.get("database_stats", {})
    
    # Create tables table
    table = Table(title="Top Tables by Size")
    table.add_column("Table", style="cyan")
    table.add_column("Size (MB)", justify="right")
    table.add_column("Rows", justify="right")
    
    for t in stats.get("tables", [])[:5]:
        table.add_row(
            t["name"],
            f"{t['size_mb']:.2f}",
            f"{t['row_count']:,}"
        )
    
    # Create metrics panel
    metrics_text = f"""
Database Size: {stats.get('database_size_mb', 0):.2f} MB
Table Count: {stats.get('table_count', 0)}

Cache Hit Ratio: {stats.get('cache_hit_ratio', 0):.2%}
Index Usage Ratio: {stats.get('index_usage_ratio', 0):.2%}

Connections:
{json.dumps(stats.get('connections', {}), indent=2)}

Locks:
{json.dumps(stats.get('locks', {}), indent=2)}
"""
    
    # Layout
    layout.split_column(
        Layout(Panel(conn_status, title="Connection Health"), size=5),
        Layout(Panel(metrics_text, title="Database Metrics"), size=15),
        Layout(table, size=10)
    )
    
    return layout


async def monitor_loop(config: DatabaseConfig, interval: int = 5):
    """Continuous monitoring loop."""
    with Live(auto_refresh=False) as live:
        while True:
            try:
                # Gather health data
                health_data = {
                    "timestamp": datetime.now().isoformat(),
                    "connection_health": await check_connection_health(config),
                    "database_stats": await get_database_stats(config)
                }
                
                # Try to get replication status
                try:
                    health_data["replication"] = await check_replication_status(config)
                except:
                    pass
                
                # Update display
                dashboard = create_health_dashboard(health_data)
                live.update(dashboard)
                live.refresh()
                
                await asyncio.sleep(interval)
            
            except KeyboardInterrupt:
                break
            except Exception as e:
                console.print(f"[red]Error: {e}[/red]")
                await asyncio.sleep(interval)


async def run_diagnostics(config: DatabaseConfig):
    """Run comprehensive diagnostics."""
    console.print("[bold]Running Database Diagnostics...[/bold]\n")
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        
        # Connection test
        task = progress.add_task("Testing connection...", total=1)
        conn_health = await check_connection_health(config)
        progress.update(task, completed=1)
        
        if conn_health["status"] == "healthy":
            console.print(f"✓ Connection: [green]Healthy[/green] ({conn_health['connection_time']:.3f}s)")
        else:
            console.print(f"✗ Connection: [red]Failed[/red] - {conn_health.get('error')}")
            return
        
        # Database stats
        task = progress.add_task("Gathering statistics...", total=1)
        stats = await get_database_stats(config)
        progress.update(task, completed=1)
        
        # Display results
        console.print(f"\n✓ Database Size: {stats['database_size_mb']:.2f} MB")
        console.print(f"✓ Tables: {stats['table_count']}")
        console.print(f"✓ Cache Hit Ratio: {stats['cache_hit_ratio']:.2%}")
        console.print(f"✓ Index Usage: {stats['index_usage_ratio']:.2%}")
        
        # Check for issues
        console.print("\n[bold]Potential Issues:[/bold]")
        issues_found = False
        
        if stats['cache_hit_ratio'] < 0.9:
            console.print("⚠️  Low cache hit ratio - consider increasing shared_buffers")
            issues_found = True
        
        if stats['index_usage_ratio'] < 0.8:
            console.print("⚠️  Low index usage - review query patterns and indexes")
            issues_found = True
        
        if stats.get('connections', {}).get('active', 0) > 50:
            console.print("⚠️  High number of active connections")
            issues_found = True
        
        if not issues_found:
            console.print("[green]✓ No issues detected[/green]")


async def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Database health monitoring")
    parser.add_argument(
        "--monitor",
        action="store_true",
        help="Run continuous monitoring"
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=5,
        help="Monitoring interval in seconds (default: 5)"
    )
    parser.add_argument(
        "--diagnostics",
        action="store_true",
        help="Run one-time diagnostics"
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output results as JSON"
    )
    
    args = parser.parse_args()
    
    # Load database configuration
    config = DatabaseConfig.from_env()
    
    try:
        if args.monitor:
            console.print(f"[bold]Monitoring database: {config.database}[/bold]")
            console.print("Press Ctrl+C to stop\n")
            await monitor_loop(config, args.interval)
        
        elif args.diagnostics:
            await run_diagnostics(config)
        
        else:
            # One-time health check
            health_data = {
                "timestamp": datetime.now().isoformat(),
                "database": config.database,
                "connection_health": await check_connection_health(config),
                "database_stats": await get_database_stats(config)
            }
            
            try:
                health_data["replication"] = await check_replication_status(config)
            except:
                pass
            
            if args.json:
                print(json.dumps(health_data, indent=2))
            else:
                dashboard = create_health_dashboard(health_data)
                console.print(dashboard)
        
        return 0
    
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        return 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))