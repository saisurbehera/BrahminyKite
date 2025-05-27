#!/usr/bin/env python3
"""
Cache management utility script.
"""

import asyncio
import argparse
import json
import sys
from pathlib import Path
from typing import List, Dict, Any

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from rich.console import Console
from rich.table import Table
from rich.prompt import Prompt, Confirm
from rich.progress import Progress, SpinnerColumn, TextColumn

from chil.caching import (
    RedisConfig, create_redis_client,
    get_cache_invalidator, invalidate_cache
)

console = Console()


async def list_cache_keys(pattern: str = "*", limit: int = 100) -> List[str]:
    """List cache keys matching pattern."""
    client = await create_redis_client()
    
    if pattern == "*":
        keys = await client.scan(count=limit)
    else:
        keys = await client.scan(match=pattern, count=limit)
    
    return keys[:limit]


async def get_key_info(key: str) -> Dict[str, Any]:
    """Get detailed information about a cache key."""
    client = await create_redis_client()
    
    info = {
        "key": key,
        "exists": False
    }
    
    # Check if key exists
    exists = await client.exists(key)
    if not exists:
        return info
    
    info["exists"] = True
    
    # Get TTL
    ttl = await client.ttl(key)
    info["ttl"] = ttl
    
    # Get value (truncated for display)
    try:
        value = await client.get(key)
        if value is not None:
            if isinstance(value, (str, int, float, bool)):
                info["value"] = str(value)[:200]
                info["type"] = type(value).__name__
            else:
                info["value"] = str(value)[:200]
                info["type"] = "object"
            
            # Get size estimate
            import sys
            info["size_bytes"] = sys.getsizeof(value)
        else:
            info["value"] = None
            info["type"] = "unknown"
    except Exception as e:
        info["value"] = f"Error: {e}"
        info["type"] = "error"
    
    return info


async def clear_cache_pattern(pattern: str, confirm: bool = True) -> int:
    """Clear cache keys matching pattern."""
    keys = await list_cache_keys(pattern, limit=10000)
    
    if not keys:
        console.print(f"No keys found matching pattern: {pattern}")
        return 0
    
    console.print(f"Found {len(keys)} keys matching pattern: {pattern}")
    
    if confirm:
        # Show sample keys
        sample_keys = keys[:10]
        table = Table(title="Sample Keys")
        table.add_column("Key", style="cyan")
        
        for key in sample_keys:
            table.add_row(key)
        
        if len(keys) > 10:
            table.add_row("...", style="dim")
            table.add_row(f"({len(keys) - 10} more)", style="dim")
        
        console.print(table)
        
        if not Confirm.ask(f"Delete {len(keys)} keys?"):
            console.print("Cancelled")
            return 0
    
    # Delete keys
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        task = progress.add_task("Deleting keys...", total=len(keys))
        
        client = await create_redis_client()
        deleted = await client.delete(keys)
        
        progress.update(task, completed=len(keys))
    
    console.print(f"Deleted {deleted} keys")
    return deleted


async def cache_stats() -> Dict[str, Any]:
    """Get cache statistics."""
    client = await create_redis_client()
    
    # Get Redis info
    info = await client.client.info()
    memory_info = await client.client.info("memory")
    stats_info = await client.client.info("stats")
    
    # Count keys by pattern
    all_keys = await client.scan(count=10000)
    
    # Categorize keys
    categories = {}
    for key in all_keys:
        parts = key.split(":")
        category = parts[0] if len(parts) > 1 else "uncategorized"
        categories[category] = categories.get(category, 0) + 1
    
    return {
        "server": {
            "version": info.get("redis_version"),
            "uptime_days": info.get("uptime_in_days"),
            "connected_clients": info.get("connected_clients")
        },
        "memory": {
            "used_memory_human": memory_info.get("used_memory_human"),
            "used_memory_peak_human": memory_info.get("used_memory_peak_human"),
            "mem_fragmentation_ratio": memory_info.get("mem_fragmentation_ratio")
        },
        "stats": {
            "total_commands_processed": stats_info.get("total_commands_processed"),
            "keyspace_hits": stats_info.get("keyspace_hits"),
            "keyspace_misses": stats_info.get("keyspace_misses"),
            "ops_per_sec": stats_info.get("instantaneous_ops_per_sec")
        },
        "keys": {
            "total_count": len(all_keys),
            "categories": categories
        }
    }


async def warm_cache(keys_file: str) -> int:
    """Warm cache with predefined keys."""
    try:
        with open(keys_file, 'r') as f:
            warm_data = json.load(f)
    except Exception as e:
        console.print(f"[red]Error reading keys file: {e}[/red]")
        return 0
    
    client = await create_redis_client()
    warmed = 0
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        task = progress.add_task("Warming cache...", total=len(warm_data))
        
        for item in warm_data:
            try:
                key = item["key"]
                value = item["value"]
                ttl = item.get("ttl", 3600)
                
                await client.set(key, value, ttl=ttl)
                warmed += 1
                
                progress.update(task, advance=1)
            except Exception as e:
                console.print(f"[red]Error warming key {item.get('key', 'unknown')}: {e}[/red]")
    
    console.print(f"Warmed {warmed} cache entries")
    return warmed


def display_stats(stats: Dict[str, Any]):
    """Display cache statistics."""
    # Server info
    server_table = Table(title="Server Information")
    server_table.add_column("Metric", style="cyan")
    server_table.add_column("Value", style="white")
    
    server_info = stats["server"]
    server_table.add_row("Version", server_info.get("version", "Unknown"))
    server_table.add_row("Uptime", f"{server_info.get('uptime_days', 0)} days")
    server_table.add_row("Connected Clients", str(server_info.get("connected_clients", 0)))
    
    # Memory info
    memory_table = Table(title="Memory Usage")
    memory_table.add_column("Metric", style="cyan")
    memory_table.add_column("Value", style="white")
    
    memory_info = stats["memory"]
    memory_table.add_row("Used Memory", memory_info.get("used_memory_human", "Unknown"))
    memory_table.add_row("Peak Memory", memory_info.get("used_memory_peak_human", "Unknown"))
    memory_table.add_row("Fragmentation", f"{memory_info.get('mem_fragmentation_ratio', 0):.2f}")
    
    # Performance stats
    perf_table = Table(title="Performance Statistics")
    perf_table.add_column("Metric", style="cyan")
    perf_table.add_column("Value", style="white")
    
    stats_info = stats["stats"]
    hits = stats_info.get("keyspace_hits", 0)
    misses = stats_info.get("keyspace_misses", 0)
    total_requests = hits + misses
    hit_rate = (hits / total_requests * 100) if total_requests > 0 else 0
    
    perf_table.add_row("Total Commands", str(stats_info.get("total_commands_processed", 0)))
    perf_table.add_row("Hit Rate", f"{hit_rate:.2f}%")
    perf_table.add_row("Ops/sec", str(stats_info.get("ops_per_sec", 0)))
    
    # Key categories
    keys_table = Table(title="Key Categories")
    keys_table.add_column("Category", style="cyan")
    keys_table.add_column("Count", style="white")
    
    keys_info = stats["keys"]
    keys_table.add_row("Total Keys", str(keys_info["total_count"]))
    
    for category, count in sorted(keys_info["categories"].items()):
        keys_table.add_row(category, str(count))
    
    console.print(server_table)
    console.print(memory_table)
    console.print(perf_table)
    console.print(keys_table)


async def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Cache management utility")
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # List keys
    list_parser = subparsers.add_parser("list", help="List cache keys")
    list_parser.add_argument("--pattern", default="*", help="Key pattern to match")
    list_parser.add_argument("--limit", type=int, default=100, help="Maximum keys to list")
    
    # Get key info
    info_parser = subparsers.add_parser("info", help="Get key information")
    info_parser.add_argument("key", help="Key to inspect")
    
    # Clear cache
    clear_parser = subparsers.add_parser("clear", help="Clear cache keys")
    clear_parser.add_argument("pattern", help="Key pattern to clear")
    clear_parser.add_argument("--force", action="store_true", help="Skip confirmation")
    
    # Invalidate cache
    invalidate_parser = subparsers.add_parser("invalidate", help="Invalidate cache entries")
    invalidate_parser.add_argument("--keys", nargs="+", help="Specific keys to invalidate")
    invalidate_parser.add_argument("--patterns", nargs="+", help="Patterns to invalidate")
    invalidate_parser.add_argument("--tags", nargs="+", help="Tags to invalidate")
    
    # Cache statistics
    subparsers.add_parser("stats", help="Show cache statistics")
    
    # Warm cache
    warm_parser = subparsers.add_parser("warm", help="Warm cache from file")
    warm_parser.add_argument("file", help="JSON file with key-value pairs")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    try:
        if args.command == "list":
            keys = await list_cache_keys(args.pattern, args.limit)
            
            if not keys:
                console.print(f"No keys found matching pattern: {args.pattern}")
            else:
                table = Table(title=f"Cache Keys (pattern: {args.pattern})")
                table.add_column("Key", style="cyan")
                table.add_column("TTL", style="white")
                
                client = await create_redis_client()
                for key in keys:
                    ttl = await client.ttl(key)
                    ttl_str = f"{ttl}s" if ttl > 0 else "No expiry" if ttl == -1 else "Expired"
                    table.add_row(key, ttl_str)
                
                console.print(table)
        
        elif args.command == "info":
            info = await get_key_info(args.key)
            
            if not info["exists"]:
                console.print(f"[red]Key not found: {args.key}[/red]")
                return 1
            
            table = Table(title=f"Key Information: {args.key}")
            table.add_column("Property", style="cyan")
            table.add_column("Value", style="white")
            
            table.add_row("Exists", "Yes")
            table.add_row("Type", info["type"])
            table.add_row("TTL", f"{info['ttl']}s" if info['ttl'] > 0 else "No expiry" if info['ttl'] == -1 else "Expired")
            table.add_row("Size", f"{info.get('size_bytes', 0)} bytes")
            table.add_row("Value", info["value"])
            
            console.print(table)
        
        elif args.command == "clear":
            deleted = await clear_cache_pattern(args.pattern, not args.force)
            console.print(f"[green]Cleared {deleted} keys[/green]")
        
        elif args.command == "invalidate":
            count = await invalidate_cache(
                keys=args.keys,
                patterns=args.patterns,
                tags=args.tags
            )
            console.print(f"[green]Invalidated {count} cache entries[/green]")
        
        elif args.command == "stats":
            stats = await cache_stats()
            display_stats(stats)
        
        elif args.command == "warm":
            warmed = await warm_cache(args.file)
            console.print(f"[green]Warmed {warmed} cache entries[/green]")
        
        return 0
    
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        return 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))