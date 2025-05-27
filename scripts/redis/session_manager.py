#!/usr/bin/env python3
"""
Session management utility script.
"""

import asyncio
import argparse
import json
import sys
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.prompt import Confirm

from chil.caching.session import RedisSessionManager, Session

console = Console()


async def list_sessions(
    user_id: str = None,
    limit: int = 50,
    include_expired: bool = False
) -> List[Session]:
    """List active sessions."""
    session_manager = RedisSessionManager()
    await session_manager.initialize()
    
    sessions = await session_manager.list_active_sessions(user_id=user_id, limit=limit)
    
    if not include_expired:
        sessions = [s for s in sessions if not s.is_expired()]
    
    return sessions


async def get_session_details(session_id: str) -> Dict[str, Any]:
    """Get detailed session information."""
    session_manager = RedisSessionManager()
    await session_manager.initialize()
    
    session = await session_manager.get_session(session_id)
    
    if not session:
        return {"error": "Session not found"}
    
    return {
        "session": session.to_dict(),
        "is_expired": session.is_expired(),
        "stats": session_manager.get_stats()
    }


async def create_test_session(user_id: str, data: Dict[str, Any] = None) -> str:
    """Create a test session."""
    session_manager = RedisSessionManager()
    await session_manager.initialize()
    
    session = await session_manager.create_session(
        user_id=user_id,
        data=data or {"test": True},
        ip_address="127.0.0.1",
        user_agent="session_manager_script",
        roles=["user"]
    )
    
    return session.session_id


async def destroy_session(session_id: str) -> bool:
    """Destroy a session."""
    session_manager = RedisSessionManager()
    await session_manager.initialize()
    
    return await session_manager.destroy_session(session_id)


async def destroy_user_sessions(user_id: str, except_session: str = None) -> int:
    """Destroy all sessions for a user."""
    session_manager = RedisSessionManager()
    await session_manager.initialize()
    
    return await session_manager.destroy_user_sessions(user_id, except_session)


async def session_analytics() -> Dict[str, Any]:
    """Get session analytics."""
    session_manager = RedisSessionManager()
    await session_manager.initialize()
    
    # Get all sessions
    sessions = await session_manager.list_active_sessions(limit=1000)
    
    # Calculate analytics
    total_sessions = len(sessions)
    authenticated_sessions = len([s for s in sessions if s.is_authenticated])
    expired_sessions = len([s for s in sessions if s.is_expired()])
    
    # User distribution
    user_sessions = {}
    for session in sessions:
        if session.user_id:
            user_sessions[session.user_id] = user_sessions.get(session.user_id, 0) + 1
    
    # IP distribution
    ip_sessions = {}
    for session in sessions:
        if session.ip_address:
            ip_sessions[session.ip_address] = ip_sessions.get(session.ip_address, 0) + 1
    
    # Age distribution
    now = datetime.now()
    age_buckets = {"<1h": 0, "1-6h": 0, "6-24h": 0, ">24h": 0}
    
    for session in sessions:
        if session.created_at:
            age_hours = (now - session.created_at).total_seconds() / 3600
            if age_hours < 1:
                age_buckets["<1h"] += 1
            elif age_hours < 6:
                age_buckets["1-6h"] += 1
            elif age_hours < 24:
                age_buckets["6-24h"] += 1
            else:
                age_buckets[">24h"] += 1
    
    return {
        "total_sessions": total_sessions,
        "authenticated_sessions": authenticated_sessions,
        "anonymous_sessions": total_sessions - authenticated_sessions,
        "expired_sessions": expired_sessions,
        "unique_users": len(user_sessions),
        "unique_ips": len(ip_sessions),
        "top_users": sorted(user_sessions.items(), key=lambda x: x[1], reverse=True)[:10],
        "top_ips": sorted(ip_sessions.items(), key=lambda x: x[1], reverse=True)[:10],
        "age_distribution": age_buckets,
        "manager_stats": session_manager.get_stats()
    }


def display_sessions(sessions: List[Session]):
    """Display sessions in a table."""
    if not sessions:
        console.print("No sessions found")
        return
    
    table = Table(title="Active Sessions")
    table.add_column("Session ID", style="cyan")
    table.add_column("User ID", style="white")
    table.add_column("Created", style="dim")
    table.add_column("Last Access", style="dim")
    table.add_column("IP Address", style="yellow")
    table.add_column("Status", style="green")
    
    for session in sessions:
        status = "🔓 Anonymous" if not session.is_authenticated else "🔒 Authenticated"
        if session.is_expired():
            status = "⏰ Expired"
        
        table.add_row(
            session.session_id[:16] + "...",
            session.user_id or "Anonymous",
            session.created_at.strftime("%H:%M:%S") if session.created_at else "Unknown",
            session.last_accessed.strftime("%H:%M:%S") if session.last_accessed else "Unknown",
            session.ip_address or "Unknown",
            status
        )
    
    console.print(table)


def display_session_details(details: Dict[str, Any]):
    """Display detailed session information."""
    if "error" in details:
        console.print(f"[red]Error: {details['error']}[/red]")
        return
    
    session_data = details["session"]
    
    # Session info panel
    info_text = f"""
Session ID: {session_data['session_id']}
User ID: {session_data.get('user_id', 'Anonymous')}
Authenticated: {'Yes' if session_data.get('is_authenticated') else 'No'}
IP Address: {session_data.get('ip_address', 'Unknown')}
User Agent: {session_data.get('user_agent', 'Unknown')}
Roles: {', '.join(session_data.get('roles', []))}

Created: {session_data.get('created_at', 'Unknown')}
Last Access: {session_data.get('last_accessed', 'Unknown')}
Expires: {session_data.get('expires_at', 'Never')}
Expired: {'Yes' if details.get('is_expired') else 'No'}
"""
    
    console.print(Panel(info_text.strip(), title="Session Information"))
    
    # Session data
    if session_data.get('data'):
        data_text = json.dumps(session_data['data'], indent=2)
        console.print(Panel(data_text, title="Session Data"))


def display_analytics(analytics: Dict[str, Any]):
    """Display session analytics."""
    # Summary stats
    summary_table = Table(title="Session Summary")
    summary_table.add_column("Metric", style="cyan")
    summary_table.add_column("Value", style="white")
    
    summary_table.add_row("Total Sessions", str(analytics["total_sessions"]))
    summary_table.add_row("Authenticated", str(analytics["authenticated_sessions"]))
    summary_table.add_row("Anonymous", str(analytics["anonymous_sessions"]))
    summary_table.add_row("Expired", str(analytics["expired_sessions"]))
    summary_table.add_row("Unique Users", str(analytics["unique_users"]))
    summary_table.add_row("Unique IPs", str(analytics["unique_ips"]))
    
    console.print(summary_table)
    
    # Top users
    if analytics["top_users"]:
        users_table = Table(title="Top Users by Session Count")
        users_table.add_column("User ID", style="cyan")
        users_table.add_column("Sessions", style="white")
        
        for user_id, count in analytics["top_users"]:
            users_table.add_row(user_id, str(count))
        
        console.print(users_table)
    
    # Age distribution
    age_table = Table(title="Session Age Distribution")
    age_table.add_column("Age Range", style="cyan")
    age_table.add_column("Count", style="white")
    
    for age_range, count in analytics["age_distribution"].items():
        age_table.add_row(age_range, str(count))
    
    console.print(age_table)
    
    # Manager stats
    stats = analytics["manager_stats"]
    stats_table = Table(title="Session Manager Statistics")
    stats_table.add_column("Metric", style="cyan")
    stats_table.add_column("Value", style="white")
    
    for key, value in stats.items():
        stats_table.add_row(key.replace("_", " ").title(), str(value))
    
    console.print(stats_table)


async def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Session management utility")
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # List sessions
    list_parser = subparsers.add_parser("list", help="List active sessions")
    list_parser.add_argument("--user", help="Filter by user ID")
    list_parser.add_argument("--limit", type=int, default=50, help="Maximum sessions to list")
    list_parser.add_argument("--include-expired", action="store_true", help="Include expired sessions")
    
    # Session details
    info_parser = subparsers.add_parser("info", help="Get session details")
    info_parser.add_argument("session_id", help="Session ID to inspect")
    
    # Create test session
    create_parser = subparsers.add_parser("create", help="Create test session")
    create_parser.add_argument("user_id", help="User ID for session")
    create_parser.add_argument("--data", help="JSON data for session")
    
    # Destroy session
    destroy_parser = subparsers.add_parser("destroy", help="Destroy session")
    destroy_parser.add_argument("session_id", help="Session ID to destroy")
    
    # Destroy user sessions
    destroy_user_parser = subparsers.add_parser("destroy-user", help="Destroy all user sessions")
    destroy_user_parser.add_argument("user_id", help="User ID")
    destroy_user_parser.add_argument("--except", dest="except_session", help="Session to preserve")
    
    # Session analytics
    subparsers.add_parser("analytics", help="Show session analytics")
    
    # Cleanup expired
    subparsers.add_parser("cleanup", help="Clean up expired sessions")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    try:
        if args.command == "list":
            sessions = await list_sessions(
                user_id=args.user,
                limit=args.limit,
                include_expired=args.include_expired
            )
            display_sessions(sessions)
        
        elif args.command == "info":
            details = await get_session_details(args.session_id)
            display_session_details(details)
        
        elif args.command == "create":
            data = None
            if args.data:
                try:
                    data = json.loads(args.data)
                except json.JSONDecodeError:
                    console.print("[red]Invalid JSON data[/red]")
                    return 1
            
            session_id = await create_test_session(args.user_id, data)
            console.print(f"[green]Created session: {session_id}[/green]")
        
        elif args.command == "destroy":
            if await destroy_session(args.session_id):
                console.print(f"[green]Destroyed session: {args.session_id}[/green]")
            else:
                console.print(f"[red]Session not found: {args.session_id}[/red]")
        
        elif args.command == "destroy-user":
            if not Confirm.ask(f"Destroy all sessions for user {args.user_id}?"):
                console.print("Cancelled")
                return 0
            
            count = await destroy_user_sessions(args.user_id, args.except_session)
            console.print(f"[green]Destroyed {count} sessions for user {args.user_id}[/green]")
        
        elif args.command == "analytics":
            analytics = await session_analytics()
            display_analytics(analytics)
        
        elif args.command == "cleanup":
            # This would be handled by the session manager's background cleanup
            console.print("[green]Cleanup is handled automatically by the session manager[/green]")
        
        return 0
    
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        return 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))