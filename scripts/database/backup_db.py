#!/usr/bin/env python3
"""
Backup the BrahminyKite database.

This script creates backups of the database in multiple formats:
1. SQL dump (pg_dump)
2. Custom format dump (for faster restores)
3. Optional: CSV export of specific tables
"""

import asyncio
import argparse
import logging
import subprocess
import sys
from datetime import datetime
from pathlib import Path
import gzip
import shutil

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from chil.persistence.core.config import DatabaseConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_backup_dir() -> Path:
    """Create backup directory with timestamp."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir = Path("backups") / f"backup_{timestamp}"
    backup_dir.mkdir(parents=True, exist_ok=True)
    return backup_dir


def run_pg_dump(
    config: DatabaseConfig,
    output_file: Path,
    format: str = "plain",
    compress: bool = True
) -> bool:
    """Run pg_dump to create database backup."""
    logger.info(f"Creating {format} format backup...")
    
    # Build pg_dump command
    cmd = [
        "pg_dump",
        "-h", config.host,
        "-p", str(config.port),
        "-U", config.username,
        "-d", config.database,
        "-f", str(output_file),
        "-v",  # Verbose
        "--no-owner",  # Don't include ownership
        "--no-privileges",  # Don't include privileges
    ]
    
    if format == "custom":
        cmd.extend(["-F", "c"])  # Custom format
        if compress:
            cmd.append("-Z9")  # Maximum compression
    elif format == "directory":
        cmd.extend(["-F", "d"])  # Directory format
        cmd.extend(["-j", "4"])  # Use 4 parallel jobs
    
    # Set password via environment
    env = {"PGPASSWORD": config.password}
    
    try:
        result = subprocess.run(cmd, env=env, capture_output=True, text=True)
        
        if result.returncode == 0:
            logger.info(f"Backup created successfully: {output_file}")
            
            # Compress plain SQL dumps
            if format == "plain" and compress and not str(output_file).endswith('.gz'):
                logger.info("Compressing SQL dump...")
                with open(output_file, 'rb') as f_in:
                    with gzip.open(f"{output_file}.gz", 'wb') as f_out:
                        shutil.copyfileobj(f_in, f_out)
                
                # Remove uncompressed file
                output_file.unlink()
                logger.info(f"Compressed backup: {output_file}.gz")
            
            return True
        else:
            logger.error(f"pg_dump failed: {result.stderr}")
            return False
    
    except Exception as e:
        logger.error(f"Failed to run pg_dump: {e}")
        return False


async def export_tables_to_csv(config: DatabaseConfig, backup_dir: Path, tables: list) -> None:
    """Export specific tables to CSV format."""
    logger.info("Exporting tables to CSV...")
    
    import asyncpg
    import csv
    
    conn = await asyncpg.connect(
        host=config.host,
        port=config.port,
        user=config.username,
        password=config.password,
        database=config.database
    )
    
    try:
        csv_dir = backup_dir / "csv_exports"
        csv_dir.mkdir(exist_ok=True)
        
        for table in tables:
            logger.info(f"Exporting table: {table}")
            
            # Get table data
            rows = await conn.fetch(f"SELECT * FROM {table}")
            
            if rows:
                # Get column names
                columns = list(rows[0].keys())
                
                # Write to CSV
                csv_file = csv_dir / f"{table}.csv"
                with open(csv_file, 'w', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=columns)
                    writer.writeheader()
                    
                    for row in rows:
                        writer.writerow(dict(row))
                
                logger.info(f"Exported {len(rows)} rows from {table}")
            else:
                logger.warning(f"Table {table} is empty")
    
    finally:
        await conn.close()


def create_backup_metadata(backup_dir: Path, config: DatabaseConfig) -> None:
    """Create metadata file with backup information."""
    metadata = {
        "timestamp": datetime.now().isoformat(),
        "database": config.database,
        "host": config.host,
        "port": config.port,
        "backup_dir": str(backup_dir),
        "files": [str(f.relative_to(backup_dir)) for f in backup_dir.rglob("*") if f.is_file()]
    }
    
    import json
    metadata_file = backup_dir / "backup_metadata.json"
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    logger.info(f"Metadata saved to {metadata_file}")


def cleanup_old_backups(keep_count: int = 5) -> None:
    """Remove old backups, keeping only the most recent ones."""
    backups_dir = Path("backups")
    if not backups_dir.exists():
        return
    
    # Get all backup directories
    backup_dirs = sorted(
        [d for d in backups_dir.iterdir() if d.is_dir() and d.name.startswith("backup_")],
        key=lambda d: d.stat().st_mtime,
        reverse=True
    )
    
    # Remove old backups
    for old_backup in backup_dirs[keep_count:]:
        logger.info(f"Removing old backup: {old_backup}")
        shutil.rmtree(old_backup)


async def main():
    """Main backup function."""
    parser = argparse.ArgumentParser(description="Backup BrahminyKite database")
    parser.add_argument(
        "--format",
        choices=["sql", "custom", "both"],
        default="both",
        help="Backup format (default: both)"
    )
    parser.add_argument(
        "--tables",
        nargs="+",
        help="Specific tables to export to CSV"
    )
    parser.add_argument(
        "--no-compress",
        action="store_true",
        help="Don't compress backups"
    )
    parser.add_argument(
        "--keep",
        type=int,
        default=5,
        help="Number of backups to keep (default: 5)"
    )
    
    args = parser.parse_args()
    
    # Load database configuration
    config = DatabaseConfig.from_env()
    logger.info(f"Backing up database: {config.database}")
    
    # Create backup directory
    backup_dir = create_backup_dir()
    logger.info(f"Backup directory: {backup_dir}")
    
    success = True
    
    try:
        # Create SQL dump
        if args.format in ["sql", "both"]:
            sql_file = backup_dir / "dump.sql"
            if not run_pg_dump(config, sql_file, "plain", not args.no_compress):
                success = False
        
        # Create custom format dump
        if args.format in ["custom", "both"]:
            custom_file = backup_dir / "dump.custom"
            if not run_pg_dump(config, custom_file, "custom", not args.no_compress):
                success = False
        
        # Export specific tables to CSV
        if args.tables:
            await export_tables_to_csv(config, backup_dir, args.tables)
        
        # Create metadata file
        create_backup_metadata(backup_dir, config)
        
        # Cleanup old backups
        cleanup_old_backups(args.keep)
        
        if success:
            logger.info("Backup completed successfully!")
            return 0
        else:
            logger.error("Backup completed with errors")
            return 1
    
    except Exception as e:
        logger.error(f"Backup failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))