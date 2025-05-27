#!/usr/bin/env python3
"""
Restore the BrahminyKite database from backup.

This script restores database backups created by backup_db.py.
"""

import asyncio
import argparse
import logging
import subprocess
import sys
from pathlib import Path
import json
import gzip
import shutil

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from chil.persistence.core.config import DatabaseConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def list_available_backups() -> list:
    """List all available backups."""
    backups_dir = Path("backups")
    if not backups_dir.exists():
        return []
    
    backups = []
    for backup_dir in sorted(backups_dir.iterdir(), reverse=True):
        if backup_dir.is_dir() and backup_dir.name.startswith("backup_"):
            metadata_file = backup_dir / "backup_metadata.json"
            if metadata_file.exists():
                with open(metadata_file) as f:
                    metadata = json.load(f)
                    backups.append({
                        "dir": backup_dir,
                        "timestamp": metadata["timestamp"],
                        "database": metadata["database"],
                        "files": metadata["files"]
                    })
    
    return backups


def run_pg_restore(
    config: DatabaseConfig,
    backup_file: Path,
    format: str = "custom",
    clean: bool = True
) -> bool:
    """Run pg_restore to restore database backup."""
    logger.info(f"Restoring from {backup_file}...")
    
    if format == "plain":
        # For plain SQL files, use psql
        cmd = [
            "psql",
            "-h", config.host,
            "-p", str(config.port),
            "-U", config.username,
            "-d", config.database,
        ]
        
        # Handle compressed files
        if str(backup_file).endswith('.gz'):
            logger.info("Decompressing backup...")
            with gzip.open(backup_file, 'rb') as f_in:
                content = f_in.read()
            
            # Run psql with input from stdin
            env = {"PGPASSWORD": config.password}
            result = subprocess.run(cmd, input=content, env=env, capture_output=True)
        else:
            cmd.extend(["-f", str(backup_file)])
            env = {"PGPASSWORD": config.password}
            result = subprocess.run(cmd, env=env, capture_output=True, text=True)
    
    else:
        # Use pg_restore for custom format
        cmd = [
            "pg_restore",
            "-h", config.host,
            "-p", str(config.port),
            "-U", config.username,
            "-d", config.database,
            "-v",  # Verbose
            "--no-owner",
            "--no-privileges",
        ]
        
        if clean:
            cmd.append("--clean")  # Drop existing objects
            cmd.append("--if-exists")
        
        cmd.append(str(backup_file))
        
        env = {"PGPASSWORD": config.password}
        result = subprocess.run(cmd, env=env, capture_output=True, text=True)
    
    if result.returncode == 0:
        logger.info("Restore completed successfully")
        return True
    else:
        # pg_restore returns warnings as errors, check if it's just warnings
        if result.stderr and "WARNING" in result.stderr and "ERROR" not in result.stderr:
            logger.warning(f"Restore completed with warnings: {result.stderr}")
            return True
        else:
            logger.error(f"Restore failed: {result.stderr}")
            return False


async def verify_restore(config: DatabaseConfig) -> bool:
    """Verify the restored database."""
    logger.info("Verifying restored database...")
    
    import asyncpg
    
    try:
        conn = await asyncpg.connect(
            host=config.host,
            port=config.port,
            user=config.username,
            password=config.password,
            database=config.database
        )
        
        # Check tables exist
        tables = await conn.fetch("""
            SELECT tablename 
            FROM pg_tables 
            WHERE schemaname = 'public'
            ORDER BY tablename
        """)
        
        logger.info(f"Found {len(tables)} tables:")
        for table in tables:
            logger.info(f"  - {table['tablename']}")
        
        # Check row counts
        for table in tables:
            count = await conn.fetchval(f"SELECT COUNT(*) FROM {table['tablename']}")
            logger.info(f"  {table['tablename']}: {count} rows")
        
        await conn.close()
        return True
    
    except Exception as e:
        logger.error(f"Verification failed: {e}")
        return False


async def import_csv_data(config: DatabaseConfig, csv_dir: Path) -> None:
    """Import data from CSV files."""
    logger.info("Importing CSV data...")
    
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
        for csv_file in csv_dir.glob("*.csv"):
            table_name = csv_file.stem
            logger.info(f"Importing {table_name} from CSV...")
            
            with open(csv_file, 'r') as f:
                reader = csv.DictReader(f)
                rows = list(reader)
                
                if rows:
                    # Build insert query
                    columns = list(rows[0].keys())
                    placeholders = [f"${i+1}" for i in range(len(columns))]
                    
                    query = f"""
                        INSERT INTO {table_name} ({', '.join(columns)})
                        VALUES ({', '.join(placeholders)})
                    """
                    
                    # Insert rows
                    for row in rows:
                        values = [row[col] for col in columns]
                        await conn.execute(query, *values)
                    
                    logger.info(f"Imported {len(rows)} rows into {table_name}")
    
    finally:
        await conn.close()


async def main():
    """Main restore function."""
    parser = argparse.ArgumentParser(description="Restore BrahminyKite database")
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available backups"
    )
    parser.add_argument(
        "--backup",
        help="Backup directory to restore from"
    )
    parser.add_argument(
        "--latest",
        action="store_true",
        help="Restore from latest backup"
    )
    parser.add_argument(
        "--no-clean",
        action="store_true",
        help="Don't drop existing objects before restore"
    )
    parser.add_argument(
        "--csv-only",
        action="store_true",
        help="Only import CSV data (assumes empty schema exists)"
    )
    parser.add_argument(
        "--no-verify",
        action="store_true",
        help="Skip verification after restore"
    )
    
    args = parser.parse_args()
    
    # List backups if requested
    if args.list:
        backups = list_available_backups()
        if not backups:
            logger.info("No backups found")
        else:
            logger.info("Available backups:")
            for i, backup in enumerate(backups):
                logger.info(f"{i+1}. {backup['timestamp']} - {backup['database']}")
                logger.info(f"   Location: {backup['dir']}")
                logger.info(f"   Files: {', '.join(backup['files'])}")
        return 0
    
    # Determine which backup to use
    backup_dir = None
    
    if args.latest:
        backups = list_available_backups()
        if backups:
            backup_dir = backups[0]["dir"]
            logger.info(f"Using latest backup: {backup_dir}")
        else:
            logger.error("No backups found")
            return 1
    
    elif args.backup:
        backup_dir = Path(args.backup)
        if not backup_dir.exists():
            # Try looking in backups directory
            backup_dir = Path("backups") / args.backup
            if not backup_dir.exists():
                logger.error(f"Backup not found: {args.backup}")
                return 1
    
    else:
        logger.error("Please specify --backup, --latest, or --list")
        return 1
    
    # Load database configuration
    config = DatabaseConfig.from_env()
    logger.info(f"Restoring to database: {config.database}")
    
    try:
        if args.csv_only:
            # Only import CSV data
            csv_dir = backup_dir / "csv_exports"
            if csv_dir.exists():
                await import_csv_data(config, csv_dir)
            else:
                logger.error("No CSV exports found in backup")
                return 1
        
        else:
            # Look for backup files
            success = False
            
            # Try custom format first
            custom_file = backup_dir / "dump.custom"
            if custom_file.exists():
                success = run_pg_restore(
                    config,
                    custom_file,
                    "custom",
                    not args.no_clean
                )
            
            else:
                # Try SQL dump
                sql_file = backup_dir / "dump.sql"
                if sql_file.exists():
                    success = run_pg_restore(
                        config,
                        sql_file,
                        "plain",
                        not args.no_clean
                    )
                else:
                    # Try compressed SQL
                    sql_gz_file = backup_dir / "dump.sql.gz"
                    if sql_gz_file.exists():
                        success = run_pg_restore(
                            config,
                            sql_gz_file,
                            "plain",
                            not args.no_clean
                        )
                    else:
                        logger.error("No backup files found")
                        return 1
            
            if not success:
                return 1
        
        # Verify restore
        if not args.no_verify:
            if not await verify_restore(config):
                logger.warning("Verification failed, but restore may have succeeded")
        
        logger.info("Restore completed successfully!")
        return 0
    
    except Exception as e:
        logger.error(f"Restore failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))