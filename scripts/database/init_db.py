#!/usr/bin/env python3
"""
Initialize the BrahminyKite database.

This script:
1. Creates the database if it doesn't exist
2. Runs all migrations
3. Optionally seeds initial data
"""

import asyncio
import argparse
import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from alembic import command
from alembic.config import Config
import asyncpg
from sqlalchemy import text

from chil.persistence.core.config import DatabaseConfig
from chil.persistence.core.database import create_database_engine

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def create_database_if_not_exists(config: DatabaseConfig) -> None:
    """Create the database if it doesn't exist."""
    # Connect to PostgreSQL server (not specific database)
    conn = await asyncpg.connect(
        host=config.host,
        port=config.port,
        user=config.username,
        password=config.password,
        database="postgres"  # Connect to default postgres database
    )
    
    try:
        # Check if database exists
        exists = await conn.fetchval(
            "SELECT 1 FROM pg_database WHERE datname = $1",
            config.database
        )
        
        if not exists:
            logger.info(f"Creating database '{config.database}'...")
            await conn.execute(f'CREATE DATABASE "{config.database}"')
            logger.info(f"Database '{config.database}' created successfully")
        else:
            logger.info(f"Database '{config.database}' already exists")
    
    finally:
        await conn.close()


async def verify_connection(config: DatabaseConfig) -> bool:
    """Verify database connection."""
    try:
        engine = await create_database_engine(config)
        
        # Test connection
        async with engine.session() as session:
            result = await session.execute(text("SELECT 1"))
            logger.info("Database connection verified successfully")
            return True
    
    except Exception as e:
        logger.error(f"Failed to connect to database: {e}")
        return False
    
    finally:
        await engine.close()


def run_migrations() -> None:
    """Run Alembic migrations."""
    logger.info("Running database migrations...")
    
    # Get alembic.ini path
    alembic_ini = Path(__file__).parent.parent.parent / "alembic.ini"
    
    if not alembic_ini.exists():
        logger.error(f"alembic.ini not found at {alembic_ini}")
        return
    
    # Create Alembic config
    alembic_cfg = Config(str(alembic_ini))
    
    try:
        # Run migrations
        command.upgrade(alembic_cfg, "head")
        logger.info("Migrations completed successfully")
    except Exception as e:
        logger.error(f"Migration failed: {e}")
        raise


async def seed_initial_data(config: DatabaseConfig) -> None:
    """Seed initial data into the database."""
    logger.info("Seeding initial data...")
    
    engine = await create_database_engine(config)
    
    try:
        async with engine.session() as session:
            # Add any initial data here
            # For example, create default configurations, admin users, etc.
            
            # Example: Create default framework configurations
            from chil.persistence.repositories import FrameworkRepository
            
            repo = FrameworkRepository(session)
            
            # Create default configs for each framework
            frameworks = [
                "consistency", "empirical", "contextual",
                "power_dynamics", "utility", "evolutionary", "meta"
            ]
            
            for framework in frameworks:
                existing = await repo.configs.get_active_config(
                    framework_name=framework,
                    version="1.0.0"
                )
                
                if not existing:
                    await repo.configs.create_config(
                        framework_name=framework,
                        version="1.0.0",
                        config_data={
                            "enabled": True,
                            "timeout": 300,
                            "max_retries": 3
                        }
                    )
                    logger.info(f"Created default config for {framework}")
            
            await session.commit()
            logger.info("Initial data seeded successfully")
    
    except Exception as e:
        logger.error(f"Failed to seed data: {e}")
        raise
    
    finally:
        await engine.close()


async def main():
    """Main initialization function."""
    parser = argparse.ArgumentParser(description="Initialize BrahminyKite database")
    parser.add_argument(
        "--no-seed",
        action="store_true",
        help="Skip seeding initial data"
    )
    parser.add_argument(
        "--drop-existing",
        action="store_true",
        help="Drop existing database before creating (WARNING: destructive!)"
    )
    
    args = parser.parse_args()
    
    # Load database configuration
    config = DatabaseConfig.from_env()
    logger.info(f"Database configuration loaded: {config.host}:{config.port}/{config.database}")
    
    try:
        # Drop existing database if requested
        if args.drop_existing:
            logger.warning("Dropping existing database...")
            conn = await asyncpg.connect(
                host=config.host,
                port=config.port,
                user=config.username,
                password=config.password,
                database="postgres"
            )
            try:
                # Terminate existing connections
                await conn.execute(f"""
                    SELECT pg_terminate_backend(pid)
                    FROM pg_stat_activity
                    WHERE datname = '{config.database}' AND pid <> pg_backend_pid()
                """)
                
                # Drop database
                await conn.execute(f'DROP DATABASE IF EXISTS "{config.database}"')
                logger.info("Existing database dropped")
            finally:
                await conn.close()
        
        # Create database if needed
        await create_database_if_not_exists(config)
        
        # Verify connection
        if not await verify_connection(config):
            logger.error("Failed to verify database connection")
            return 1
        
        # Run migrations
        run_migrations()
        
        # Seed initial data
        if not args.no_seed:
            await seed_initial_data(config)
        
        logger.info("Database initialization completed successfully!")
        return 0
    
    except Exception as e:
        logger.error(f"Database initialization failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))