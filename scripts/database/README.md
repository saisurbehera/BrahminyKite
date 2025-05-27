# Database Management Scripts

This directory contains utility scripts for managing the BrahminyKite PostgreSQL database.

## Scripts

### init_db.py
Initialize the database with schema and optional seed data.

```bash
# Basic initialization
python scripts/database/init_db.py

# Skip seed data
python scripts/database/init_db.py --no-seed

# Drop and recreate database (WARNING: destructive!)
python scripts/database/init_db.py --drop-existing
```

### backup_db.py
Create database backups in multiple formats.

```bash
# Create full backup (SQL and custom format)
python scripts/database/backup_db.py

# SQL dump only
python scripts/database/backup_db.py --format sql

# Custom format only (faster restores)
python scripts/database/backup_db.py --format custom

# Export specific tables to CSV
python scripts/database/backup_db.py --tables consensus_nodes verification_requests

# Keep only last 3 backups
python scripts/database/backup_db.py --keep 3
```

### restore_db.py
Restore database from backup.

```bash
# List available backups
python scripts/database/restore_db.py --list

# Restore from latest backup
python scripts/database/restore_db.py --latest

# Restore from specific backup
python scripts/database/restore_db.py --backup backup_20240115_120000

# Restore without dropping existing objects
python scripts/database/restore_db.py --latest --no-clean

# Import only CSV data
python scripts/database/restore_db.py --backup backup_20240115_120000 --csv-only
```

### db_health.py
Monitor database health and performance.

```bash
# One-time health check
python scripts/database/db_health.py

# Continuous monitoring (updates every 5 seconds)
python scripts/database/db_health.py --monitor

# Custom monitoring interval
python scripts/database/db_health.py --monitor --interval 10

# Run diagnostics
python scripts/database/db_health.py --diagnostics

# Output as JSON
python scripts/database/db_health.py --json
```

## Environment Variables

All scripts use the following environment variables for database connection:

```bash
export DATABASE_HOST=localhost
export DATABASE_PORT=5432
export DATABASE_NAME=brahminykite
export DATABASE_USER=postgres
export DATABASE_PASSWORD=password

# Or use a single URL
export DATABASE_URL=postgresql://user:password@localhost:5432/brahminykite
```

## Prerequisites

### System Requirements
- PostgreSQL client tools (psql, pg_dump, pg_restore)
- Python 3.8+

### Python Dependencies
```bash
pip install asyncpg rich alembic sqlalchemy
```

## Backup Strategy

### Recommended Schedule
- **Daily**: SQL dump with compression
- **Weekly**: Custom format backup for faster restores
- **Monthly**: Full backup with CSV exports

### Example Cron Jobs
```cron
# Daily SQL backup at 2 AM
0 2 * * * /usr/bin/python /path/to/scripts/database/backup_db.py --format sql

# Weekly custom backup on Sundays at 3 AM
0 3 * * 0 /usr/bin/python /path/to/scripts/database/backup_db.py --format custom

# Monthly full backup on 1st at 4 AM
0 4 1 * * /usr/bin/python /path/to/scripts/database/backup_db.py --tables ALL
```

## Monitoring

### Key Metrics to Watch
- **Cache Hit Ratio**: Should be > 90%
- **Index Usage**: Should be > 80%
- **Active Connections**: Monitor for spikes
- **Lock Contention**: Check for blocking queries
- **Replication Lag**: Should be < 1 second

### Alerts
Set up alerts for:
- Connection failures
- Low cache hit ratio
- High lock contention
- Long-running queries (> 5 minutes)
- Replication lag > 10 seconds

## Troubleshooting

### Connection Issues
```bash
# Test connection
psql -h $DATABASE_HOST -p $DATABASE_PORT -U $DATABASE_USER -d $DATABASE_NAME -c "SELECT 1"

# Check PostgreSQL service
sudo systemctl status postgresql
```

### Performance Issues
```bash
# Check slow queries
python scripts/database/db_health.py --diagnostics

# Analyze query plans
psql -d brahminykite -c "EXPLAIN ANALYZE <query>"
```

### Backup/Restore Issues
```bash
# Verify backup integrity
pg_restore -l backup_file.custom

# Check backup size
du -h backups/

# Test restore to different database
createdb test_restore
python scripts/database/restore_db.py --latest
dropdb test_restore
```

## Security

### Best Practices
1. Use strong passwords and rotate regularly
2. Restrict database user permissions
3. Enable SSL/TLS connections
4. Encrypt backups at rest
5. Limit network access with firewall rules

### Backup Encryption
```bash
# Encrypt backup
gpg --symmetric --cipher-algo AES256 backup.sql

# Decrypt backup
gpg --decrypt backup.sql.gpg > backup.sql
```