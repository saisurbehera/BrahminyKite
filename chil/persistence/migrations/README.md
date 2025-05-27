# Database Migrations

This directory contains Alembic database migrations for the BrahminyKite project.

## Setup

1. Ensure your database connection is configured via environment variables:
```bash
export DATABASE_URL="postgresql://user:password@localhost:5432/brahminykite"
```

2. Create the initial migration:
```bash
alembic revision --autogenerate -m "Initial schema"
```

3. Apply migrations:
```bash
alembic upgrade head
```

## Common Commands

### Create a new migration
```bash
# Auto-generate based on model changes
alembic revision --autogenerate -m "Description of changes"

# Create empty migration
alembic revision -m "Description of changes"
```

### Apply migrations
```bash
# Upgrade to latest
alembic upgrade head

# Upgrade to specific revision
alembic upgrade <revision>

# Upgrade one revision
alembic upgrade +1
```

### Downgrade migrations
```bash
# Downgrade to previous
alembic downgrade -1

# Downgrade to specific revision
alembic downgrade <revision>

# Downgrade all
alembic downgrade base
```

### View migration history
```bash
# Show current revision
alembic current

# Show history
alembic history

# Show history with details
alembic history --verbose
```

## Best Practices

1. **Always review auto-generated migrations** - Alembic's autogenerate is good but not perfect
2. **Test migrations** - Run upgrade and downgrade on a test database
3. **Keep migrations small** - One logical change per migration
4. **Use descriptive messages** - Make it clear what the migration does
5. **Don't edit applied migrations** - Create a new migration to fix issues

## Troubleshooting

### Migration conflicts
If you get conflicts when multiple developers create migrations:
1. Merge the latest changes
2. Delete your local migration
3. Recreate it with `alembic revision --autogenerate`

### Failed migrations
If a migration fails partway through:
1. Manually fix the database state
2. Update the alembic_version table if needed
3. Consider creating a fixing migration

### Out of sync
If models and database are out of sync:
```bash
# Check what would be generated
alembic revision --autogenerate --sql

# Force sync (be careful!)
alembic stamp head
```