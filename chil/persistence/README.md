# BrahminyKite PostgreSQL Persistence Layer

Production-ready data persistence layer for the BrahminyKite AI verification system.

## Overview

The persistence layer provides:
- **Database Models**: SQLAlchemy ORM models for all entities
- **Connection Pooling**: Efficient connection management
- **Repository Pattern**: Clean data access abstraction
- **Migrations**: Version-controlled schema changes
- **Transaction Management**: ACID compliance
- **Query Optimization**: Indexes and performance tuning
- **Monitoring**: Database performance metrics

## Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Services      │    │   Consensus     │    │   API Gateway   │
└────────┬────────┘    └────────┬────────┘    └────────┬────────┘
         │                      │                      │
         └──────────────────────┼──────────────────────┘
                                │
                    ┌───────────▼───────────┐
                    │   Repository Layer    │
                    │  (Data Access Logic)  │
                    └───────────┬───────────┘
                                │
                    ┌───────────▼───────────┐
                    │    ORM Models         │
                    │   (SQLAlchemy)        │
                    └───────────┬───────────┘
                                │
                    ┌───────────▼───────────┐
                    │  Connection Pool      │
                    │   (asyncpg/psycopg3)  │
                    └───────────┬───────────┘
                                │
                    ┌───────────▼───────────┐
                    │     PostgreSQL        │
                    │     Database          │
                    └───────────────────────┘
```

## Database Schema

### Core Tables

```sql
-- Verification Results
CREATE TABLE verification_results (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    claim_id VARCHAR(255) NOT NULL,
    framework VARCHAR(50) NOT NULL,
    result JSONB NOT NULL,
    confidence FLOAT NOT NULL CHECK (confidence >= 0 AND confidence <= 1),
    evidence JSONB,
    processing_time_ms INTEGER,
    model_version VARCHAR(50),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_claim_framework (claim_id, framework),
    INDEX idx_created_at (created_at DESC)
);

-- Consensus Decisions
CREATE TABLE consensus_decisions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    claim_id VARCHAR(255) NOT NULL UNIQUE,
    consensus_result JSONB NOT NULL,
    participating_frameworks JSONB NOT NULL,
    consensus_type VARCHAR(50) NOT NULL,
    consensus_confidence FLOAT NOT NULL,
    decision_time_ms INTEGER,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_consensus_created (created_at DESC)
);

-- Claims
CREATE TABLE claims (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    external_id VARCHAR(255) UNIQUE NOT NULL,
    content TEXT NOT NULL,
    claim_type VARCHAR(100),
    source VARCHAR(255),
    metadata JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_claim_type (claim_type),
    INDEX idx_created_at (created_at DESC)
);

-- Training Data
CREATE TABLE training_data (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    framework VARCHAR(50) NOT NULL,
    claim_id UUID REFERENCES claims(id),
    claim_text TEXT NOT NULL,
    evidence JSONB NOT NULL,
    validity BOOLEAN NOT NULL,
    confidence FLOAT,
    metadata JSONB,
    used_in_training BOOLEAN DEFAULT FALSE,
    training_batch_id VARCHAR(255),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_framework_batch (framework, training_batch_id),
    INDEX idx_training_status (used_in_training, framework)
);

-- Model Checkpoints
CREATE TABLE model_checkpoints (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    model_name VARCHAR(255) NOT NULL,
    framework VARCHAR(50),
    version VARCHAR(50) NOT NULL,
    checkpoint_path TEXT NOT NULL,
    metrics JSONB NOT NULL,
    training_config JSONB,
    is_active BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(model_name, version),
    INDEX idx_active_models (is_active, framework)
);

-- Service Health
CREATE TABLE service_health (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    service_name VARCHAR(50) NOT NULL,
    instance_id VARCHAR(255) NOT NULL,
    status VARCHAR(20) NOT NULL,
    last_heartbeat TIMESTAMP WITH TIME ZONE NOT NULL,
    metadata JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_service_heartbeat (service_name, last_heartbeat DESC)
);

-- Audit Log
CREATE TABLE audit_log (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    entity_type VARCHAR(50) NOT NULL,
    entity_id VARCHAR(255) NOT NULL,
    action VARCHAR(50) NOT NULL,
    user_id VARCHAR(255),
    changes JSONB,
    metadata JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_entity (entity_type, entity_id),
    INDEX idx_audit_created (created_at DESC)
);
```

## Usage

### Basic Configuration

```python
from chil.persistence import DatabaseConfig, create_database_engine

# Configure database
config = DatabaseConfig(
    host="localhost",
    port=5432,
    database="brahminykite",
    user="brahminykite",
    password="your_password",
    pool_size=20,
    max_overflow=10
)

# Create engine
engine = await create_database_engine(config)
```

### Repository Pattern

```python
from chil.persistence import VerificationRepository

# Create repository
repo = VerificationRepository(engine)

# Save verification result
result = await repo.create_verification(
    claim_id="claim-123",
    framework="empirical",
    result={"valid": True, "reasoning": "..."},
    confidence=0.95,
    evidence={"sources": [...]}
)

# Query results
results = await repo.get_by_claim_id("claim-123")
recent = await repo.get_recent_verifications(limit=100)
```

### Transaction Management

```python
from chil.persistence import transaction

@transaction
async def process_verification(repo, claim, results):
    # All database operations in transaction
    claim_record = await repo.create_claim(claim)
    
    for framework, result in results.items():
        await repo.create_verification(
            claim_id=claim_record.id,
            framework=framework,
            **result
        )
    
    # Automatic commit on success, rollback on error
```

### Migrations

```bash
# Create new migration
alembic revision -m "Add new table"

# Apply migrations
alembic upgrade head

# Rollback
alembic downgrade -1
```

## Models

### SQLAlchemy Models

```python
from chil.persistence.models import (
    VerificationResult,
    ConsensusDecision,
    Claim,
    TrainingData,
    ModelCheckpoint,
    ServiceHealth,
    AuditLog
)

# Type-safe model usage
verification = VerificationResult(
    claim_id="claim-123",
    framework="empirical",
    result={"valid": True},
    confidence=0.95
)
```

## Connection Pooling

The persistence layer uses asyncpg for high-performance async PostgreSQL access:

- **Connection Pool Size**: Configurable (default: 20)
- **Max Overflow**: Additional connections under load
- **Connection Recycling**: Automatic stale connection handling
- **Health Checks**: Periodic connection validation
- **Retry Logic**: Automatic retry on transient failures

## Performance Optimization

### Indexes

- Composite indexes for common query patterns
- Partial indexes for filtered queries
- GIN indexes for JSONB fields
- BRIN indexes for time-series data

### Query Optimization

- Prepared statements for repeated queries
- Batch operations for bulk inserts
- Connection pooling per service
- Read replicas for scaling

### Monitoring

```python
# Database metrics
from chil.persistence.monitoring import DatabaseMetrics

metrics = DatabaseMetrics()
metrics.record_query_duration("get_verification", 0.025)
metrics.record_connection_pool_size(15)
metrics.record_transaction_status("success")
```

## Security

- **SQL Injection Protection**: Parameterized queries only
- **Connection Encryption**: SSL/TLS required
- **Access Control**: Row-level security where needed
- **Audit Logging**: All modifications tracked
- **Sensitive Data**: Encryption at rest

## Development

### Running Tests

```bash
# Unit tests
pytest chil/persistence/tests/unit/

# Integration tests (requires PostgreSQL)
pytest chil/persistence/tests/integration/

# Performance tests
pytest chil/persistence/tests/performance/
```

### Local Development

```bash
# Start PostgreSQL
docker-compose up -d postgres

# Run migrations
alembic upgrade head

# Seed test data
python scripts/seed_database.py
```

## Best Practices

1. **Use Repositories**: Don't access models directly from services
2. **Transaction Boundaries**: Keep transactions short
3. **Batch Operations**: Use bulk inserts/updates
4. **Connection Management**: Share connection pools
5. **Error Handling**: Graceful degradation on DB issues
6. **Monitoring**: Track all database operations
7. **Testing**: Test with realistic data volumes