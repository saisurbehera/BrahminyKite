"""
Base models and mixins for database entities.
"""

import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import Column, DateTime, String, func
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.ext.asyncio import AsyncAttrs

# Create base class for all models
Base = declarative_base(cls=AsyncAttrs)


class TimestampMixin:
    """Adds created_at and updated_at timestamps to models."""
    
    created_at = Column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now()
    )
    
    updated_at = Column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now()
    )


class UUIDMixin:
    """Adds UUID primary key to models."""
    
    id = Column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        nullable=False
    )


class NamedMixin:
    """Adds name and description fields to models."""
    
    name = Column(String(255), nullable=False)
    description = Column(String(1000), nullable=True)


def get_table_name(model_name: str) -> str:
    """
    Convert model name to table name.
    
    Examples:
        ConsensusNode -> consensus_nodes
        VerificationRequest -> verification_requests
    """
    # Convert CamelCase to snake_case
    import re
    name = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', model_name)
    name = re.sub('([a-z0-9])([A-Z])', r'\1_\2', name).lower()
    
    # Pluralize
    if name.endswith('y'):
        return name[:-1] + 'ies'
    elif name.endswith('s') or name.endswith('x') or name.endswith('ch'):
        return name + 'es'
    else:
        return name + 's'