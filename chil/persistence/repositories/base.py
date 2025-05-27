"""
Base repository pattern implementation.
"""

import logging
from typing import TypeVar, Generic, Type, Optional, List, Dict, Any, Union
from uuid import UUID
from datetime import datetime

from sqlalchemy import select, update, delete, func, and_, or_
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload, joinedload
from sqlalchemy.sql import Select

from ..models.base import Base

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=Base)


class RepositoryError(Exception):
    """Base exception for repository errors."""
    pass


class BaseRepository(Generic[T]):
    """
    Base repository providing common CRUD operations.
    
    Subclasses should implement domain-specific queries and operations.
    """
    
    def __init__(self, model: Type[T], session: AsyncSession):
        self.model = model
        self.session = session
    
    async def get(self, id: Union[UUID, str], load_relationships: List[str] = None) -> Optional[T]:
        """
        Get a single entity by ID.
        
        Args:
            id: Entity ID
            load_relationships: List of relationship names to eagerly load
        
        Returns:
            Entity instance or None if not found
        """
        query = select(self.model).where(self.model.id == id)
        
        if load_relationships:
            for rel in load_relationships:
                query = query.options(selectinload(getattr(self.model, rel)))
        
        result = await self.session.execute(query)
        return result.scalar_one_or_none()
    
    async def get_by(self, **kwargs) -> Optional[T]:
        """
        Get a single entity by field values.
        
        Args:
            **kwargs: Field name and value pairs
        
        Returns:
            Entity instance or None if not found
        """
        conditions = [getattr(self.model, k) == v for k, v in kwargs.items()]
        query = select(self.model).where(and_(*conditions))
        result = await self.session.execute(query)
        return result.scalar_one_or_none()
    
    async def get_many(
        self,
        ids: List[Union[UUID, str]],
        load_relationships: List[str] = None
    ) -> List[T]:
        """
        Get multiple entities by IDs.
        
        Args:
            ids: List of entity IDs
            load_relationships: List of relationship names to eagerly load
        
        Returns:
            List of entity instances
        """
        query = select(self.model).where(self.model.id.in_(ids))
        
        if load_relationships:
            for rel in load_relationships:
                query = query.options(selectinload(getattr(self.model, rel)))
        
        result = await self.session.execute(query)
        return list(result.scalars().all())
    
    async def list(
        self,
        skip: int = 0,
        limit: int = 100,
        filters: Dict[str, Any] = None,
        order_by: str = None,
        order_desc: bool = False,
        load_relationships: List[str] = None
    ) -> List[T]:
        """
        List entities with pagination and filtering.
        
        Args:
            skip: Number of records to skip
            limit: Maximum number of records to return
            filters: Dictionary of field filters
            order_by: Field name to order by
            order_desc: Whether to order descending
            load_relationships: List of relationship names to eagerly load
        
        Returns:
            List of entity instances
        """
        query = select(self.model)
        
        # Apply filters
        if filters:
            conditions = []
            for field, value in filters.items():
                if isinstance(value, list):
                    conditions.append(getattr(self.model, field).in_(value))
                elif isinstance(value, dict):
                    # Handle operators like {"gte": 5, "lte": 10}
                    for op, val in value.items():
                        if op == "gte":
                            conditions.append(getattr(self.model, field) >= val)
                        elif op == "gt":
                            conditions.append(getattr(self.model, field) > val)
                        elif op == "lte":
                            conditions.append(getattr(self.model, field) <= val)
                        elif op == "lt":
                            conditions.append(getattr(self.model, field) < val)
                        elif op == "ne":
                            conditions.append(getattr(self.model, field) != val)
                        elif op == "like":
                            conditions.append(getattr(self.model, field).like(val))
                        elif op == "ilike":
                            conditions.append(getattr(self.model, field).ilike(val))
                else:
                    conditions.append(getattr(self.model, field) == value)
            
            if conditions:
                query = query.where(and_(*conditions))
        
        # Apply ordering
        if order_by:
            order_field = getattr(self.model, order_by)
            if order_desc:
                query = query.order_by(order_field.desc())
            else:
                query = query.order_by(order_field)
        else:
            # Default ordering by created_at if available
            if hasattr(self.model, "created_at"):
                query = query.order_by(self.model.created_at.desc())
        
        # Apply eager loading
        if load_relationships:
            for rel in load_relationships:
                query = query.options(selectinload(getattr(self.model, rel)))
        
        # Apply pagination
        query = query.offset(skip).limit(limit)
        
        result = await self.session.execute(query)
        return list(result.scalars().all())
    
    async def count(self, filters: Dict[str, Any] = None) -> int:
        """
        Count entities matching filters.
        
        Args:
            filters: Dictionary of field filters
        
        Returns:
            Count of matching entities
        """
        query = select(func.count()).select_from(self.model)
        
        if filters:
            conditions = [getattr(self.model, k) == v for k, v in filters.items()]
            query = query.where(and_(*conditions))
        
        result = await self.session.execute(query)
        return result.scalar()
    
    async def create(self, **kwargs) -> T:
        """
        Create a new entity.
        
        Args:
            **kwargs: Entity field values
        
        Returns:
            Created entity instance
        """
        entity = self.model(**kwargs)
        self.session.add(entity)
        await self.session.flush()
        return entity
    
    async def bulk_create(self, entities: List[Dict[str, Any]]) -> List[T]:
        """
        Create multiple entities.
        
        Args:
            entities: List of entity field dictionaries
        
        Returns:
            List of created entity instances
        """
        instances = [self.model(**entity) for entity in entities]
        self.session.add_all(instances)
        await self.session.flush()
        return instances
    
    async def update(self, id: Union[UUID, str], **kwargs) -> Optional[T]:
        """
        Update an entity by ID.
        
        Args:
            id: Entity ID
            **kwargs: Fields to update
        
        Returns:
            Updated entity instance or None if not found
        """
        entity = await self.get(id)
        if not entity:
            return None
        
        for key, value in kwargs.items():
            setattr(entity, key, value)
        
        await self.session.flush()
        return entity
    
    async def bulk_update(
        self,
        filters: Dict[str, Any],
        updates: Dict[str, Any]
    ) -> int:
        """
        Update multiple entities matching filters.
        
        Args:
            filters: Dictionary of field filters
            updates: Dictionary of fields to update
        
        Returns:
            Number of updated entities
        """
        stmt = update(self.model)
        
        conditions = [getattr(self.model, k) == v for k, v in filters.items()]
        if conditions:
            stmt = stmt.where(and_(*conditions))
        
        stmt = stmt.values(**updates)
        
        result = await self.session.execute(stmt)
        return result.rowcount
    
    async def delete(self, id: Union[UUID, str]) -> bool:
        """
        Delete an entity by ID.
        
        Args:
            id: Entity ID
        
        Returns:
            True if deleted, False if not found
        """
        entity = await self.get(id)
        if not entity:
            return False
        
        await self.session.delete(entity)
        await self.session.flush()
        return True
    
    async def bulk_delete(self, filters: Dict[str, Any]) -> int:
        """
        Delete multiple entities matching filters.
        
        Args:
            filters: Dictionary of field filters
        
        Returns:
            Number of deleted entities
        """
        stmt = delete(self.model)
        
        conditions = [getattr(self.model, k) == v for k, v in filters.items()]
        if conditions:
            stmt = stmt.where(and_(*conditions))
        
        result = await self.session.execute(stmt)
        return result.rowcount
    
    async def exists(self, **kwargs) -> bool:
        """
        Check if an entity exists with given field values.
        
        Args:
            **kwargs: Field name and value pairs
        
        Returns:
            True if exists, False otherwise
        """
        entity = await self.get_by(**kwargs)
        return entity is not None
    
    async def refresh(self, entity: T) -> T:
        """
        Refresh an entity from the database.
        
        Args:
            entity: Entity instance to refresh
        
        Returns:
            Refreshed entity instance
        """
        await self.session.refresh(entity)
        return entity
    
    def build_query(self) -> Select:
        """
        Build a base query for the model.
        
        Subclasses can override this to add common filters or joins.
        
        Returns:
            SQLAlchemy Select query
        """
        return select(self.model)