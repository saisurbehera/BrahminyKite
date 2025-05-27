"""
Repository for power dynamics database operations.
"""

import logging
from typing import List, Optional, Dict, Any, Tuple, Set
from uuid import UUID
from datetime import datetime, timedelta
from collections import defaultdict

from sqlalchemy import select, func, and_, or_, exists
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from .base import BaseRepository, RepositoryError
from ..models.power_dynamics import (
    PowerNode,
    PowerRelation,
    PowerMetrics,
    PowerEvent,
    NodeType,
    RelationType,
    EventType
)

logger = logging.getLogger(__name__)


class PowerNodeRepository(BaseRepository[PowerNode]):
    """Repository for power nodes."""
    
    def __init__(self, session: AsyncSession):
        super().__init__(PowerNode, session)
    
    async def get_by_identifier(self, node_identifier: str) -> Optional[PowerNode]:
        """Get node by its unique identifier."""
        return await self.get_by(node_identifier=node_identifier)
    
    async def create_node(
        self,
        node_identifier: str,
        node_type: NodeType,
        power_score: float = 0.5,
        influence_score: float = 0.5,
        reputation_score: float = 0.5,
        resource_capacity: Optional[Dict[str, float]] = None,
        capabilities: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> PowerNode:
        """Create a new power node."""
        # Check if node already exists
        existing = await self.get_by_identifier(node_identifier)
        if existing:
            raise RepositoryError(f"Node {node_identifier} already exists")
        
        return await self.create(
            node_identifier=node_identifier,
            node_type=node_type,
            power_score=power_score,
            influence_score=influence_score,
            reputation_score=reputation_score,
            resource_capacity=resource_capacity or {},
            resource_consumption={},
            capabilities=capabilities or [],
            specializations=[],
            is_active=True,
            last_active_at=datetime.utcnow(),
            metadata=metadata or {}
        )
    
    async def update_scores(
        self,
        node_id: Union[UUID, str],
        power_score: Optional[float] = None,
        influence_score: Optional[float] = None,
        reputation_score: Optional[float] = None
    ) -> Optional[PowerNode]:
        """Update node power scores."""
        updates = {}
        if power_score is not None:
            updates["power_score"] = max(0.0, min(1.0, power_score))
        if influence_score is not None:
            updates["influence_score"] = max(0.0, min(1.0, influence_score))
        if reputation_score is not None:
            updates["reputation_score"] = max(0.0, min(1.0, reputation_score))
        
        if updates:
            updates["last_active_at"] = datetime.utcnow()
            return await self.update(node_id, **updates)
        
        return await self.get(node_id)
    
    async def update_resources(
        self,
        node_id: Union[UUID, str],
        resource_capacity: Optional[Dict[str, float]] = None,
        resource_consumption: Optional[Dict[str, float]] = None
    ) -> Optional[PowerNode]:
        """Update node resource information."""
        node = await self.get(node_id)
        if not node:
            return None
        
        updates = {"last_active_at": datetime.utcnow()}
        
        if resource_capacity is not None:
            updates["resource_capacity"] = resource_capacity
        
        if resource_consumption is not None:
            updates["resource_consumption"] = resource_consumption
        
        return await self.update(node_id, **updates)
    
    async def get_active_nodes(
        self,
        node_type: Optional[NodeType] = None,
        min_power_score: Optional[float] = None
    ) -> List[PowerNode]:
        """Get active nodes with optional filtering."""
        filters = {"is_active": True}
        
        if node_type:
            filters["node_type"] = node_type
        
        if min_power_score is not None:
            filters["power_score"] = {"gte": min_power_score}
        
        return await self.list(
            filters=filters,
            order_by="power_score",
            order_desc=True,
            load_relationships=["outgoing_relations", "incoming_relations"]
        )
    
    async def get_top_nodes(
        self,
        metric: str = "power_score",
        limit: int = 10,
        node_type: Optional[NodeType] = None
    ) -> List[PowerNode]:
        """Get top nodes by a specific metric."""
        filters = {"is_active": True}
        if node_type:
            filters["node_type"] = node_type
        
        return await self.list(
            filters=filters,
            order_by=metric,
            order_desc=True,
            limit=limit
        )
    
    async def deactivate_node(
        self,
        node_id: Union[UUID, str]
    ) -> Optional[PowerNode]:
        """Deactivate a power node."""
        return await self.update(
            node_id,
            is_active=False,
            last_active_at=datetime.utcnow()
        )


class PowerRelationRepository(BaseRepository[PowerRelation]):
    """Repository for power relations."""
    
    def __init__(self, session: AsyncSession):
        super().__init__(PowerRelation, session)
    
    async def create_relation(
        self,
        source_node_id: Union[UUID, str],
        target_node_id: Union[UUID, str],
        relation_type: RelationType,
        strength: float = 0.5,
        reciprocity: float = 0.0,
        influence_weight: float = 1.0,
        resource_flow: Optional[Dict[str, float]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> PowerRelation:
        """Create a new power relation."""
        # Check if relation already exists
        existing = await self.get_by(
            source_node_id=source_node_id,
            target_node_id=target_node_id,
            relation_type=relation_type
        )
        
        if existing:
            if existing.is_active:
                raise RepositoryError("Relation already exists")
            else:
                # Reactivate existing relation
                return await self.update(
                    existing.id,
                    is_active=True,
                    strength=strength,
                    reciprocity=reciprocity,
                    influence_weight=influence_weight,
                    resource_flow=resource_flow or {},
                    established_at=datetime.utcnow(),
                    dissolved_at=None,
                    metadata=metadata or {}
                )
        
        return await self.create(
            source_node_id=source_node_id,
            target_node_id=target_node_id,
            relation_type=relation_type,
            strength=strength,
            reciprocity=reciprocity,
            is_active=True,
            established_at=datetime.utcnow(),
            influence_weight=influence_weight,
            resource_flow=resource_flow or {},
            metadata=metadata or {}
        )
    
    async def dissolve_relation(
        self,
        relation_id: Union[UUID, str]
    ) -> Optional[PowerRelation]:
        """Dissolve a power relation."""
        return await self.update(
            relation_id,
            is_active=False,
            dissolved_at=datetime.utcnow()
        )
    
    async def update_relation_strength(
        self,
        relation_id: Union[UUID, str],
        strength: float,
        reciprocity: Optional[float] = None
    ) -> Optional[PowerRelation]:
        """Update relation strength and reciprocity."""
        updates = {"strength": max(0.0, min(1.0, strength))}
        
        if reciprocity is not None:
            updates["reciprocity"] = max(-1.0, min(1.0, reciprocity))
        
        return await self.update(relation_id, **updates)
    
    async def get_node_relations(
        self,
        node_id: Union[UUID, str],
        relation_type: Optional[RelationType] = None,
        direction: str = "both"  # "outgoing", "incoming", or "both"
    ) -> List[PowerRelation]:
        """Get all relations for a node."""
        conditions = [self.model.is_active == True]
        
        if direction in ["outgoing", "both"]:
            conditions.append(self.model.source_node_id == node_id)
        
        if direction in ["incoming", "both"]:
            conditions.append(self.model.target_node_id == node_id)
        
        if relation_type:
            conditions.append(self.model.relation_type == relation_type)
        
        query = select(self.model).where(
            or_(*conditions) if direction == "both" else and_(*conditions)
        )
        
        # Load relationships
        query = query.options(
            selectinload(self.model.source_node),
            selectinload(self.model.target_node)
        )
        
        result = await self.session.execute(query)
        return list(result.scalars().all())
    
    async def get_network_density(
        self,
        node_ids: Optional[List[Union[UUID, str]]] = None
    ) -> float:
        """Calculate network density (ratio of actual to possible connections)."""
        if node_ids:
            # Count nodes
            node_count = len(node_ids)
            
            # Count active relations between these nodes
            relation_count = await self.session.execute(
                select(func.count(self.model.id))
                .where(
                    and_(
                        self.model.is_active == True,
                        self.model.source_node_id.in_(node_ids),
                        self.model.target_node_id.in_(node_ids)
                    )
                )
            )
        else:
            # Count all active nodes
            node_count = await self.session.execute(
                select(func.count(PowerNode.id))
                .where(PowerNode.is_active == True)
            )
            node_count = node_count.scalar()
            
            # Count all active relations
            relation_count = await self.count(filters={"is_active": True})
        
        # Calculate density
        if node_count <= 1:
            return 0.0
        
        max_relations = node_count * (node_count - 1)  # Directed graph
        actual_relations = relation_count.scalar() if hasattr(relation_count, 'scalar') else relation_count
        
        return actual_relations / max_relations if max_relations > 0 else 0.0


class PowerMetricsRepository(BaseRepository[PowerMetrics]):
    """Repository for power metrics."""
    
    def __init__(self, session: AsyncSession):
        super().__init__(PowerMetrics, session)
    
    async def record_metrics(
        self,
        node_id: Union[UUID, str],
        power_score: float,
        influence_score: float,
        reputation_score: float,
        centrality_score: Optional[float] = None,
        clustering_coefficient: Optional[float] = None,
        betweenness_centrality: Optional[float] = None,
        interactions_count: int = 0,
        coalitions_count: int = 0,
        conflicts_count: int = 0,
        total_resources: Optional[float] = None,
        resource_efficiency: Optional[float] = None
    ) -> PowerMetrics:
        """Record metrics snapshot for a node."""
        return await self.create(
            node_id=node_id,
            timestamp=datetime.utcnow(),
            power_score=power_score,
            influence_score=influence_score,
            reputation_score=reputation_score,
            centrality_score=centrality_score,
            clustering_coefficient=clustering_coefficient,
            betweenness_centrality=betweenness_centrality,
            interactions_count=interactions_count,
            coalitions_count=coalitions_count,
            conflicts_count=conflicts_count,
            total_resources=total_resources,
            resource_efficiency=resource_efficiency
        )
    
    async def get_node_metrics_history(
        self,
        node_id: Union[UUID, str],
        hours: int = 24,
        limit: Optional[int] = None
    ) -> List[PowerMetrics]:
        """Get metrics history for a node."""
        since = datetime.utcnow() - timedelta(hours=hours)
        
        return await self.list(
            filters={
                "node_id": node_id,
                "timestamp": {"gte": since}
            },
            order_by="timestamp",
            order_desc=True,
            limit=limit
        )
    
    async def get_power_trends(
        self,
        node_id: Union[UUID, str],
        days: int = 7
    ) -> List[Dict[str, Any]]:
        """Get power score trends for a node."""
        since = datetime.utcnow() - timedelta(days=days)
        
        result = await self.session.execute(
            select(
                func.date_trunc('hour', self.model.timestamp).label("hour"),
                func.avg(self.model.power_score).label("avg_power"),
                func.avg(self.model.influence_score).label("avg_influence"),
                func.avg(self.model.reputation_score).label("avg_reputation")
            )
            .where(
                and_(
                    self.model.node_id == node_id,
                    self.model.timestamp >= since
                )
            )
            .group_by(func.date_trunc('hour', self.model.timestamp))
            .order_by(func.date_trunc('hour', self.model.timestamp))
        )
        
        return [
            {
                "timestamp": row.hour.isoformat(),
                "power_score": float(row.avg_power),
                "influence_score": float(row.avg_influence),
                "reputation_score": float(row.avg_reputation)
            }
            for row in result
        ]


class PowerEventRepository(BaseRepository[PowerEvent]):
    """Repository for power events."""
    
    def __init__(self, session: AsyncSession):
        super().__init__(PowerEvent, session)
    
    async def record_event(
        self,
        event_type: EventType,
        initiator_id: Union[UUID, str],
        participants: List[Union[UUID, str]],
        event_data: Dict[str, Any],
        impact_score: float = 0.0,
        affected_nodes: Optional[List[Union[UUID, str]]] = None
    ) -> PowerEvent:
        """Record a power dynamics event."""
        return await self.create(
            event_type=event_type,
            initiator_id=initiator_id,
            participants=participants,
            event_data=event_data,
            outcome=None,
            impact_score=impact_score,
            affected_nodes=affected_nodes or [],
            start_time=datetime.utcnow()
        )
    
    async def complete_event(
        self,
        event_id: Union[UUID, str],
        outcome: Dict[str, Any],
        impact_score: Optional[float] = None,
        affected_nodes: Optional[List[Union[UUID, str]]] = None
    ) -> Optional[PowerEvent]:
        """Complete an ongoing event."""
        updates = {
            "outcome": outcome,
            "end_time": datetime.utcnow()
        }
        
        if impact_score is not None:
            updates["impact_score"] = impact_score
        
        if affected_nodes is not None:
            updates["affected_nodes"] = affected_nodes
        
        return await self.update(event_id, **updates)
    
    async def get_recent_events(
        self,
        event_type: Optional[EventType] = None,
        hours: int = 24,
        min_impact: Optional[float] = None
    ) -> List[PowerEvent]:
        """Get recent events with optional filtering."""
        since = datetime.utcnow() - timedelta(hours=hours)
        
        conditions = [self.model.start_time >= since]
        
        if event_type:
            conditions.append(self.model.event_type == event_type)
        
        if min_impact is not None:
            conditions.append(self.model.impact_score >= min_impact)
        
        query = select(self.model).where(and_(*conditions))
        query = query.options(selectinload(self.model.initiator))
        query = query.order_by(self.model.start_time.desc())
        
        result = await self.session.execute(query)
        return list(result.scalars().all())
    
    async def get_node_events(
        self,
        node_id: Union[UUID, str],
        as_initiator: bool = True,
        as_participant: bool = True,
        limit: int = 50
    ) -> List[PowerEvent]:
        """Get events involving a specific node."""
        conditions = []
        
        if as_initiator:
            conditions.append(self.model.initiator_id == node_id)
        
        if as_participant:
            # This requires checking the JSON array
            conditions.append(
                func.jsonb_contains(
                    self.model.participants,
                    func.to_jsonb(str(node_id))
                )
            )
        
        if not conditions:
            return []
        
        query = select(self.model).where(
            or_(*conditions) if len(conditions) > 1 else conditions[0]
        )
        query = query.order_by(self.model.start_time.desc())
        query = query.limit(limit)
        
        result = await self.session.execute(query)
        return list(result.scalars().all())


class PowerDynamicsRepository:
    """
    Composite repository for all power dynamics operations.
    
    Provides a unified interface for power dynamics data access.
    """
    
    def __init__(self, session: AsyncSession):
        self.session = session
        self.nodes = PowerNodeRepository(session)
        self.relations = PowerRelationRepository(session)
        self.metrics = PowerMetricsRepository(session)
        self.events = PowerEventRepository(session)
    
    async def create_power_node(
        self,
        node_identifier: str,
        node_type: NodeType,
        **kwargs
    ) -> PowerNode:
        """Create a new power node with initial metrics."""
        # Create node
        node = await self.nodes.create_node(
            node_identifier=node_identifier,
            node_type=node_type,
            **kwargs
        )
        
        # Record initial metrics
        await self.metrics.record_metrics(
            node_id=node.id,
            power_score=node.power_score,
            influence_score=node.influence_score,
            reputation_score=node.reputation_score
        )
        
        return node
    
    async def establish_relation(
        self,
        source_identifier: str,
        target_identifier: str,
        relation_type: RelationType,
        **kwargs
    ) -> PowerRelation:
        """Establish a relation between nodes."""
        # Get nodes
        source = await self.nodes.get_by_identifier(source_identifier)
        target = await self.nodes.get_by_identifier(target_identifier)
        
        if not source or not target:
            raise RepositoryError("One or both nodes not found")
        
        # Create relation
        relation = await self.relations.create_relation(
            source_node_id=source.id,
            target_node_id=target.id,
            relation_type=relation_type,
            **kwargs
        )
        
        # Record event
        await self.events.record_event(
            event_type=EventType.INFLUENCE_CHANGE,
            initiator_id=source.id,
            participants=[target.id],
            event_data={
                "action": "relation_established",
                "relation_type": relation_type,
                "strength": relation.strength
            },
            impact_score=relation.strength
        )
        
        return relation
    
    async def calculate_network_metrics(
        self,
        node_identifier: str
    ) -> Dict[str, float]:
        """Calculate network metrics for a node."""
        node = await self.nodes.get_by_identifier(node_identifier)
        if not node:
            raise RepositoryError(f"Node {node_identifier} not found")
        
        # Get all relations
        relations = await self.relations.get_node_relations(node.id)
        
        # Calculate metrics
        outgoing = [r for r in relations if r.source_node_id == node.id]
        incoming = [r for r in relations if r.target_node_id == node.id]
        
        # Degree centrality
        total_nodes = await self.nodes.count(filters={"is_active": True})
        degree_centrality = len(relations) / (total_nodes - 1) if total_nodes > 1 else 0
        
        # Average relation strength
        avg_strength = sum(r.strength for r in relations) / len(relations) if relations else 0
        
        # Influence flow
        influence_in = sum(r.influence_weight for r in incoming)
        influence_out = sum(r.influence_weight for r in outgoing)
        
        return {
            "degree_centrality": degree_centrality,
            "in_degree": len(incoming),
            "out_degree": len(outgoing),
            "avg_relation_strength": avg_strength,
            "influence_inflow": influence_in,
            "influence_outflow": influence_out,
            "net_influence": influence_in - influence_out
        }
    
    async def form_coalition(
        self,
        initiator_identifier: str,
        member_identifiers: List[str],
        coalition_data: Dict[str, Any]
    ) -> PowerEvent:
        """Form a coalition between nodes."""
        # Get initiator
        initiator = await self.nodes.get_by_identifier(initiator_identifier)
        if not initiator:
            raise RepositoryError(f"Initiator {initiator_identifier} not found")
        
        # Get members
        members = []
        for identifier in member_identifiers:
            member = await self.nodes.get_by_identifier(identifier)
            if member:
                members.append(member)
        
        if len(members) < len(member_identifiers):
            logger.warning("Some coalition members not found")
        
        member_ids = [m.id for m in members]
        
        # Create collaborative relations between all members
        for i, source in enumerate(members):
            for target in members[i+1:]:
                await self.relations.create_relation(
                    source_node_id=source.id,
                    target_node_id=target.id,
                    relation_type=RelationType.COLLABORATIVE,
                    strength=0.7,
                    reciprocity=0.8,
                    metadata={"coalition": coalition_data.get("name", "unnamed")}
                )
        
        # Record coalition event
        event = await self.events.record_event(
            event_type=EventType.COALITION_FORMED,
            initiator_id=initiator.id,
            participants=member_ids,
            event_data=coalition_data,
            impact_score=len(members) / total_nodes if (total_nodes := await self.nodes.count(filters={"is_active": True})) > 0 else 0,
            affected_nodes=member_ids
        )
        
        # Update coalition counts in metrics
        for member in members:
            latest_metrics = await self.metrics.get_node_metrics_history(
                member.id,
                hours=1,
                limit=1
            )
            
            coalitions_count = latest_metrics[0].coalitions_count + 1 if latest_metrics else 1
            
            await self.metrics.record_metrics(
                node_id=member.id,
                power_score=member.power_score,
                influence_score=member.influence_score,
                reputation_score=member.reputation_score,
                coalitions_count=coalitions_count
            )
        
        return event
    
    async def simulate_power_transfer(
        self,
        from_identifier: str,
        to_identifier: str,
        power_amount: float,
        transfer_data: Dict[str, Any]
    ) -> Tuple[PowerNode, PowerNode, PowerEvent]:
        """Simulate a power transfer between nodes."""
        # Get nodes
        from_node = await self.nodes.get_by_identifier(from_identifier)
        to_node = await self.nodes.get_by_identifier(to_identifier)
        
        if not from_node or not to_node:
            raise RepositoryError("One or both nodes not found")
        
        # Calculate new power scores
        transfer_amount = min(power_amount, from_node.power_score * 0.5)  # Max 50% transfer
        new_from_power = from_node.power_score - transfer_amount
        new_to_power = min(1.0, to_node.power_score + transfer_amount * 0.8)  # 80% efficiency
        
        # Update power scores
        from_node = await self.nodes.update_scores(
            from_node.id,
            power_score=new_from_power
        )
        
        to_node = await self.nodes.update_scores(
            to_node.id,
            power_score=new_to_power
        )
        
        # Record event
        event = await self.events.record_event(
            event_type=EventType.POWER_TRANSFER,
            initiator_id=from_node.id,
            participants=[to_node.id],
            event_data={
                **transfer_data,
                "amount": transfer_amount,
                "efficiency": 0.8
            },
            impact_score=transfer_amount,
            affected_nodes=[from_node.id, to_node.id]
        )
        
        # Update metrics
        for node in [from_node, to_node]:
            await self.metrics.record_metrics(
                node_id=node.id,
                power_score=node.power_score,
                influence_score=node.influence_score,
                reputation_score=node.reputation_score,
                interactions_count=1
            )
        
        return from_node, to_node, event