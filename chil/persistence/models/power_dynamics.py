"""
Power dynamics models for tracking agent interactions and influence.
"""

from enum import Enum
from typing import Optional
from datetime import datetime

from sqlalchemy import (
    Column, String, Integer, Float, Boolean, Text, JSON,
    ForeignKey, UniqueConstraint, Index, DateTime, CheckConstraint
)
from sqlalchemy.dialects.postgresql import UUID, ENUM
from sqlalchemy.orm import relationship

from .base import Base, TimestampMixin, UUIDMixin, get_table_name


class NodeType(str, Enum):
    """Types of nodes in the power network."""
    AGENT = "agent"
    HUMAN = "human"
    SYSTEM = "system"
    ORGANIZATION = "organization"


class RelationType(str, Enum):
    """Types of power relationships."""
    HIERARCHICAL = "hierarchical"
    COLLABORATIVE = "collaborative"
    COMPETITIVE = "competitive"
    DEPENDENCY = "dependency"
    INFLUENCE = "influence"


class EventType(str, Enum):
    """Types of power dynamics events."""
    POWER_TRANSFER = "power_transfer"
    COALITION_FORMED = "coalition_formed"
    COALITION_DISSOLVED = "coalition_dissolved"
    CONFLICT_STARTED = "conflict_started"
    CONFLICT_RESOLVED = "conflict_resolved"
    INFLUENCE_CHANGE = "influence_change"


class PowerNode(Base, UUIDMixin, TimestampMixin):
    """Represents an entity in the power dynamics network."""
    
    __tablename__ = get_table_name("PowerNode")
    
    # Node identification
    node_identifier = Column(String(255), unique=True, nullable=False, index=True)
    node_type = Column(
        ENUM(NodeType, name="node_type"),
        nullable=False,
        index=True
    )
    
    # Power metrics
    power_score = Column(Float, nullable=False, default=0.5)  # 0.0 to 1.0
    influence_score = Column(Float, nullable=False, default=0.5)  # 0.0 to 1.0
    reputation_score = Column(Float, nullable=False, default=0.5)  # 0.0 to 1.0
    
    # Resources
    resource_capacity = Column(JSON, nullable=False, default=dict)  # {resource_type: amount}
    resource_consumption = Column(JSON, nullable=False, default=dict)
    
    # Capabilities
    capabilities = Column(JSON, nullable=False, default=list)  # List of capability strings
    specializations = Column(JSON, nullable=False, default=list)
    
    # Status
    is_active = Column(Boolean, nullable=False, default=True)
    last_active_at = Column(DateTime(timezone=True), nullable=True)
    
    # Metadata
    metadata = Column(JSON, nullable=True, default=dict)
    
    # Relationships
    outgoing_relations = relationship(
        "PowerRelation",
        foreign_keys="PowerRelation.source_node_id",
        back_populates="source_node"
    )
    incoming_relations = relationship(
        "PowerRelation",
        foreign_keys="PowerRelation.target_node_id",
        back_populates="target_node"
    )
    metrics = relationship("PowerMetrics", back_populates="node")
    events_initiated = relationship(
        "PowerEvent",
        foreign_keys="PowerEvent.initiator_id",
        back_populates="initiator"
    )
    
    __table_args__ = (
        CheckConstraint("power_score >= 0 AND power_score <= 1", name="check_power_score_range"),
        CheckConstraint("influence_score >= 0 AND influence_score <= 1", name="check_influence_score_range"),
        CheckConstraint("reputation_score >= 0 AND reputation_score <= 1", name="check_reputation_score_range"),
        Index("idx_power_nodes_scores", "power_score", "influence_score"),
    )


class PowerRelation(Base, UUIDMixin, TimestampMixin):
    """Represents a relationship between power nodes."""
    
    __tablename__ = get_table_name("PowerRelation")
    
    # Relationship identification
    source_node_id = Column(UUID(as_uuid=True), ForeignKey(f"{PowerNode.__tablename__}.id"))
    target_node_id = Column(UUID(as_uuid=True), ForeignKey(f"{PowerNode.__tablename__}.id"))
    relation_type = Column(
        ENUM(RelationType, name="relation_type"),
        nullable=False
    )
    
    # Relationship strength
    strength = Column(Float, nullable=False, default=0.5)  # 0.0 to 1.0
    reciprocity = Column(Float, nullable=False, default=0.0)  # -1.0 to 1.0
    
    # Dynamics
    is_active = Column(Boolean, nullable=False, default=True)
    established_at = Column(DateTime(timezone=True), nullable=False)
    dissolved_at = Column(DateTime(timezone=True), nullable=True)
    
    # Influence flow
    influence_weight = Column(Float, nullable=False, default=1.0)
    resource_flow = Column(JSON, nullable=True, default=dict)  # {resource_type: flow_rate}
    
    # Metadata
    metadata = Column(JSON, nullable=True, default=dict)
    
    # Relationships
    source_node = relationship(
        "PowerNode",
        foreign_keys=[source_node_id],
        back_populates="outgoing_relations"
    )
    target_node = relationship(
        "PowerNode",
        foreign_keys=[target_node_id],
        back_populates="incoming_relations"
    )
    
    __table_args__ = (
        UniqueConstraint("source_node_id", "target_node_id", "relation_type", name="uq_power_relations_nodes_type"),
        CheckConstraint("strength >= 0 AND strength <= 1", name="check_relation_strength_range"),
        CheckConstraint("reciprocity >= -1 AND reciprocity <= 1", name="check_reciprocity_range"),
        CheckConstraint("source_node_id != target_node_id", name="check_no_self_relations"),
        Index("idx_power_relations_active_type", "is_active", "relation_type"),
    )


class PowerMetrics(Base, UUIDMixin, TimestampMixin):
    """Time-series metrics for power nodes."""
    
    __tablename__ = get_table_name("PowerMetrics")
    
    # Metrics identification
    node_id = Column(UUID(as_uuid=True), ForeignKey(f"{PowerNode.__tablename__}.id"))
    timestamp = Column(DateTime(timezone=True), nullable=False)
    
    # Power metrics snapshot
    power_score = Column(Float, nullable=False)
    influence_score = Column(Float, nullable=False)
    reputation_score = Column(Float, nullable=False)
    
    # Network metrics
    centrality_score = Column(Float, nullable=True)
    clustering_coefficient = Column(Float, nullable=True)
    betweenness_centrality = Column(Float, nullable=True)
    
    # Activity metrics
    interactions_count = Column(Integer, nullable=False, default=0)
    coalitions_count = Column(Integer, nullable=False, default=0)
    conflicts_count = Column(Integer, nullable=False, default=0)
    
    # Resource metrics
    total_resources = Column(Float, nullable=True)
    resource_efficiency = Column(Float, nullable=True)
    
    # Relationships
    node = relationship("PowerNode", back_populates="metrics")
    
    __table_args__ = (
        Index("idx_power_metrics_node_time", "node_id", "timestamp"),
        Index("idx_power_metrics_time", "timestamp"),
    )


class PowerEvent(Base, UUIDMixin, TimestampMixin):
    """Events in the power dynamics system."""
    
    __tablename__ = get_table_name("PowerEvent")
    
    # Event identification
    event_type = Column(
        ENUM(EventType, name="event_type"),
        nullable=False,
        index=True
    )
    
    # Participants
    initiator_id = Column(UUID(as_uuid=True), ForeignKey(f"{PowerNode.__tablename__}.id"))
    participants = Column(JSON, nullable=False, default=list)  # List of node IDs
    
    # Event details
    event_data = Column(JSON, nullable=False)
    outcome = Column(JSON, nullable=True)
    
    # Impact
    impact_score = Column(Float, nullable=False, default=0.0)  # Magnitude of impact
    affected_nodes = Column(JSON, nullable=False, default=list)  # List of affected node IDs
    
    # Duration
    start_time = Column(DateTime(timezone=True), nullable=False)
    end_time = Column(DateTime(timezone=True), nullable=True)
    
    # Relationships
    initiator = relationship(
        "PowerNode",
        foreign_keys=[initiator_id],
        back_populates="events_initiated"
    )
    
    __table_args__ = (
        Index("idx_power_events_type_time", "event_type", "start_time"),
        Index("idx_power_events_initiator", "initiator_id"),
    )