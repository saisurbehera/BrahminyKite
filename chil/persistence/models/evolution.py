"""
Evolutionary framework models for tracking agent evolution and generations.
"""

from enum import Enum
from typing import Optional
from datetime import datetime

from sqlalchemy import (
    Column, String, Integer, Float, Boolean, Text, JSON,
    ForeignKey, UniqueConstraint, Index, DateTime, CheckConstraint
)
from sqlalchemy.dialects.postgresql import UUID, ENUM, ARRAY
from sqlalchemy.orm import relationship

from .base import Base, TimestampMixin, UUIDMixin, get_table_name


class AgentStatus(str, Enum):
    """Agent status in evolutionary system."""
    ACTIVE = "active"
    DORMANT = "dormant"
    EVOLVED = "evolved"
    DEPRECATED = "deprecated"
    TERMINATED = "terminated"


class SelectionStrategy(str, Enum):
    """Selection strategies for evolution."""
    FITNESS_PROPORTIONATE = "fitness_proportionate"
    TOURNAMENT = "tournament"
    RANK_BASED = "rank_based"
    ELITISM = "elitism"
    DIVERSITY_PRESERVING = "diversity_preserving"


class MutationType(str, Enum):
    """Types of mutations."""
    PARAMETER = "parameter"
    STRUCTURAL = "structural"
    BEHAVIORAL = "behavioral"
    CAPABILITY = "capability"
    HYBRID = "hybrid"


class EvolutionaryAgent(Base, UUIDMixin, TimestampMixin):
    """Represents an agent in the evolutionary system."""
    
    __tablename__ = get_table_name("EvolutionaryAgent")
    
    # Agent identification
    agent_identifier = Column(String(255), unique=True, nullable=False, index=True)
    generation_id = Column(UUID(as_uuid=True), ForeignKey("evolution_generations.id"))
    
    # Lineage
    parent_agents = Column(ARRAY(UUID(as_uuid=True)), nullable=True)  # Can have multiple parents
    lineage_depth = Column(Integer, nullable=False, default=0)
    
    # Genetic information
    genome = Column(JSON, nullable=False)  # Agent's configuration/parameters
    phenotype = Column(JSON, nullable=False)  # Expressed characteristics
    mutations = Column(JSON, nullable=False, default=list)  # Applied mutations
    
    # Fitness metrics
    fitness_score = Column(Float, nullable=False, default=0.0)
    normalized_fitness = Column(Float, nullable=True)  # Normalized within generation
    
    # Performance metrics
    task_success_rate = Column(Float, nullable=True)
    adaptability_score = Column(Float, nullable=True)
    innovation_score = Column(Float, nullable=True)
    stability_score = Column(Float, nullable=True)
    
    # Status
    status = Column(
        ENUM(AgentStatus, name="agent_status"),
        nullable=False,
        default=AgentStatus.ACTIVE,
        index=True
    )
    
    # Selection info
    selection_count = Column(Integer, nullable=False, default=0)  # Times selected for breeding
    offspring_count = Column(Integer, nullable=False, default=0)
    
    # Metadata
    capabilities = Column(JSON, nullable=False, default=list)
    metadata = Column(JSON, nullable=True, default=dict)
    
    # Relationships
    generation = relationship("EvolutionGeneration", back_populates="agents")
    metrics = relationship("EvolutionMetrics", back_populates="agent")
    
    __table_args__ = (
        Index("idx_evolutionary_agents_generation_fitness", "generation_id", "fitness_score"),
        Index("idx_evolutionary_agents_status", "status"),
    )


class EvolutionGeneration(Base, UUIDMixin, TimestampMixin):
    """Represents a generation in the evolutionary process."""
    
    __tablename__ = get_table_name("EvolutionGeneration")
    
    # Generation identification
    generation_number = Column(Integer, nullable=False, unique=True)
    experiment_id = Column(String(255), nullable=False, index=True)
    
    # Population info
    population_size = Column(Integer, nullable=False)
    elite_size = Column(Integer, nullable=False, default=0)
    
    # Selection strategy
    selection_strategy = Column(
        ENUM(SelectionStrategy, name="selection_strategy"),
        nullable=False
    )
    selection_params = Column(JSON, nullable=True, default=dict)
    
    # Mutation settings
    mutation_rate = Column(Float, nullable=False, default=0.1)
    mutation_types = Column(
        ARRAY(ENUM(MutationType, name="mutation_type")),
        nullable=False
    )
    
    # Fitness statistics
    avg_fitness = Column(Float, nullable=True)
    max_fitness = Column(Float, nullable=True)
    min_fitness = Column(Float, nullable=True)
    fitness_variance = Column(Float, nullable=True)
    
    # Diversity metrics
    genetic_diversity = Column(Float, nullable=True)
    phenotypic_diversity = Column(Float, nullable=True)
    
    # Progress tracking
    improvement_rate = Column(Float, nullable=True)  # Compared to previous generation
    convergence_score = Column(Float, nullable=True)  # 0.0 to 1.0
    
    # Timing
    started_at = Column(DateTime(timezone=True), nullable=True)
    completed_at = Column(DateTime(timezone=True), nullable=True)
    
    # Relationships
    agents = relationship("EvolutionaryAgent", back_populates="generation")
    state = relationship("EvolutionState", back_populates="generation", uselist=False)
    
    __table_args__ = (
        Index("idx_evolution_generations_experiment", "experiment_id", "generation_number"),
        CheckConstraint("mutation_rate >= 0 AND mutation_rate <= 1", name="check_mutation_rate_range"),
    )


class EvolutionMetrics(Base, UUIDMixin, TimestampMixin):
    """Detailed metrics for evolutionary agents."""
    
    __tablename__ = get_table_name("EvolutionMetrics")
    
    # Metrics identification
    agent_id = Column(UUID(as_uuid=True), ForeignKey(f"{EvolutionaryAgent.__tablename__}.id"))
    evaluation_round = Column(Integer, nullable=False)
    
    # Task performance
    tasks_completed = Column(Integer, nullable=False, default=0)
    tasks_failed = Column(Integer, nullable=False, default=0)
    avg_task_time_ms = Column(Float, nullable=True)
    
    # Behavioral metrics
    exploration_rate = Column(Float, nullable=True)  # Novel actions taken
    exploitation_rate = Column(Float, nullable=True)  # Optimal actions taken
    cooperation_score = Column(Float, nullable=True)
    
    # Resource usage
    compute_efficiency = Column(Float, nullable=True)
    memory_efficiency = Column(Float, nullable=True)
    energy_efficiency = Column(Float, nullable=True)
    
    # Learning metrics
    learning_rate = Column(Float, nullable=True)
    knowledge_retention = Column(Float, nullable=True)
    generalization_score = Column(Float, nullable=True)
    
    # Environmental adaptation
    environment_fitness = Column(JSON, nullable=True)  # Fitness in different environments
    robustness_score = Column(Float, nullable=True)
    
    # Relationships
    agent = relationship("EvolutionaryAgent", back_populates="metrics")
    
    __table_args__ = (
        Index("idx_evolution_metrics_agent_round", "agent_id", "evaluation_round"),
    )


class EvolutionState(Base, UUIDMixin, TimestampMixin):
    """State snapshots for evolutionary experiments."""
    
    __tablename__ = get_table_name("EvolutionState")
    
    # State identification
    generation_id = Column(UUID(as_uuid=True), ForeignKey(f"{EvolutionGeneration.__tablename__}.id"), unique=True)
    experiment_id = Column(String(255), nullable=False)
    
    # Population state
    population_snapshot = Column(JSON, nullable=False)  # Compressed population data
    elite_pool = Column(JSON, nullable=False)  # Top performers
    
    # Evolution parameters
    current_params = Column(JSON, nullable=False)
    param_history = Column(JSON, nullable=False, default=list)
    
    # Progress tracking
    fitness_trajectory = Column(JSON, nullable=False, default=list)
    diversity_trajectory = Column(JSON, nullable=False, default=list)
    
    # Checkpointing
    checkpoint_path = Column(String(500), nullable=True)
    is_recoverable = Column(Boolean, nullable=False, default=True)
    
    # Analysis
    bottlenecks = Column(JSON, nullable=True)  # Identified evolutionary bottlenecks
    breakthrough_events = Column(JSON, nullable=True)  # Significant improvements
    
    # Relationships
    generation = relationship("EvolutionGeneration", back_populates="state")
    
    __table_args__ = (
        Index("idx_evolution_state_experiment", "experiment_id"),
    )