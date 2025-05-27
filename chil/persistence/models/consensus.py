"""
Consensus-related database models.
"""

from enum import Enum
from typing import Optional
from datetime import datetime

from sqlalchemy import (
    Column, String, Integer, Float, Boolean, Text, JSON,
    ForeignKey, UniqueConstraint, Index, DateTime
)
from sqlalchemy.dialects.postgresql import UUID, ENUM
from sqlalchemy.orm import relationship

from .base import Base, TimestampMixin, UUIDMixin, get_table_name


class NodeStatus(str, Enum):
    """Node status enumeration."""
    ACTIVE = "active"
    INACTIVE = "inactive"
    FAILED = "failed"
    SYNCING = "syncing"


class ProposalStatus(str, Enum):
    """Proposal status enumeration."""
    PENDING = "pending"
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    TIMEOUT = "timeout"


class ConsensusNode(Base, UUIDMixin, TimestampMixin):
    """Represents a node in the consensus network."""
    
    __tablename__ = get_table_name("ConsensusNode")
    
    # Node identification
    node_id = Column(String(255), unique=True, nullable=False, index=True)
    address = Column(String(255), nullable=False)
    public_key = Column(Text, nullable=True)
    
    # Node status
    status = Column(
        ENUM(NodeStatus, name="node_status"),
        nullable=False,
        default=NodeStatus.ACTIVE
    )
    
    # Consensus metrics
    term = Column(Integer, nullable=False, default=0)
    voted_for = Column(String(255), nullable=True)
    commit_index = Column(Integer, nullable=False, default=0)
    last_applied = Column(Integer, nullable=False, default=0)
    
    # Health metrics
    last_heartbeat = Column(DateTime(timezone=True), nullable=True)
    failure_score = Column(Float, nullable=False, default=0.0)
    
    # Metadata
    metadata = Column(JSON, nullable=True, default=dict)
    
    # Relationships
    proposals = relationship("ConsensusProposal", back_populates="proposer")
    votes = relationship("ConsensusVote", back_populates="voter")
    
    __table_args__ = (
        Index("idx_consensus_nodes_status_heartbeat", "status", "last_heartbeat"),
    )


class ConsensusProposal(Base, UUIDMixin, TimestampMixin):
    """Represents a consensus proposal."""
    
    __tablename__ = get_table_name("ConsensusProposal")
    
    # Proposal identification
    proposal_id = Column(String(255), unique=True, nullable=False, index=True)
    term = Column(Integer, nullable=False, index=True)
    
    # Proposer
    proposer_id = Column(UUID(as_uuid=True), ForeignKey(f"{ConsensusNode.__tablename__}.id"))
    proposer = relationship("ConsensusNode", back_populates="proposals")
    
    # Proposal content
    proposal_type = Column(String(50), nullable=False)
    content = Column(JSON, nullable=False)
    
    # Status
    status = Column(
        ENUM(ProposalStatus, name="proposal_status"),
        nullable=False,
        default=ProposalStatus.PENDING,
        index=True
    )
    
    # Voting
    votes_required = Column(Integer, nullable=False)
    votes_received = Column(Integer, nullable=False, default=0)
    
    # Timing
    expires_at = Column(DateTime(timezone=True), nullable=False)
    decided_at = Column(DateTime(timezone=True), nullable=True)
    
    # Relationships
    votes = relationship("ConsensusVote", back_populates="proposal")
    
    __table_args__ = (
        Index("idx_consensus_proposals_term_status", "term", "status"),
        Index("idx_consensus_proposals_expires", "expires_at"),
    )


class ConsensusVote(Base, UUIDMixin, TimestampMixin):
    """Represents a vote on a consensus proposal."""
    
    __tablename__ = get_table_name("ConsensusVote")
    
    # Vote identification
    proposal_id = Column(UUID(as_uuid=True), ForeignKey(f"{ConsensusProposal.__tablename__}.id"))
    voter_id = Column(UUID(as_uuid=True), ForeignKey(f"{ConsensusNode.__tablename__}.id"))
    
    # Vote
    vote = Column(Boolean, nullable=False)  # True = accept, False = reject
    
    # Metadata
    reason = Column(Text, nullable=True)
    signature = Column(Text, nullable=True)
    
    # Relationships
    proposal = relationship("ConsensusProposal", back_populates="votes")
    voter = relationship("ConsensusNode", back_populates="votes")
    
    __table_args__ = (
        UniqueConstraint("proposal_id", "voter_id", name="uq_consensus_votes_proposal_voter"),
        Index("idx_consensus_votes_proposal", "proposal_id"),
    )


class ConsensusState(Base, UUIDMixin, TimestampMixin):
    """Represents the consensus state log."""
    
    __tablename__ = get_table_name("ConsensusState")
    
    # State identification
    term = Column(Integer, nullable=False)
    index = Column(Integer, nullable=False)
    
    # State content
    state_type = Column(String(50), nullable=False)
    state_data = Column(JSON, nullable=False)
    
    # Consensus info
    leader_id = Column(String(255), nullable=True)
    committed = Column(Boolean, nullable=False, default=False)
    
    # Hash chain
    previous_hash = Column(String(64), nullable=True)
    state_hash = Column(String(64), nullable=False)
    
    __table_args__ = (
        UniqueConstraint("term", "index", name="uq_consensus_state_term_index"),
        Index("idx_consensus_state_committed", "committed", "term", "index"),
    )