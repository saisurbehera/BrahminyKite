"""
Repository for consensus-related database operations.
"""

import logging
from typing import List, Optional, Dict, Any, Tuple
from uuid import UUID
from datetime import datetime, timedelta
import hashlib
import json

from sqlalchemy import select, func, and_, or_, update
from sqlalchemy.ext.asyncio import AsyncSession

from .base import BaseRepository, RepositoryError
from ..models.consensus import (
    ConsensusNode,
    ConsensusProposal,
    ConsensusVote,
    ConsensusState,
    NodeStatus,
    ProposalStatus
)

logger = logging.getLogger(__name__)


class ConsensusNodeRepository(BaseRepository[ConsensusNode]):
    """Repository for consensus nodes."""
    
    def __init__(self, session: AsyncSession):
        super().__init__(ConsensusNode, session)
    
    async def get_by_node_id(self, node_id: str) -> Optional[ConsensusNode]:
        """Get node by its unique node ID."""
        return await self.get_by(node_id=node_id)
    
    async def get_active_nodes(self) -> List[ConsensusNode]:
        """Get all active nodes."""
        return await self.list(
            filters={"status": NodeStatus.ACTIVE},
            order_by="last_heartbeat",
            order_desc=True
        )
    
    async def update_heartbeat(
        self,
        node_id: str,
        failure_score: Optional[float] = None
    ) -> Optional[ConsensusNode]:
        """Update node heartbeat and optionally failure score."""
        node = await self.get_by_node_id(node_id)
        if not node:
            return None
        
        updates = {"last_heartbeat": datetime.utcnow()}
        if failure_score is not None:
            updates["failure_score"] = failure_score
        
        return await self.update(node.id, **updates)
    
    async def mark_failed(self, node_id: str) -> Optional[ConsensusNode]:
        """Mark a node as failed."""
        node = await self.get_by_node_id(node_id)
        if not node:
            return None
        
        return await self.update(
            node.id,
            status=NodeStatus.FAILED,
            failure_score=1.0
        )
    
    async def get_leader(self, term: int) -> Optional[ConsensusNode]:
        """Get the current leader for a term."""
        # In Raft, the leader is determined by elections
        # This would need to check proposals and votes
        result = await self.session.execute(
            select(self.model)
            .join(ConsensusProposal, ConsensusProposal.proposer_id == self.model.id)
            .where(
                and_(
                    ConsensusProposal.term == term,
                    ConsensusProposal.status == ProposalStatus.ACCEPTED,
                    self.model.status == NodeStatus.ACTIVE
                )
            )
            .order_by(ConsensusProposal.created_at.desc())
            .limit(1)
        )
        return result.scalar_one_or_none()
    
    async def get_cluster_health(self) -> Dict[str, Any]:
        """Get overall cluster health metrics."""
        now = datetime.utcnow()
        recent_threshold = now - timedelta(seconds=30)
        
        # Get node counts by status
        status_counts = await self.session.execute(
            select(
                self.model.status,
                func.count(self.model.id).label("count")
            )
            .group_by(self.model.status)
        )
        
        # Get nodes with recent heartbeats
        recent_count = await self.count(
            filters={"last_heartbeat": {"gte": recent_threshold}}
        )
        
        # Get average failure score
        avg_failure = await self.session.execute(
            select(func.avg(self.model.failure_score))
            .where(self.model.status == NodeStatus.ACTIVE)
        )
        
        total_nodes = sum(row.count for row in status_counts)
        active_nodes = next((row.count for row in status_counts if row.status == NodeStatus.ACTIVE), 0)
        
        return {
            "total_nodes": total_nodes,
            "active_nodes": active_nodes,
            "status_distribution": {row.status: row.count for row in status_counts},
            "recent_heartbeats": recent_count,
            "avg_failure_score": float(avg_failure.scalar() or 0),
            "health_percentage": (active_nodes / total_nodes * 100) if total_nodes > 0 else 0
        }


class ConsensusProposalRepository(BaseRepository[ConsensusProposal]):
    """Repository for consensus proposals."""
    
    def __init__(self, session: AsyncSession):
        super().__init__(ConsensusProposal, session)
    
    async def get_by_proposal_id(self, proposal_id: str) -> Optional[ConsensusProposal]:
        """Get proposal by its unique proposal ID."""
        return await self.get_by(proposal_id=proposal_id)
    
    async def get_pending_proposals(
        self,
        term: Optional[int] = None
    ) -> List[ConsensusProposal]:
        """Get pending proposals, optionally filtered by term."""
        filters = {"status": ProposalStatus.PENDING}
        if term is not None:
            filters["term"] = term
        
        return await self.list(
            filters=filters,
            order_by="created_at",
            load_relationships=["proposer", "votes"]
        )
    
    async def get_expired_proposals(self) -> List[ConsensusProposal]:
        """Get proposals that have expired."""
        now = datetime.utcnow()
        
        query = select(self.model).where(
            and_(
                self.model.status == ProposalStatus.PENDING,
                self.model.expires_at < now
            )
        )
        
        result = await self.session.execute(query)
        return list(result.scalars().all())
    
    async def update_vote_count(
        self,
        proposal_id: Union[UUID, str],
        increment: int = 1
    ) -> Optional[ConsensusProposal]:
        """Update the vote count for a proposal."""
        proposal = await self.get(proposal_id)
        if not proposal:
            return None
        
        new_count = proposal.votes_received + increment
        updates = {"votes_received": new_count}
        
        # Check if proposal should be accepted
        if new_count >= proposal.votes_required:
            updates["status"] = ProposalStatus.ACCEPTED
            updates["decided_at"] = datetime.utcnow()
        
        return await self.update(proposal_id, **updates)
    
    async def mark_timeout(self, proposal_ids: List[Union[UUID, str]]) -> int:
        """Mark multiple proposals as timed out."""
        return await self.bulk_update(
            filters={"id": proposal_ids},
            updates={
                "status": ProposalStatus.TIMEOUT,
                "decided_at": datetime.utcnow()
            }
        )
    
    async def get_proposal_success_rate(
        self,
        days: int = 30,
        proposal_type: Optional[str] = None
    ) -> Dict[str, Any]:
        """Get proposal success rate statistics."""
        since = datetime.utcnow() - timedelta(days=days)
        
        query = select(
            self.model.status,
            func.count(self.model.id).label("count")
        ).where(
            self.model.created_at >= since
        )
        
        if proposal_type:
            query = query.where(self.model.proposal_type == proposal_type)
        
        query = query.group_by(self.model.status)
        
        result = await self.session.execute(query)
        
        counts = {row.status: row.count for row in result}
        total = sum(counts.values())
        
        return {
            "period_days": days,
            "proposal_type": proposal_type,
            "total_proposals": total,
            "status_counts": counts,
            "success_rate": (counts.get(ProposalStatus.ACCEPTED, 0) / total) if total > 0 else 0,
            "timeout_rate": (counts.get(ProposalStatus.TIMEOUT, 0) / total) if total > 0 else 0
        }


class ConsensusVoteRepository(BaseRepository[ConsensusVote]):
    """Repository for consensus votes."""
    
    def __init__(self, session: AsyncSession):
        super().__init__(ConsensusVote, session)
    
    async def cast_vote(
        self,
        proposal_id: Union[UUID, str],
        voter_id: Union[UUID, str],
        vote: bool,
        reason: Optional[str] = None,
        signature: Optional[str] = None
    ) -> ConsensusVote:
        """Cast a vote on a proposal."""
        # Check if vote already exists
        existing = await self.get_by(
            proposal_id=proposal_id,
            voter_id=voter_id
        )
        
        if existing:
            raise RepositoryError("Vote already cast for this proposal")
        
        return await self.create(
            proposal_id=proposal_id,
            voter_id=voter_id,
            vote=vote,
            reason=reason,
            signature=signature
        )
    
    async def get_proposal_votes(
        self,
        proposal_id: Union[UUID, str]
    ) -> Tuple[int, int]:
        """Get vote counts for a proposal (accepts, rejects)."""
        votes = await self.list(
            filters={"proposal_id": proposal_id}
        )
        
        accepts = sum(1 for v in votes if v.vote)
        rejects = len(votes) - accepts
        
        return accepts, rejects
    
    async def has_voted(
        self,
        proposal_id: Union[UUID, str],
        voter_id: Union[UUID, str]
    ) -> bool:
        """Check if a node has already voted on a proposal."""
        return await self.exists(
            proposal_id=proposal_id,
            voter_id=voter_id
        )


class ConsensusStateRepository(BaseRepository[ConsensusState]):
    """Repository for consensus state log."""
    
    def __init__(self, session: AsyncSession):
        super().__init__(ConsensusState, session)
    
    async def append_state(
        self,
        term: int,
        index: int,
        state_type: str,
        state_data: Dict[str, Any],
        leader_id: Optional[str] = None,
        previous_hash: Optional[str] = None
    ) -> ConsensusState:
        """Append a new state to the log."""
        # Calculate state hash
        state_str = json.dumps({
            "term": term,
            "index": index,
            "state_type": state_type,
            "state_data": state_data,
            "previous_hash": previous_hash
        }, sort_keys=True)
        
        state_hash = hashlib.sha256(state_str.encode()).hexdigest()
        
        return await self.create(
            term=term,
            index=index,
            state_type=state_type,
            state_data=state_data,
            leader_id=leader_id,
            previous_hash=previous_hash,
            state_hash=state_hash
        )
    
    async def get_latest_state(self) -> Optional[ConsensusState]:
        """Get the latest committed state."""
        query = select(self.model).where(
            self.model.committed == True
        ).order_by(
            self.model.term.desc(),
            self.model.index.desc()
        ).limit(1)
        
        result = await self.session.execute(query)
        return result.scalar_one_or_none()
    
    async def get_uncommitted_states(
        self,
        term: int
    ) -> List[ConsensusState]:
        """Get all uncommitted states for a term."""
        return await self.list(
            filters={
                "term": term,
                "committed": False
            },
            order_by="index"
        )
    
    async def commit_states(
        self,
        term: int,
        up_to_index: int
    ) -> int:
        """Commit states up to a certain index."""
        return await self.bulk_update(
            filters={
                "term": term,
                "index": {"lte": up_to_index},
                "committed": False
            },
            updates={"committed": True}
        )
    
    async def verify_hash_chain(
        self,
        from_index: int = 0,
        to_index: Optional[int] = None
    ) -> bool:
        """Verify the hash chain integrity."""
        query = select(self.model).where(
            self.model.index >= from_index
        )
        
        if to_index is not None:
            query = query.where(self.model.index <= to_index)
        
        query = query.order_by(self.model.index)
        
        result = await self.session.execute(query)
        states = list(result.scalars().all())
        
        if not states:
            return True
        
        # Verify each state's hash
        for i, state in enumerate(states):
            if i == 0 and state.previous_hash is not None:
                # First state should have no previous hash unless continuing from earlier
                continue
            
            if i > 0:
                # Verify previous hash matches
                if state.previous_hash != states[i-1].state_hash:
                    logger.error(
                        f"Hash chain broken at index {state.index}: "
                        f"expected {states[i-1].state_hash}, got {state.previous_hash}"
                    )
                    return False
        
        return True


class ConsensusRepository:
    """
    Composite repository for all consensus-related operations.
    
    Provides a unified interface for consensus data access.
    """
    
    def __init__(self, session: AsyncSession):
        self.session = session
        self.nodes = ConsensusNodeRepository(session)
        self.proposals = ConsensusProposalRepository(session)
        self.votes = ConsensusVoteRepository(session)
        self.states = ConsensusStateRepository(session)
    
    async def register_node(
        self,
        node_id: str,
        address: str,
        public_key: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> ConsensusNode:
        """Register a new node in the consensus network."""
        # Check if node already exists
        existing = await self.nodes.get_by_node_id(node_id)
        if existing:
            # Update existing node
            return await self.nodes.update(
                existing.id,
                address=address,
                public_key=public_key,
                status=NodeStatus.ACTIVE,
                last_heartbeat=datetime.utcnow(),
                failure_score=0.0,
                metadata=metadata or {}
            )
        
        # Create new node
        return await self.nodes.create(
            node_id=node_id,
            address=address,
            public_key=public_key,
            status=NodeStatus.ACTIVE,
            last_heartbeat=datetime.utcnow(),
            metadata=metadata or {}
        )
    
    async def create_proposal(
        self,
        proposal_id: str,
        proposer_node_id: str,
        term: int,
        proposal_type: str,
        content: Dict[str, Any],
        votes_required: int,
        timeout_seconds: int = 300
    ) -> ConsensusProposal:
        """Create a new consensus proposal."""
        # Get proposer node
        proposer = await self.nodes.get_by_node_id(proposer_node_id)
        if not proposer:
            raise RepositoryError(f"Proposer node {proposer_node_id} not found")
        
        if proposer.status != NodeStatus.ACTIVE:
            raise RepositoryError(f"Proposer node {proposer_node_id} is not active")
        
        expires_at = datetime.utcnow() + timedelta(seconds=timeout_seconds)
        
        return await self.proposals.create(
            proposal_id=proposal_id,
            proposer_id=proposer.id,
            term=term,
            proposal_type=proposal_type,
            content=content,
            votes_required=votes_required,
            expires_at=expires_at
        )
    
    async def process_vote(
        self,
        proposal_id: str,
        voter_node_id: str,
        vote: bool,
        reason: Optional[str] = None,
        signature: Optional[str] = None
    ) -> Tuple[ConsensusVote, ConsensusProposal]:
        """
        Process a vote on a proposal.
        
        Returns the vote and updated proposal.
        """
        # Get proposal
        proposal = await self.proposals.get_by_proposal_id(proposal_id)
        if not proposal:
            raise RepositoryError(f"Proposal {proposal_id} not found")
        
        if proposal.status != ProposalStatus.PENDING:
            raise RepositoryError(f"Proposal {proposal_id} is not pending")
        
        # Get voter node
        voter = await self.nodes.get_by_node_id(voter_node_id)
        if not voter:
            raise RepositoryError(f"Voter node {voter_node_id} not found")
        
        if voter.status != NodeStatus.ACTIVE:
            raise RepositoryError(f"Voter node {voter_node_id} is not active")
        
        # Cast vote
        vote_record = await self.votes.cast_vote(
            proposal_id=proposal.id,
            voter_id=voter.id,
            vote=vote,
            reason=reason,
            signature=signature
        )
        
        # Update proposal vote count
        updated_proposal = await self.proposals.update_vote_count(proposal.id)
        
        return vote_record, updated_proposal
    
    async def cleanup_expired_proposals(self) -> int:
        """Clean up expired proposals."""
        expired = await self.proposals.get_expired_proposals()
        
        if not expired:
            return 0
        
        proposal_ids = [p.id for p in expired]
        return await self.proposals.mark_timeout(proposal_ids)