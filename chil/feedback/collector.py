"""
Feedback Collection System

Handles collection and storage of human feedback on verification results.
"""

import uuid
import logging
import json
from datetime import datetime
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict, field
from enum import Enum
from pathlib import Path
import asyncio
import aiofiles


class FeedbackType(Enum):
    """Types of feedback that can be provided"""
    ACCURACY = "accuracy"  # Is the verification result accurate?
    COMPLETENESS = "completeness"  # Is the analysis complete?
    RELEVANCE = "relevance"  # Are the factors considered relevant?
    CLARITY = "clarity"  # Is the reasoning clear?
    CORRECTION = "correction"  # Correcting a specific error
    SUGGESTION = "suggestion"  # General improvement suggestion
    CONFIRMATION = "confirmation"  # Confirming result is correct
    DISPUTE = "dispute"  # Disputing the result


@dataclass
class FeedbackItem:
    """Individual feedback item"""
    feedback_id: str
    verification_id: str  # ID of the verification this feedback relates to
    feedback_type: FeedbackType
    timestamp: datetime
    
    # Feedback content
    rating: Optional[float] = None  # 0.0 to 1.0
    text: Optional[str] = None
    corrections: Optional[Dict[str, Any]] = None
    
    # Metadata
    user_id: Optional[str] = None
    user_expertise: Optional[str] = None  # Domain expertise level
    confidence: float = 1.0  # Confidence in feedback
    
    # Verification context
    claim_text: Optional[str] = None
    verification_result: Optional[Dict[str, Any]] = None
    framework_scores: Optional[Dict[str, float]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage"""
        data = asdict(self)
        data['feedback_type'] = self.feedback_type.value
        data['timestamp'] = self.timestamp.isoformat()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'FeedbackItem':
        """Create from dictionary"""
        data['feedback_type'] = FeedbackType(data['feedback_type'])
        data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        return cls(**data)


@dataclass
class FeedbackSession:
    """A feedback session for batch feedback collection"""
    session_id: str
    start_time: datetime
    end_time: Optional[datetime] = None
    user_id: Optional[str] = None
    feedback_items: List[FeedbackItem] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def add_feedback(self, feedback: FeedbackItem):
        """Add feedback to session"""
        self.feedback_items.append(feedback)
    
    def close(self):
        """Close the session"""
        self.end_time = datetime.utcnow()
    
    @property
    def duration(self) -> Optional[float]:
        """Get session duration in seconds"""
        if self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return None


class FeedbackCollector:
    """Collects and manages human feedback"""
    
    def __init__(self, storage_path: Path):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(__name__)
        
        # Active sessions
        self.active_sessions: Dict[str, FeedbackSession] = {}
        
        # Feedback index for quick lookup
        self.feedback_index: Dict[str, List[str]] = {}  # verification_id -> feedback_ids
        self._load_index()
        
        # Metrics
        self.metrics = {
            "total_feedback": 0,
            "feedback_by_type": {t.value: 0 for t in FeedbackType},
            "average_rating": 0.0,
            "total_sessions": 0
        }
        self._load_metrics()
    
    def _load_index(self):
        """Load feedback index from disk"""
        index_file = self.storage_path / "feedback_index.json"
        if index_file.exists():
            try:
                with open(index_file, 'r') as f:
                    self.feedback_index = json.load(f)
            except Exception as e:
                self.logger.error(f"Failed to load feedback index: {e}")
    
    def _save_index(self):
        """Save feedback index to disk"""
        index_file = self.storage_path / "feedback_index.json"
        try:
            with open(index_file, 'w') as f:
                json.dump(self.feedback_index, f, indent=2)
        except Exception as e:
            self.logger.error(f"Failed to save feedback index: {e}")
    
    def _load_metrics(self):
        """Load metrics from disk"""
        metrics_file = self.storage_path / "metrics.json"
        if metrics_file.exists():
            try:
                with open(metrics_file, 'r') as f:
                    self.metrics = json.load(f)
            except Exception as e:
                self.logger.error(f"Failed to load metrics: {e}")
    
    def _save_metrics(self):
        """Save metrics to disk"""
        metrics_file = self.storage_path / "metrics.json"
        try:
            with open(metrics_file, 'w') as f:
                json.dump(self.metrics, f, indent=2)
        except Exception as e:
            self.logger.error(f"Failed to save metrics: {e}")
    
    def start_session(self, user_id: Optional[str] = None) -> FeedbackSession:
        """Start a new feedback session"""
        session = FeedbackSession(
            session_id=str(uuid.uuid4()),
            start_time=datetime.utcnow(),
            user_id=user_id
        )
        self.active_sessions[session.session_id] = session
        self.metrics["total_sessions"] += 1
        return session
    
    def end_session(self, session_id: str) -> Optional[FeedbackSession]:
        """End a feedback session"""
        if session_id in self.active_sessions:
            session = self.active_sessions.pop(session_id)
            session.close()
            
            # Save session data
            asyncio.create_task(self._save_session(session))
            
            return session
        return None
    
    async def collect_feedback(
        self,
        verification_id: str,
        feedback_type: FeedbackType,
        rating: Optional[float] = None,
        text: Optional[str] = None,
        corrections: Optional[Dict[str, Any]] = None,
        user_id: Optional[str] = None,
        user_expertise: Optional[str] = None,
        session_id: Optional[str] = None,
        verification_context: Optional[Dict[str, Any]] = None
    ) -> FeedbackItem:
        """Collect a piece of feedback"""
        
        # Create feedback item
        feedback = FeedbackItem(
            feedback_id=str(uuid.uuid4()),
            verification_id=verification_id,
            feedback_type=feedback_type,
            timestamp=datetime.utcnow(),
            rating=rating,
            text=text,
            corrections=corrections,
            user_id=user_id,
            user_expertise=user_expertise
        )
        
        # Add verification context if provided
        if verification_context:
            feedback.claim_text = verification_context.get("claim_text")
            feedback.verification_result = verification_context.get("result")
            feedback.framework_scores = verification_context.get("framework_scores")
        
        # Add to session if provided
        if session_id and session_id in self.active_sessions:
            self.active_sessions[session_id].add_feedback(feedback)
        
        # Save feedback
        await self._save_feedback(feedback)
        
        # Update index
        if verification_id not in self.feedback_index:
            self.feedback_index[verification_id] = []
        self.feedback_index[verification_id].append(feedback.feedback_id)
        self._save_index()
        
        # Update metrics
        self._update_metrics(feedback)
        
        self.logger.info(f"Collected {feedback_type.value} feedback for verification {verification_id}")
        
        return feedback
    
    async def _save_feedback(self, feedback: FeedbackItem):
        """Save feedback to disk"""
        feedback_file = self.storage_path / f"feedback_{feedback.feedback_id}.json"
        
        try:
            async with aiofiles.open(feedback_file, 'w') as f:
                await f.write(json.dumps(feedback.to_dict(), indent=2))
        except Exception as e:
            self.logger.error(f"Failed to save feedback: {e}")
    
    async def _save_session(self, session: FeedbackSession):
        """Save session data"""
        session_file = self.storage_path / f"session_{session.session_id}.json"
        
        session_data = {
            "session_id": session.session_id,
            "start_time": session.start_time.isoformat(),
            "end_time": session.end_time.isoformat() if session.end_time else None,
            "user_id": session.user_id,
            "feedback_items": [f.to_dict() for f in session.feedback_items],
            "metadata": session.metadata,
            "duration": session.duration
        }
        
        try:
            async with aiofiles.open(session_file, 'w') as f:
                await f.write(json.dumps(session_data, indent=2))
        except Exception as e:
            self.logger.error(f"Failed to save session: {e}")
    
    def _update_metrics(self, feedback: FeedbackItem):
        """Update metrics with new feedback"""
        self.metrics["total_feedback"] += 1
        self.metrics["feedback_by_type"][feedback.feedback_type.value] += 1
        
        # Update average rating
        if feedback.rating is not None:
            current_avg = self.metrics["average_rating"]
            total = self.metrics["total_feedback"]
            new_avg = ((current_avg * (total - 1)) + feedback.rating) / total
            self.metrics["average_rating"] = new_avg
        
        self._save_metrics()
    
    async def get_feedback_for_verification(self, verification_id: str) -> List[FeedbackItem]:
        """Get all feedback for a specific verification"""
        feedback_items = []
        
        feedback_ids = self.feedback_index.get(verification_id, [])
        for feedback_id in feedback_ids:
            feedback = await self._load_feedback(feedback_id)
            if feedback:
                feedback_items.append(feedback)
        
        return feedback_items
    
    async def _load_feedback(self, feedback_id: str) -> Optional[FeedbackItem]:
        """Load feedback from disk"""
        feedback_file = self.storage_path / f"feedback_{feedback_id}.json"
        
        if feedback_file.exists():
            try:
                async with aiofiles.open(feedback_file, 'r') as f:
                    data = json.loads(await f.read())
                    return FeedbackItem.from_dict(data)
            except Exception as e:
                self.logger.error(f"Failed to load feedback {feedback_id}: {e}")
        
        return None
    
    async def get_feedback_by_type(self, feedback_type: FeedbackType, 
                                  limit: int = 100) -> List[FeedbackItem]:
        """Get recent feedback of a specific type"""
        feedback_items = []
        
        # List all feedback files
        feedback_files = sorted(
            self.storage_path.glob("feedback_*.json"),
            key=lambda f: f.stat().st_mtime,
            reverse=True
        )
        
        for feedback_file in feedback_files[:limit * 2]:  # Check more files than needed
            try:
                async with aiofiles.open(feedback_file, 'r') as f:
                    data = json.loads(await f.read())
                    if data['feedback_type'] == feedback_type.value:
                        feedback_items.append(FeedbackItem.from_dict(data))
                        if len(feedback_items) >= limit:
                            break
            except Exception as e:
                self.logger.error(f"Failed to load feedback from {feedback_file}: {e}")
        
        return feedback_items
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get feedback collection metrics"""
        return self.metrics.copy()
    
    async def export_feedback(self, output_file: Path, 
                            start_date: Optional[datetime] = None,
                            end_date: Optional[datetime] = None) -> int:
        """Export feedback to a file"""
        exported_count = 0
        feedback_data = []
        
        # List all feedback files
        for feedback_file in self.storage_path.glob("feedback_*.json"):
            try:
                async with aiofiles.open(feedback_file, 'r') as f:
                    data = json.loads(await f.read())
                    
                    # Check date range
                    timestamp = datetime.fromisoformat(data['timestamp'])
                    if start_date and timestamp < start_date:
                        continue
                    if end_date and timestamp > end_date:
                        continue
                    
                    feedback_data.append(data)
                    exported_count += 1
                    
            except Exception as e:
                self.logger.error(f"Failed to export feedback from {feedback_file}: {e}")
        
        # Write to output file
        try:
            async with aiofiles.open(output_file, 'w') as f:
                await f.write(json.dumps({
                    "export_date": datetime.utcnow().isoformat(),
                    "total_feedback": exported_count,
                    "feedback_items": feedback_data
                }, indent=2))
        except Exception as e:
            self.logger.error(f"Failed to write export file: {e}")
            return 0
        
        return exported_count
    
    def get_feedback_summary(self) -> Dict[str, Any]:
        """Get summary of collected feedback"""
        summary = {
            "total_feedback": self.metrics["total_feedback"],
            "feedback_by_type": self.metrics["feedback_by_type"],
            "average_rating": self.metrics["average_rating"],
            "total_sessions": self.metrics["total_sessions"],
            "active_sessions": len(self.active_sessions),
            "unique_verifications": len(self.feedback_index)
        }
        
        # Calculate feedback trends
        if self.metrics["total_feedback"] > 0:
            summary["most_common_type"] = max(
                self.metrics["feedback_by_type"].items(),
                key=lambda x: x[1]
            )[0]
            
            # Calculate satisfaction rate (based on ratings)
            if self.metrics["average_rating"] > 0:
                summary["satisfaction_rate"] = self.metrics["average_rating"]
        
        return summary