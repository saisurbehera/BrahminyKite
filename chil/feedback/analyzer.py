"""
Feedback analyzer for detecting patterns and extracting insights from human feedback.
"""

import asyncio
from collections import defaultdict, Counter
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set, Tuple, Any
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity

from .collector import FeedbackCollector, FeedbackType, FeedbackItem


@dataclass
class FeedbackPattern:
    """Represents a pattern detected in feedback"""
    pattern_type: str
    description: str
    frequency: int
    confidence: float
    examples: List[FeedbackItem] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FeedbackInsight:
    """Represents an actionable insight from feedback analysis"""
    insight_type: str
    title: str
    description: str
    severity: str  # "low", "medium", "high", "critical"
    affected_components: List[str]
    recommended_actions: List[str]
    supporting_patterns: List[FeedbackPattern]
    confidence: float
    created_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class AnalysisReport:
    """Comprehensive feedback analysis report"""
    period_start: datetime
    period_end: datetime
    total_feedback: int
    unique_users: int
    patterns: List[FeedbackPattern]
    insights: List[FeedbackInsight]
    metrics: Dict[str, Any]
    recommendations: List[str]


class FeedbackAnalyzer:
    """Analyzes human feedback to detect patterns and generate insights"""
    
    def __init__(self, collector: FeedbackCollector):
        self.collector = collector
        self._pattern_detectors = self._initialize_pattern_detectors()
        self._insight_generators = self._initialize_insight_generators()
        
    def _initialize_pattern_detectors(self) -> Dict[str, Any]:
        """Initialize pattern detection algorithms"""
        return {
            "accuracy_trends": self._detect_accuracy_trends,
            "common_corrections": self._detect_common_corrections,
            "dispute_clusters": self._detect_dispute_clusters,
            "user_agreement": self._detect_user_agreement_patterns,
            "temporal_patterns": self._detect_temporal_patterns,
            "component_issues": self._detect_component_issues,
        }
    
    def _initialize_insight_generators(self) -> Dict[str, Any]:
        """Initialize insight generation algorithms"""
        return {
            "system_weaknesses": self._generate_weakness_insights,
            "improvement_opportunities": self._generate_improvement_insights,
            "user_satisfaction": self._generate_satisfaction_insights,
            "reliability_issues": self._generate_reliability_insights,
        }
    
    async def analyze_feedback(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        min_feedback_count: int = 10
    ) -> AnalysisReport:
        """Perform comprehensive feedback analysis"""
        # Get feedback for period
        if not end_date:
            end_date = datetime.utcnow()
        if not start_date:
            start_date = end_date - timedelta(days=30)
            
        feedback_items = await self._get_feedback_for_period(start_date, end_date)
        
        if len(feedback_items) < min_feedback_count:
            return AnalysisReport(
                period_start=start_date,
                period_end=end_date,
                total_feedback=len(feedback_items),
                unique_users=len(set(f.user_id for f in feedback_items)),
                patterns=[],
                insights=[],
                metrics={},
                recommendations=["Insufficient feedback for meaningful analysis"]
            )
        
        # Detect patterns
        patterns = await self._detect_all_patterns(feedback_items)
        
        # Generate insights
        insights = await self._generate_all_insights(patterns, feedback_items)
        
        # Calculate metrics
        metrics = self._calculate_metrics(feedback_items, patterns, insights)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(insights)
        
        return AnalysisReport(
            period_start=start_date,
            period_end=end_date,
            total_feedback=len(feedback_items),
            unique_users=len(set(f.user_id for f in feedback_items)),
            patterns=patterns,
            insights=insights,
            metrics=metrics,
            recommendations=recommendations
        )
    
    async def _get_feedback_for_period(
        self,
        start_date: datetime,
        end_date: datetime
    ) -> List[FeedbackItem]:
        """Get all feedback items for the specified period"""
        # This would typically query a database
        # For now, we'll use the in-memory storage from collector
        all_feedback = []
        for items in self.collector._feedback_storage.values():
            for item in items:
                if start_date <= item.timestamp <= end_date:
                    all_feedback.append(item)
        return all_feedback
    
    async def _detect_all_patterns(
        self,
        feedback_items: List[FeedbackItem]
    ) -> List[FeedbackPattern]:
        """Run all pattern detection algorithms"""
        patterns = []
        for detector_name, detector_func in self._pattern_detectors.items():
            try:
                detected = await detector_func(feedback_items)
                patterns.extend(detected)
            except Exception as e:
                print(f"Error in pattern detector {detector_name}: {e}")
        return patterns
    
    async def _generate_all_insights(
        self,
        patterns: List[FeedbackPattern],
        feedback_items: List[FeedbackItem]
    ) -> List[FeedbackInsight]:
        """Generate insights from patterns and feedback"""
        insights = []
        for generator_name, generator_func in self._insight_generators.items():
            try:
                generated = await generator_func(patterns, feedback_items)
                insights.extend(generated)
            except Exception as e:
                print(f"Error in insight generator {generator_name}: {e}")
        return insights
    
    async def _detect_accuracy_trends(
        self,
        feedback_items: List[FeedbackItem]
    ) -> List[FeedbackPattern]:
        """Detect trends in accuracy feedback"""
        accuracy_feedback = [
            f for f in feedback_items 
            if f.feedback_type == FeedbackType.ACCURACY
        ]
        
        if not accuracy_feedback:
            return []
        
        patterns = []
        
        # Group by verification ID to find consistently inaccurate verifications
        verification_accuracy = defaultdict(list)
        for feedback in accuracy_feedback:
            verification_accuracy[feedback.verification_id].append(feedback.rating)
        
        # Find verifications with low average accuracy
        low_accuracy_verifications = []
        for vid, ratings in verification_accuracy.items():
            avg_rating = np.mean(ratings)
            if avg_rating < 3.0 and len(ratings) >= 3:
                low_accuracy_verifications.append((vid, avg_rating, ratings))
        
        if low_accuracy_verifications:
            pattern = FeedbackPattern(
                pattern_type="low_accuracy_trend",
                description=f"Found {len(low_accuracy_verifications)} verifications with consistently low accuracy ratings",
                frequency=len(low_accuracy_verifications),
                confidence=0.85,
                examples=accuracy_feedback[:5],
                metadata={
                    "affected_verifications": [v[0] for v in low_accuracy_verifications],
                    "average_ratings": {v[0]: v[1] for v in low_accuracy_verifications}
                }
            )
            patterns.append(pattern)
        
        return patterns
    
    async def _detect_common_corrections(
        self,
        feedback_items: List[FeedbackItem]
    ) -> List[FeedbackPattern]:
        """Detect common correction patterns"""
        corrections = [
            f for f in feedback_items 
            if f.feedback_type == FeedbackType.CORRECTION
        ]
        
        if not corrections:
            return []
        
        patterns = []
        
        # Extract correction texts
        correction_texts = [c.text for c in corrections if c.text]
        
        if len(correction_texts) >= 5:
            # Use TF-IDF to find similar corrections
            vectorizer = TfidfVectorizer(max_features=100, stop_words='english')
            tfidf_matrix = vectorizer.fit_transform(correction_texts)
            
            # Cluster similar corrections
            n_clusters = min(5, len(correction_texts) // 3)
            if n_clusters > 1:
                kmeans = KMeans(n_clusters=n_clusters, random_state=42)
                clusters = kmeans.fit_predict(tfidf_matrix)
                
                # Find large clusters
                cluster_sizes = Counter(clusters)
                for cluster_id, size in cluster_sizes.items():
                    if size >= 3:
                        cluster_corrections = [
                            corrections[i] for i, c in enumerate(clusters) 
                            if c == cluster_id
                        ]
                        
                        pattern = FeedbackPattern(
                            pattern_type="common_correction_cluster",
                            description=f"Cluster of {size} similar corrections",
                            frequency=size,
                            confidence=0.75,
                            examples=cluster_corrections[:3],
                            metadata={
                                "cluster_id": int(cluster_id),
                                "sample_texts": [c.text for c in cluster_corrections[:3]]
                            }
                        )
                        patterns.append(pattern)
        
        return patterns
    
    async def _detect_dispute_clusters(
        self,
        feedback_items: List[FeedbackItem]
    ) -> List[FeedbackPattern]:
        """Detect clusters of disputes"""
        disputes = [
            f for f in feedback_items 
            if f.feedback_type == FeedbackType.DISPUTE
        ]
        
        if not disputes:
            return []
        
        patterns = []
        
        # Group disputes by verification ID
        verification_disputes = defaultdict(list)
        for dispute in disputes:
            verification_disputes[dispute.verification_id].append(dispute)
        
        # Find verifications with multiple disputes
        contested_verifications = [
            (vid, disputes_list) 
            for vid, disputes_list in verification_disputes.items() 
            if len(disputes_list) >= 2
        ]
        
        if contested_verifications:
            pattern = FeedbackPattern(
                pattern_type="contested_verifications",
                description=f"Found {len(contested_verifications)} verifications with multiple disputes",
                frequency=len(contested_verifications),
                confidence=0.9,
                examples=[d for _, disputes in contested_verifications[:3] for d in disputes],
                metadata={
                    "contested_verification_ids": [v[0] for v in contested_verifications],
                    "dispute_counts": {v[0]: len(v[1]) for v in contested_verifications}
                }
            )
            patterns.append(pattern)
        
        return patterns
    
    async def _detect_user_agreement_patterns(
        self,
        feedback_items: List[FeedbackItem]
    ) -> List[FeedbackPattern]:
        """Detect patterns in user agreement/disagreement"""
        confirmations = [
            f for f in feedback_items 
            if f.feedback_type == FeedbackType.CONFIRMATION
        ]
        disputes = [
            f for f in feedback_items 
            if f.feedback_type == FeedbackType.DISPUTE
        ]
        
        patterns = []
        
        # Calculate agreement ratio
        total_judgments = len(confirmations) + len(disputes)
        if total_judgments >= 10:
            agreement_ratio = len(confirmations) / total_judgments
            
            if agreement_ratio < 0.5:
                pattern = FeedbackPattern(
                    pattern_type="low_user_agreement",
                    description=f"Low user agreement rate: {agreement_ratio:.2%}",
                    frequency=total_judgments,
                    confidence=0.85,
                    examples=disputes[:5],
                    metadata={
                        "confirmations": len(confirmations),
                        "disputes": len(disputes),
                        "agreement_ratio": agreement_ratio
                    }
                )
                patterns.append(pattern)
            elif agreement_ratio > 0.9:
                pattern = FeedbackPattern(
                    pattern_type="high_user_agreement",
                    description=f"High user agreement rate: {agreement_ratio:.2%}",
                    frequency=total_judgments,
                    confidence=0.85,
                    examples=confirmations[:5],
                    metadata={
                        "confirmations": len(confirmations),
                        "disputes": len(disputes),
                        "agreement_ratio": agreement_ratio
                    }
                )
                patterns.append(pattern)
        
        return patterns
    
    async def _detect_temporal_patterns(
        self,
        feedback_items: List[FeedbackItem]
    ) -> List[FeedbackPattern]:
        """Detect temporal patterns in feedback"""
        if not feedback_items:
            return []
        
        patterns = []
        
        # Group feedback by hour of day
        hourly_feedback = defaultdict(list)
        for feedback in feedback_items:
            hour = feedback.timestamp.hour
            hourly_feedback[hour].append(feedback)
        
        # Find peak hours
        peak_hours = sorted(
            hourly_feedback.items(),
            key=lambda x: len(x[1]),
            reverse=True
        )[:3]
        
        if peak_hours and len(peak_hours[0][1]) >= 5:
            pattern = FeedbackPattern(
                pattern_type="peak_feedback_hours",
                description=f"Peak feedback hours: {', '.join(f'{h[0]}:00' for h in peak_hours)}",
                frequency=sum(len(h[1]) for h in peak_hours),
                confidence=0.7,
                examples=peak_hours[0][1][:3],
                metadata={
                    "peak_hours": [(h[0], len(h[1])) for h in peak_hours],
                    "hourly_distribution": {h: len(f) for h, f in hourly_feedback.items()}
                }
            )
            patterns.append(pattern)
        
        return patterns
    
    async def _detect_component_issues(
        self,
        feedback_items: List[FeedbackItem]
    ) -> List[FeedbackPattern]:
        """Detect issues with specific verification components"""
        patterns = []
        
        # Group feedback by metadata components
        component_feedback = defaultdict(list)
        for feedback in feedback_items:
            if feedback.metadata and "component" in feedback.metadata:
                component = feedback.metadata["component"]
                component_feedback[component].append(feedback)
        
        # Find components with negative feedback
        for component, feedbacks in component_feedback.items():
            negative_count = sum(
                1 for f in feedbacks 
                if (f.rating and f.rating < 3) or f.feedback_type == FeedbackType.DISPUTE
            )
            
            if negative_count >= 3 and negative_count / len(feedbacks) > 0.5:
                pattern = FeedbackPattern(
                    pattern_type="component_issues",
                    description=f"Component '{component}' has high negative feedback rate",
                    frequency=negative_count,
                    confidence=0.8,
                    examples=feedbacks[:3],
                    metadata={
                        "component": component,
                        "total_feedback": len(feedbacks),
                        "negative_feedback": negative_count,
                        "negative_rate": negative_count / len(feedbacks)
                    }
                )
                patterns.append(pattern)
        
        return patterns
    
    async def _generate_weakness_insights(
        self,
        patterns: List[FeedbackPattern],
        feedback_items: List[FeedbackItem]
    ) -> List[FeedbackInsight]:
        """Generate insights about system weaknesses"""
        insights = []
        
        # Check for low accuracy patterns
        low_accuracy_patterns = [
            p for p in patterns 
            if p.pattern_type == "low_accuracy_trend"
        ]
        
        if low_accuracy_patterns:
            affected_verifications = set()
            for pattern in low_accuracy_patterns:
                affected_verifications.update(
                    pattern.metadata.get("affected_verifications", [])
                )
            
            insight = FeedbackInsight(
                insight_type="system_weakness",
                title="Systematic Accuracy Issues",
                description=f"Multiple verifications show consistently low accuracy ratings, affecting {len(affected_verifications)} verifications",
                severity="high",
                affected_components=["verification_engine"],
                recommended_actions=[
                    "Review verification algorithms for the affected verifications",
                    "Increase training data for low-performing models",
                    "Implement additional validation checks"
                ],
                supporting_patterns=low_accuracy_patterns,
                confidence=0.85
            )
            insights.append(insight)
        
        # Check for component issues
        component_patterns = [
            p for p in patterns 
            if p.pattern_type == "component_issues"
        ]
        
        for pattern in component_patterns:
            component = pattern.metadata.get("component", "unknown")
            negative_rate = pattern.metadata.get("negative_rate", 0)
            
            severity = "critical" if negative_rate > 0.8 else "high" if negative_rate > 0.6 else "medium"
            
            insight = FeedbackInsight(
                insight_type="system_weakness",
                title=f"Component Reliability Issue: {component}",
                description=f"Component '{component}' has {negative_rate:.0%} negative feedback rate",
                severity=severity,
                affected_components=[component],
                recommended_actions=[
                    f"Investigate issues with {component} component",
                    "Review recent changes to this component",
                    "Add additional monitoring and logging",
                    "Consider rolling back recent updates if applicable"
                ],
                supporting_patterns=[pattern],
                confidence=0.8
            )
            insights.append(insight)
        
        return insights
    
    async def _generate_improvement_insights(
        self,
        patterns: List[FeedbackPattern],
        feedback_items: List[FeedbackItem]
    ) -> List[FeedbackInsight]:
        """Generate improvement opportunity insights"""
        insights = []
        
        # Check for common correction clusters
        correction_patterns = [
            p for p in patterns 
            if p.pattern_type == "common_correction_cluster"
        ]
        
        if correction_patterns:
            total_corrections = sum(p.frequency for p in correction_patterns)
            
            insight = FeedbackInsight(
                insight_type="improvement_opportunity",
                title="Recurring Correction Patterns",
                description=f"Found {len(correction_patterns)} clusters of similar corrections, affecting {total_corrections} verifications",
                severity="medium",
                affected_components=["verification_engine", "training_data"],
                recommended_actions=[
                    "Analyze common correction themes",
                    "Update training data with corrected examples",
                    "Implement specific handlers for common error types",
                    "Create automated tests for identified edge cases"
                ],
                supporting_patterns=correction_patterns,
                confidence=0.75
            )
            insights.append(insight)
        
        # Check for suggestions
        suggestions = [
            f for f in feedback_items 
            if f.feedback_type == FeedbackType.SUGGESTION
        ]
        
        if len(suggestions) >= 5:
            insight = FeedbackInsight(
                insight_type="improvement_opportunity",
                title="User Suggestions Available",
                description=f"Users have provided {len(suggestions)} suggestions for improvements",
                severity="low",
                affected_components=["user_experience"],
                recommended_actions=[
                    "Review and prioritize user suggestions",
                    "Implement high-value suggestions",
                    "Communicate back to users about implemented suggestions"
                ],
                supporting_patterns=[],
                confidence=0.9
            )
            insights.append(insight)
        
        return insights
    
    async def _generate_satisfaction_insights(
        self,
        patterns: List[FeedbackPattern],
        feedback_items: List[FeedbackItem]
    ) -> List[FeedbackInsight]:
        """Generate user satisfaction insights"""
        insights = []
        
        # Check agreement patterns
        agreement_patterns = [
            p for p in patterns 
            if p.pattern_type in ["low_user_agreement", "high_user_agreement"]
        ]
        
        for pattern in agreement_patterns:
            if pattern.pattern_type == "low_user_agreement":
                agreement_ratio = pattern.metadata.get("agreement_ratio", 0)
                
                insight = FeedbackInsight(
                    insight_type="user_satisfaction",
                    title="Low User Agreement with Verifications",
                    description=f"Only {agreement_ratio:.0%} of users agree with verification results",
                    severity="high",
                    affected_components=["verification_engine", "user_interface"],
                    recommended_actions=[
                        "Investigate reasons for disagreement",
                        "Improve explanation of verification results",
                        "Consider adjusting verification thresholds",
                        "Gather more detailed feedback on disputes"
                    ],
                    supporting_patterns=[pattern],
                    confidence=0.85
                )
                insights.append(insight)
            
            elif pattern.pattern_type == "high_user_agreement":
                agreement_ratio = pattern.metadata.get("agreement_ratio", 0)
                
                insight = FeedbackInsight(
                    insight_type="user_satisfaction",
                    title="High User Agreement with Verifications",
                    description=f"{agreement_ratio:.0%} of users agree with verification results",
                    severity="low",
                    affected_components=["verification_engine"],
                    recommended_actions=[
                        "Continue current verification approach",
                        "Document successful patterns",
                        "Share best practices across components"
                    ],
                    supporting_patterns=[pattern],
                    confidence=0.85
                )
                insights.append(insight)
        
        return insights
    
    async def _generate_reliability_insights(
        self,
        patterns: List[FeedbackPattern],
        feedback_items: List[FeedbackItem]
    ) -> List[FeedbackInsight]:
        """Generate reliability insights"""
        insights = []
        
        # Check for contested verifications
        contested_patterns = [
            p for p in patterns 
            if p.pattern_type == "contested_verifications"
        ]
        
        if contested_patterns:
            total_contested = sum(
                len(p.metadata.get("contested_verification_ids", [])) 
                for p in contested_patterns
            )
            
            insight = FeedbackInsight(
                insight_type="reliability_issue",
                title="High Dispute Rate on Verifications",
                description=f"{total_contested} verifications have multiple disputes, indicating reliability concerns",
                severity="high",
                affected_components=["verification_engine", "consensus_mechanism"],
                recommended_actions=[
                    "Review disputed verifications for common patterns",
                    "Implement additional verification steps for contentious topics",
                    "Consider multi-source verification for disputed claims",
                    "Improve confidence scoring to reflect uncertainty"
                ],
                supporting_patterns=contested_patterns,
                confidence=0.9
            )
            insights.append(insight)
        
        return insights
    
    def _calculate_metrics(
        self,
        feedback_items: List[FeedbackItem],
        patterns: List[FeedbackPattern],
        insights: List[FeedbackInsight]
    ) -> Dict[str, Any]:
        """Calculate comprehensive metrics"""
        metrics = {
            "feedback_distribution": Counter(f.feedback_type.value for f in feedback_items),
            "average_rating": np.mean([f.rating for f in feedback_items if f.rating]) if any(f.rating for f in feedback_items) else None,
            "pattern_counts": Counter(p.pattern_type for p in patterns),
            "insight_severity": Counter(i.severity for i in insights),
            "unique_users": len(set(f.user_id for f in feedback_items)),
            "unique_verifications": len(set(f.verification_id for f in feedback_items)),
            "feedback_per_user": len(feedback_items) / len(set(f.user_id for f in feedback_items)) if feedback_items else 0,
        }
        
        # Calculate response times if available
        session_times = []
        for feedback in feedback_items:
            if feedback.metadata and "session_duration" in feedback.metadata:
                session_times.append(feedback.metadata["session_duration"])
        
        if session_times:
            metrics["average_session_duration"] = np.mean(session_times)
            metrics["median_session_duration"] = np.median(session_times)
        
        return metrics
    
    def _generate_recommendations(
        self,
        insights: List[FeedbackInsight]
    ) -> List[str]:
        """Generate actionable recommendations based on insights"""
        recommendations = []
        
        # Prioritize by severity
        critical_insights = [i for i in insights if i.severity == "critical"]
        high_insights = [i for i in insights if i.severity == "high"]
        
        if critical_insights:
            recommendations.append(
                f"URGENT: Address {len(critical_insights)} critical issues immediately"
            )
            for insight in critical_insights[:3]:
                recommendations.extend(insight.recommended_actions[:2])
        
        if high_insights:
            recommendations.append(
                f"Priority: Address {len(high_insights)} high-severity issues"
            )
            for insight in high_insights[:2]:
                recommendations.extend(insight.recommended_actions[:1])
        
        # General recommendations based on patterns
        if not insights:
            recommendations.append("Continue monitoring feedback for emerging patterns")
        
        # Limit total recommendations
        return recommendations[:10]
    
    async def get_real_time_metrics(self) -> Dict[str, Any]:
        """Get real-time feedback metrics"""
        # Get recent feedback (last hour)
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(hours=1)
        
        recent_feedback = await self._get_feedback_for_period(start_time, end_time)
        
        metrics = {
            "last_hour_count": len(recent_feedback),
            "last_hour_types": Counter(f.feedback_type.value for f in recent_feedback),
            "active_users": len(set(f.user_id for f in recent_feedback)),
            "average_rating": np.mean([f.rating for f in recent_feedback if f.rating]) if any(f.rating for f in recent_feedback) else None,
        }
        
        return metrics