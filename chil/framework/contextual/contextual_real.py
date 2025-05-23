"""
Real Contextual Verification Framework

Implements contextual verification using NLP, cultural analysis, and situational relevance assessment.
"""

import asyncio
import logging
import statistics
from datetime import datetime
from typing import List, Dict, Any, Optional, Set, Tuple
from dataclasses import dataclass, field
from collections import Counter
import re

from .unified_base import UnifiedVerificationComponent
from ...consensus_types import Claim, VerificationResult, ConsensusProposal, NodeContext, ConsensusVerificationResult


@dataclass
class ContextualEntity:
    """Named entity extracted from text"""
    text: str
    label: str  # PERSON, ORG, GPE, DATE, etc.
    start: int
    end: int
    confidence: float = 0.0


@dataclass
class CulturalContext:
    """Cultural context analysis"""
    cultural_indicators: List[str] = field(default_factory=list)
    bias_indicators: List[str] = field(default_factory=list)
    perspective_diversity: float = 0.0
    cultural_sensitivity_score: float = 0.0
    dominant_perspective: Optional[str] = None


@dataclass
class SituationalRelevance:
    """Situational relevance assessment"""
    domain_relevance: float = 0.0
    temporal_relevance: float = 0.0
    geographical_relevance: float = 0.0
    stakeholder_relevance: float = 0.0
    context_completeness: float = 0.0


@dataclass
class SemanticAnalysis:
    """Semantic analysis results"""
    key_concepts: List[str] = field(default_factory=list)
    semantic_density: float = 0.0
    coherence_score: float = 0.0
    ambiguity_score: float = 0.0
    complexity_score: float = 0.0


class RealContextualFramework(UnifiedVerificationComponent):
    """
    Real contextual verification using NLP and cultural analysis
    """
    
    def __init__(self, enable_advanced_nlp: bool = False):
        self.enable_advanced_nlp = enable_advanced_nlp
        self.cache = {}
        self.logger = logging.getLogger(__name__)
        
        # Initialize NLP components (simplified - in real implementation would use spaCy, BERT, etc.)
        self._init_nlp_components()
        
        # Cultural bias patterns
        self.cultural_bias_patterns = {
            "western_bias": [
                "american", "european", "western", "developed", "civilized", "modern",
                "advanced", "progressive", "democratic", "free market"
            ],
            "temporal_bias": [
                "nowadays", "today", "modern times", "current", "contemporary",
                "recent", "latest", "up-to-date", "cutting-edge"
            ],
            "socioeconomic_bias": [
                "educated", "professional", "middle class", "affluent", "privileged",
                "elite", "sophisticated", "refined", "cultured"
            ],
            "gender_bias": [
                "mankind", "manpower", "chairman", "he/she", "guys", "brotherhood",
                "founding fathers", "gentlemen's agreement"
            ],
            "ableism": [
                "normal", "able-bodied", "healthy", "sane", "crazy", "insane",
                "lame", "blind to", "deaf to", "crippled by"
            ]
        }
        
        # Domain-specific indicators
        self.domain_indicators = {
            "scientific": ["study", "research", "experiment", "data", "evidence", "analysis", "peer-reviewed"],
            "political": ["government", "policy", "election", "vote", "democracy", "legislation", "candidate"],
            "economic": ["market", "economy", "financial", "investment", "trade", "business", "profit"],
            "social": ["community", "society", "culture", "social", "people", "group", "collective"],
            "technological": ["technology", "digital", "software", "algorithm", "computer", "internet", "AI"],
            "healthcare": ["medical", "health", "patient", "doctor", "treatment", "disease", "clinical"],
            "environmental": ["environment", "climate", "ecosystem", "pollution", "sustainability", "carbon"],
            "legal": ["law", "legal", "court", "judge", "rights", "constitution", "justice", "attorney"]
        }
        
        # Stakeholder categories
        self.stakeholder_patterns = {
            "individuals": ["person", "individual", "citizen", "consumer", "patient", "student"],
            "organizations": ["company", "corporation", "NGO", "organization", "institution", "agency"],
            "government": ["government", "state", "federal", "municipal", "public sector", "administration"],
            "communities": ["community", "neighborhood", "local", "residents", "population", "society"],
            "experts": ["expert", "specialist", "professional", "researcher", "scientist", "academic"],
            "vulnerable_groups": ["elderly", "children", "disabled", "minority", "indigenous", "poor"]
        }
    
    def _init_nlp_components(self):
        """Initialize NLP processing components"""
        # In real implementation, would initialize spaCy, BERT, etc.
        # For now, use simple rule-based approaches
        
        # Common English stop words
        self.stop_words = {
            'a', 'an', 'and', 'are', 'as', 'at', 'be', 'been', 'by', 'for', 'from',
            'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the', 'to',
            'was', 'will', 'with', 'the', 'this', 'but', 'they', 'have', 'had',
            'what', 'said', 'each', 'which', 'she', 'do', 'how', 'their', 'if',
            'up', 'out', 'many', 'then', 'them', 'these', 'so', 'some', 'her',
            'would', 'make', 'like', 'into', 'him', 'time', 'two', 'more', 'go',
            'no', 'way', 'could', 'my', 'than', 'first', 'well', 'water', 'been',
            'call', 'who', 'its', 'now', 'find', 'long', 'down', 'day', 'did',
            'get', 'come', 'made', 'may', 'part'
        }
        
        # Simple entity patterns (in real implementation, use NER models)
        self.entity_patterns = {
            'PERSON': re.compile(r'\b[A-Z][a-z]+ [A-Z][a-z]+\b'),
            'DATE': re.compile(r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b|\b\d{4}\b'),
            'MONEY': re.compile(r'\$\d+(?:,\d{3})*(?:\.\d{2})?|\b\d+(?:,\d{3})* dollars?\b'),
            'PERCENT': re.compile(r'\d+(?:\.\d+)?%|\b\d+(?:\.\d+)? percent\b'),
            'ORG': re.compile(r'\b[A-Z][a-z]* (?:Inc|LLC|Corp|Company|Organization|Agency|Department)\b')
        }
    
    async def verify_individual(self, claim: Claim) -> VerificationResult:
        """Verify claim through contextual analysis"""
        try:
            # Check cache
            cache_key = self._generate_cache_key(claim)
            if cache_key in self.cache:
                self.logger.info(f"Using cached contextual result for claim: {claim.content[:50]}...")
                return self.cache[cache_key]
            
            # Perform parallel contextual analyses
            entity_task = self._extract_entities(claim)
            cultural_task = self._analyze_cultural_context(claim)
            situational_task = self._assess_situational_relevance(claim)
            semantic_task = self._perform_semantic_analysis(claim)
            
            entities, cultural_context, situational_relevance, semantic_analysis = await asyncio.gather(
                entity_task, cultural_task, situational_task, semantic_task,
                return_exceptions=True
            )
            
            # Handle exceptions
            if isinstance(entities, Exception):
                self.logger.warning(f"Entity extraction failed: {entities}")
                entities = []
            if isinstance(cultural_context, Exception):
                self.logger.warning(f"Cultural analysis failed: {cultural_context}")
                cultural_context = CulturalContext()
            if isinstance(situational_relevance, Exception):
                self.logger.warning(f"Situational analysis failed: {situational_relevance}")
                situational_relevance = SituationalRelevance()
            if isinstance(semantic_analysis, Exception):
                self.logger.warning(f"Semantic analysis failed: {semantic_analysis}")
                semantic_analysis = SemanticAnalysis()
            
            # Combine contextual scores
            overall_score = self._combine_contextual_scores(
                entities, cultural_context, situational_relevance, semantic_analysis, claim
            )
            
            # Generate reasoning
            reasoning = self._generate_contextual_reasoning(
                entities, cultural_context, situational_relevance, semantic_analysis
            )
            
            # Identify uncertainty factors
            uncertainty_factors = self._identify_contextual_uncertainties(
                entities, cultural_context, situational_relevance, semantic_analysis
            )
            
            # Generate contextual notes
            contextual_notes = self._generate_contextual_insights(
                claim, entities, cultural_context, situational_relevance, semantic_analysis
            )
            
            result = VerificationResult(
                framework_name="contextual",
                confidence_score=overall_score,
                reasoning=reasoning,
                evidence_references=self._collect_contextual_references(entities, cultural_context),
                uncertainty_factors=uncertainty_factors,
                contextual_notes=contextual_notes,
                metadata={
                    "entities_found": len(entities),
                    "cultural_bias_indicators": len(cultural_context.bias_indicators),
                    "domain_relevance": situational_relevance.domain_relevance,
                    "semantic_coherence": semantic_analysis.coherence_score,
                    "verification_timestamp": datetime.utcnow().isoformat(),
                    "nlp_mode": "advanced" if self.enable_advanced_nlp else "basic"
                }
            )
            
            # Cache result
            self.cache[cache_key] = result
            
            return result
            
        except Exception as e:
            self.logger.error(f"Contextual verification failed: {e}")
            return VerificationResult(
                framework_name="contextual",
                confidence_score=0.1,
                reasoning=f"Contextual verification failed due to technical error: {str(e)}",
                evidence_references=[],
                uncertainty_factors=["Technical failure", "Unable to perform contextual analysis"],
                contextual_notes="This result should be treated with extreme caution."
            )
    
    async def verify_consensus(self, proposal: ConsensusProposal, node_context: NodeContext) -> ConsensusVerificationResult:
        """Verify proposal in consensus mode with node-specific contextual adjustments"""
        # Extract claim from proposal
        claim_data = proposal.content.get("claim", {})
        claim = Claim(
            content=claim_data.get("content", ""),
            context=claim_data.get("context", {}),
            metadata=claim_data.get("metadata", {})
        )
        
        # Run individual verification
        individual_result = await self.verify_individual(claim)
        
        # Apply node-specific contextual adjustments
        node_adjusted_score = self._apply_contextual_node_adjustments(
            individual_result.confidence_score, node_context, individual_result
        )
        
        # Assess contextual consensus readiness
        consensus_readiness = self._assess_contextual_consensus_readiness(
            individual_result, node_context
        )
        
        return ConsensusVerificationResult(
            node_id=node_context.node_id,
            framework_name="contextual",
            confidence_score=node_adjusted_score,
            reasoning=individual_result.reasoning,
            evidence_quality=self._assess_contextual_evidence_quality(individual_result),
            consensus_readiness=consensus_readiness,
            suggested_refinements=self._suggest_contextual_refinements(individual_result, node_context),
            metadata={
                **individual_result.metadata,
                "node_contextual_adjustments": True,
                "consensus_mode": True,
                "node_cultural_sensitivity": node_context.philosophical_weights.get("contextual", 1.0)
            }
        )
    
    async def _extract_entities(self, claim: Claim) -> List[ContextualEntity]:
        """Extract named entities from claim text"""
        entities = []
        text = claim.content
        
        # Simple entity extraction (in real implementation, use spaCy or BERT-NER)
        for entity_type, pattern in self.entity_patterns.items():
            for match in pattern.finditer(text):
                entities.append(ContextualEntity(
                    text=match.group(),
                    label=entity_type,
                    start=match.start(),
                    end=match.end(),
                    confidence=0.8  # Would be actual model confidence
                ))
        
        # Sort by position in text
        entities.sort(key=lambda x: x.start)
        
        return entities
    
    async def _analyze_cultural_context(self, claim: Claim) -> CulturalContext:
        """Analyze cultural biases and perspectives in the claim"""
        content_lower = claim.content.lower()
        
        # Detect cultural bias indicators
        bias_indicators = []
        cultural_indicators = []
        
        for bias_type, patterns in self.cultural_bias_patterns.items():
            found_patterns = [pattern for pattern in patterns if pattern in content_lower]
            if found_patterns:
                bias_indicators.extend([f"{bias_type}: {pattern}" for pattern in found_patterns])
                cultural_indicators.extend(found_patterns)
        
        # Calculate perspective diversity (simplified)
        # In real implementation, would use more sophisticated analysis
        unique_perspectives = len(set(cultural_indicators))
        perspective_diversity = min(unique_perspectives / 5.0, 1.0)  # Normalize by max expected diversity
        
        # Cultural sensitivity score
        total_bias_indicators = len(bias_indicators)
        cultural_sensitivity_score = max(1.0 - (total_bias_indicators / 10.0), 0.0)
        
        # Identify dominant perspective
        if "western_bias" in str(bias_indicators):
            dominant_perspective = "Western/Anglo-centric"
        elif "temporal_bias" in str(bias_indicators):
            dominant_perspective = "Contemporary/Modern-centric"
        elif "socioeconomic_bias" in str(bias_indicators):
            dominant_perspective = "Socioeconomically privileged"
        else:
            dominant_perspective = "Unclear or balanced"
        
        return CulturalContext(
            cultural_indicators=cultural_indicators,
            bias_indicators=bias_indicators,
            perspective_diversity=perspective_diversity,
            cultural_sensitivity_score=cultural_sensitivity_score,
            dominant_perspective=dominant_perspective
        )
    
    async def _assess_situational_relevance(self, claim: Claim) -> SituationalRelevance:
        """Assess relevance to different situational contexts"""
        content_lower = claim.content.lower()
        
        # Domain relevance
        domain_scores = {}
        for domain, indicators in self.domain_indicators.items():
            score = sum(1 for indicator in indicators if indicator in content_lower)
            domain_scores[domain] = score / len(indicators)
        
        domain_relevance = max(domain_scores.values()) if domain_scores else 0.0
        
        # Temporal relevance (check for time-sensitive language)
        temporal_indicators = ["now", "today", "current", "recent", "latest", "modern", "contemporary"]
        temporal_score = sum(1 for indicator in temporal_indicators if indicator in content_lower)
        temporal_relevance = min(temporal_score / 5.0, 1.0)
        
        # Geographical relevance (check for location-specific language)
        geo_indicators = ["local", "regional", "national", "global", "worldwide", "international"]
        geo_score = sum(1 for indicator in geo_indicators if indicator in content_lower)
        geographical_relevance = min(geo_score / 3.0, 1.0)
        
        # Stakeholder relevance
        stakeholder_scores = {}
        for stakeholder_type, patterns in self.stakeholder_patterns.items():
            score = sum(1 for pattern in patterns if pattern in content_lower)
            stakeholder_scores[stakeholder_type] = score
        
        total_stakeholder_mentions = sum(stakeholder_scores.values())
        stakeholder_relevance = min(total_stakeholder_mentions / 5.0, 1.0)
        
        # Context completeness (check if claim provides sufficient context)
        context_elements = len(claim.context) if claim.context else 0
        metadata_elements = len(claim.metadata) if claim.metadata else 0
        context_completeness = min((context_elements + metadata_elements) / 5.0, 1.0)
        
        return SituationalRelevance(
            domain_relevance=domain_relevance,
            temporal_relevance=temporal_relevance,
            geographical_relevance=geographical_relevance,
            stakeholder_relevance=stakeholder_relevance,
            context_completeness=context_completeness
        )
    
    async def _perform_semantic_analysis(self, claim: Claim) -> SemanticAnalysis:
        """Perform semantic analysis of the claim"""
        words = claim.content.lower().split()
        
        # Remove stop words and get key concepts
        meaningful_words = [word for word in words if word not in self.stop_words and len(word) > 2]
        
        # Key concepts (most frequent meaningful words)
        word_counts = Counter(meaningful_words)
        key_concepts = [word for word, count in word_counts.most_common(10)]
        
        # Semantic density (ratio of meaningful words to total words)
        semantic_density = len(meaningful_words) / max(len(words), 1)
        
        # Coherence score (simplified - based on word repetition and structure)
        # In real implementation, would use semantic similarity models
        repeated_concepts = sum(1 for word, count in word_counts.items() if count > 1)
        coherence_score = min(repeated_concepts / max(len(key_concepts), 1), 1.0)
        
        # Ambiguity score (check for ambiguous language)
        ambiguous_words = ["might", "could", "possibly", "perhaps", "maybe", "sometimes", "often", "usually"]
        ambiguity_indicators = sum(1 for word in ambiguous_words if word in words)
        ambiguity_score = min(ambiguity_indicators / 5.0, 1.0)
        
        # Complexity score (based on sentence length and vocabulary complexity)
        avg_word_length = sum(len(word) for word in meaningful_words) / max(len(meaningful_words), 1)
        sentence_count = claim.content.count('.') + claim.content.count('!') + claim.content.count('?') + 1
        avg_sentence_length = len(words) / sentence_count
        
        complexity_score = min((avg_word_length / 8.0 + avg_sentence_length / 20.0) / 2.0, 1.0)
        
        return SemanticAnalysis(
            key_concepts=key_concepts,
            semantic_density=semantic_density,
            coherence_score=coherence_score,
            ambiguity_score=ambiguity_score,
            complexity_score=complexity_score
        )
    
    def _combine_contextual_scores(self, entities: List[ContextualEntity],
                                 cultural_context: CulturalContext,
                                 situational_relevance: SituationalRelevance,
                                 semantic_analysis: SemanticAnalysis,
                                 claim: Claim) -> float:
        """Combine all contextual scores into overall contextual verification score"""
        
        scores = []
        weights = []
        
        # Entity richness score
        entity_score = min(len(entities) / 5.0, 1.0)  # Normalize by expected entity count
        scores.append(entity_score)
        weights.append(0.15)
        
        # Cultural sensitivity score
        scores.append(cultural_context.cultural_sensitivity_score)
        weights.append(0.2)
        
        # Perspective diversity score
        scores.append(cultural_context.perspective_diversity)
        weights.append(0.15)
        
        # Domain relevance score
        scores.append(situational_relevance.domain_relevance)
        weights.append(0.2)
        
        # Context completeness score
        scores.append(situational_relevance.context_completeness)
        weights.append(0.1)
        
        # Semantic coherence score
        scores.append(semantic_analysis.coherence_score)
        weights.append(0.15)
        
        # Penalize high ambiguity
        ambiguity_penalty = 1.0 - semantic_analysis.ambiguity_score
        scores.append(ambiguity_penalty)
        weights.append(0.05)
        
        # Calculate weighted average
        overall_score = sum(s * w for s, w in zip(scores, weights)) / sum(weights)
        
        return overall_score
    
    def _generate_contextual_reasoning(self, entities: List[ContextualEntity],
                                     cultural_context: CulturalContext,
                                     situational_relevance: SituationalRelevance,
                                     semantic_analysis: SemanticAnalysis) -> str:
        """Generate human-readable reasoning for contextual verification"""
        reasoning_parts = []
        
        # Entity analysis
        if entities:
            entity_types = Counter([e.label for e in entities])
            reasoning_parts.append(f"Identified {len(entities)} contextual entities including {dict(entity_types)}")
        else:
            reasoning_parts.append("Limited contextual entities identified")
        
        # Cultural analysis
        if cultural_context.bias_indicators:
            reasoning_parts.append(f"Detected {len(cultural_context.bias_indicators)} potential cultural bias indicators")
        
        if cultural_context.perspective_diversity > 0.7:
            reasoning_parts.append("High perspective diversity indicates inclusive contextual framing")
        elif cultural_context.perspective_diversity < 0.3:
            reasoning_parts.append("Low perspective diversity suggests potential contextual limitations")
        
        # Situational relevance
        if situational_relevance.domain_relevance > 0.7:
            reasoning_parts.append("Strong domain-specific contextual relevance")
        elif situational_relevance.domain_relevance < 0.3:
            reasoning_parts.append("Limited domain-specific context provided")
        
        # Semantic analysis
        if semantic_analysis.coherence_score > 0.7:
            reasoning_parts.append("High semantic coherence indicates clear contextual communication")
        elif semantic_analysis.ambiguity_score > 0.5:
            reasoning_parts.append("Significant ambiguity detected in contextual expression")
        
        return ". ".join(reasoning_parts) + "." if reasoning_parts else "Basic contextual analysis completed."
    
    def _identify_contextual_uncertainties(self, entities: List[ContextualEntity],
                                         cultural_context: CulturalContext,
                                         situational_relevance: SituationalRelevance,
                                         semantic_analysis: SemanticAnalysis) -> List[str]:
        """Identify factors contributing to contextual uncertainty"""
        uncertainties = []
        
        if len(entities) < 2:
            uncertainties.append("Limited contextual entities for verification")
        
        if len(cultural_context.bias_indicators) > 5:
            uncertainties.append("Multiple cultural bias indicators detected")
        
        if cultural_context.perspective_diversity < 0.3:
            uncertainties.append("Limited perspective diversity")
        
        if situational_relevance.context_completeness < 0.5:
            uncertainties.append("Incomplete contextual information provided")
        
        if semantic_analysis.ambiguity_score > 0.6:
            uncertainties.append("High ambiguity in contextual expression")
        
        if situational_relevance.domain_relevance < 0.3:
            uncertainties.append("Unclear domain-specific context")
        
        return uncertainties
    
    def _generate_contextual_insights(self, claim: Claim,
                                    entities: List[ContextualEntity],
                                    cultural_context: CulturalContext,
                                    situational_relevance: SituationalRelevance,
                                    semantic_analysis: SemanticAnalysis) -> str:
        """Generate contextual insights and recommendations"""
        insights = []
        
        # Domain insights
        if situational_relevance.domain_relevance > 0.7:
            insights.append("Claim demonstrates strong domain-specific contextualization")
        
        # Cultural insights
        if cultural_context.dominant_perspective and cultural_context.dominant_perspective != "Unclear or balanced":
            insights.append(f"Claim reflects {cultural_context.dominant_perspective} perspective")
        
        # Temporal insights
        if situational_relevance.temporal_relevance > 0.7:
            insights.append("Claim shows strong temporal context awareness")
        
        # Stakeholder insights
        if situational_relevance.stakeholder_relevance > 0.7:
            insights.append("Multiple stakeholder perspectives considered")
        
        # Semantic insights
        if semantic_analysis.complexity_score > 0.7:
            insights.append("High semantic complexity may affect accessibility")
        elif semantic_analysis.complexity_score < 0.3:
            insights.append("Simple semantic structure enhances clarity")
        
        # Key concepts
        if semantic_analysis.key_concepts:
            top_concepts = semantic_analysis.key_concepts[:3]
            insights.append(f"Key contextual concepts: {', '.join(top_concepts)}")
        
        return ". ".join(insights) + "." if insights else "Standard contextual analysis completed."
    
    def _collect_contextual_references(self, entities: List[ContextualEntity],
                                     cultural_context: CulturalContext) -> List[str]:
        """Collect references for contextual evidence"""
        references = []
        
        # Entity references
        for entity in entities[:5]:  # Limit to top 5
            references.append(f"{entity.label}: {entity.text}")
        
        # Cultural context references
        if cultural_context.bias_indicators:
            references.append(f"Cultural bias indicators: {', '.join(cultural_context.bias_indicators[:3])}")
        
        # Perspective analysis
        if cultural_context.dominant_perspective:
            references.append(f"Dominant perspective: {cultural_context.dominant_perspective}")
        
        return references
    
    def _apply_contextual_node_adjustments(self, score: float, node_context: NodeContext,
                                         result: VerificationResult) -> float:
        """Apply node-specific contextual adjustments"""
        # Get node's contextual framework weight
        contextual_weight = node_context.philosophical_weights.get("contextual", 1.0)
        
        # Adjust based on node's contextual sensitivity
        adjusted_score = score * contextual_weight
        
        # Additional adjustments based on node context
        if hasattr(node_context, 'cultural_background'):
            # Could adjust based on cultural alignment
            pass
        
        return min(adjusted_score, 1.0)
    
    def _assess_contextual_evidence_quality(self, result: VerificationResult) -> float:
        """Assess quality of contextual evidence for consensus"""
        quality_factors = []
        
        # Entity richness
        entities_found = result.metadata.get("entities_found", 0)
        entity_quality = min(entities_found / 5.0, 1.0)
        quality_factors.append(entity_quality)
        
        # Cultural bias awareness
        bias_indicators = result.metadata.get("cultural_bias_indicators", 0)
        bias_quality = max(1.0 - (bias_indicators / 10.0), 0.2)  # Fewer bias indicators = higher quality
        quality_factors.append(bias_quality)
        
        # Domain relevance
        domain_relevance = result.metadata.get("domain_relevance", 0.0)
        quality_factors.append(domain_relevance)
        
        # Semantic coherence
        semantic_coherence = result.metadata.get("semantic_coherence", 0.0)
        quality_factors.append(semantic_coherence)
        
        return statistics.mean(quality_factors)
    
    def _assess_contextual_consensus_readiness(self, result: VerificationResult,
                                             node_context: NodeContext) -> bool:
        """Assess if contextual analysis is ready for consensus"""
        # Base readiness on confidence score
        base_readiness = result.confidence_score > 0.6
        
        # Check for critical uncertainties
        critical_uncertainties = [
            "Multiple cultural bias indicators detected",
            "Limited perspective diversity",
            "Incomplete contextual information provided"
        ]
        
        has_critical_issues = any(uncertainty in result.uncertainty_factors 
                                for uncertainty in critical_uncertainties)
        
        return base_readiness and not has_critical_issues
    
    def _suggest_contextual_refinements(self, result: VerificationResult,
                                      node_context: NodeContext) -> List[str]:
        """Suggest refinements for better contextual verification"""
        suggestions = []
        
        if result.confidence_score < 0.7:
            suggestions.append("Provide additional contextual information to strengthen claim")
        
        if "Limited contextual entities for verification" in result.uncertainty_factors:
            suggestions.append("Include more specific contextual details (names, dates, locations)")
        
        if "Multiple cultural bias indicators detected" in result.uncertainty_factors:
            suggestions.append("Consider rephrasing to reduce cultural bias and improve inclusivity")
        
        if "Limited perspective diversity" in result.uncertainty_factors:
            suggestions.append("Incorporate multiple perspectives and stakeholder viewpoints")
        
        if "Incomplete contextual information provided" in result.uncertainty_factors:
            suggestions.append("Add relevant background context, domain information, and situational details")
        
        if "High ambiguity in contextual expression" in result.uncertainty_factors:
            suggestions.append("Clarify ambiguous language and provide more precise contextual framing")
        
        return suggestions
    
    def _generate_cache_key(self, claim: Claim) -> str:
        """Generate cache key for contextual analysis"""
        import hashlib
        content_hash = hashlib.md5(claim.content.encode()).hexdigest()
        context_hash = hashlib.md5(str(sorted(claim.context.items())).encode()).hexdigest()
        return f"contextual_{content_hash}_{context_hash}"
    
    def clear_cache(self):
        """Clear contextual analysis cache"""
        self.cache.clear()
    
    def get_contextual_stats(self) -> Dict[str, Any]:
        """Get contextual analysis statistics"""
        return {
            "cached_analyses": len(self.cache),
            "nlp_mode": "advanced" if self.enable_advanced_nlp else "basic",
            "cultural_bias_categories": len(self.cultural_bias_patterns),
            "domain_categories": len(self.domain_indicators),
            "stakeholder_categories": len(self.stakeholder_patterns)
        }