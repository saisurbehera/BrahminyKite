"""
Real Power Dynamics Verification Framework

Analyzes power structures, institutional biases, and perspective diversity
to ensure equitable and inclusive verification.
"""

import asyncio
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional, Set, Tuple
from dataclasses import dataclass, field
from collections import Counter
import statistics

from ..individual.components.unified_base import UnifiedVerificationComponent
from ...consensus_types import Claim, VerificationResult, ConsensusProposal, NodeContext, ConsensusVerificationResult


@dataclass
class PowerStructure:
    """Identified power structure in a claim"""
    structure_type: str  # "institutional", "economic", "political", "social"
    actors: List[str] = field(default_factory=list)
    power_level: float = 0.0  # 0.0 to 1.0
    influence_scope: str = ""
    bias_potential: float = 0.0


@dataclass
class PerspectiveAnalysis:
    """Analysis of perspective diversity and inclusion"""
    represented_perspectives: List[str] = field(default_factory=list)
    missing_perspectives: List[str] = field(default_factory=list)
    dominant_perspective: Optional[str] = None
    diversity_score: float = 0.0
    inclusion_score: float = 0.0


@dataclass
class BiasAssessment:
    """Assessment of various types of bias"""
    institutional_bias: float = 0.0
    economic_bias: float = 0.0
    cultural_bias: float = 0.0
    gender_bias: float = 0.0
    racial_bias: float = 0.0
    class_bias: float = 0.0
    overall_bias_score: float = 0.0
    bias_indicators: List[str] = field(default_factory=list)


class RealPowerDynamicsFramework(UnifiedVerificationComponent):
    """
    Real power dynamics verification analyzing institutional biases and power structures
    """
    
    def __init__(self):
        self.cache = {}
        self.logger = logging.getLogger(__name__)
        
        # Power structure indicators
        self.power_indicators = {
            "institutional": [
                "government", "institution", "organization", "agency", "department",
                "university", "corporation", "company", "authority", "official"
            ],
            "economic": [
                "wealthy", "rich", "poor", "class", "income", "profit", "business",
                "industry", "market", "financial", "economic", "capitalism", "corporate"
            ],
            "political": [
                "political", "party", "election", "vote", "democracy", "republican",
                "democratic", "conservative", "liberal", "left", "right", "ideology"
            ],
            "social": [
                "elite", "privileged", "disadvantaged", "marginalized", "minority",
                "majority", "status", "hierarchy", "social", "community"
            ]
        }
        
        # Bias indicators
        self.bias_patterns = {
            "institutional": [
                "official position", "established", "mainstream", "traditional",
                "conventional", "standard", "normal", "accepted", "recognized"
            ],
            "economic": [
                "market-based", "profitable", "cost-effective", "efficient",
                "business-friendly", "investor", "shareholder", "economic growth"
            ],
            "cultural": [
                "western", "american", "european", "modern", "civilized",
                "developed", "advanced", "progressive", "educated", "sophisticated"
            ],
            "gender": [
                "masculine", "feminine", "manly", "womanly", "guys", "mankind",
                "manpower", "brotherhood", "sisterhood", "gendered"
            ],
            "racial": [
                "racial", "ethnic", "immigrant", "foreign", "native", "indigenous",
                "minority", "majority", "diversity", "multicultural"
            ],
            "class": [
                "upper class", "middle class", "working class", "blue collar",
                "white collar", "educated", "uneducated", "sophisticated", "refined"
            ]
        }
        
        # Perspective indicators
        self.perspective_categories = {
            "stakeholder_groups": [
                "consumers", "workers", "employers", "citizens", "taxpayers",
                "patients", "students", "parents", "elderly", "youth"
            ],
            "marginalized_voices": [
                "minorities", "disabled", "lgbtq", "immigrants", "refugees",
                "homeless", "unemployed", "rural", "urban", "indigenous"
            ],
            "expert_perspectives": [
                "scientists", "researchers", "academics", "professionals",
                "specialists", "experts", "practitioners", "analysts"
            ],
            "geographic_perspectives": [
                "global", "international", "national", "local", "regional",
                "urban", "rural", "developed", "developing", "first world", "third world"
            ]
        }
    
    async def verify_individual(self, claim: Claim) -> VerificationResult:
        """Verify claim through power dynamics analysis"""
        try:
            # Check cache
            cache_key = self._generate_cache_key(claim)
            if cache_key in self.cache:
                self.logger.info(f"Using cached power dynamics result for claim: {claim.content[:50]}...")
                return self.cache[cache_key]
            
            # Perform parallel power dynamics analyses
            power_structures_task = self._analyze_power_structures(claim)
            perspective_task = self._analyze_perspective_diversity(claim)
            bias_task = self._assess_bias_indicators(claim)
            stakeholder_task = self._identify_stakeholder_impact(claim)
            
            power_structures, perspective_analysis, bias_assessment, stakeholder_impact = await asyncio.gather(
                power_structures_task, perspective_task, bias_task, stakeholder_task,
                return_exceptions=True
            )
            
            # Handle exceptions
            if isinstance(power_structures, Exception):
                self.logger.warning(f"Power structure analysis failed: {power_structures}")
                power_structures = []
            if isinstance(perspective_analysis, Exception):
                self.logger.warning(f"Perspective analysis failed: {perspective_analysis}")
                perspective_analysis = PerspectiveAnalysis()
            if isinstance(bias_assessment, Exception):
                self.logger.warning(f"Bias assessment failed: {bias_assessment}")
                bias_assessment = BiasAssessment()
            if isinstance(stakeholder_impact, Exception):
                self.logger.warning(f"Stakeholder analysis failed: {stakeholder_impact}")
                stakeholder_impact = {}
            
            # Combine power dynamics scores
            overall_score = self._combine_power_dynamics_scores(
                power_structures, perspective_analysis, bias_assessment, stakeholder_impact
            )
            
            # Generate reasoning
            reasoning = self._generate_power_dynamics_reasoning(
                power_structures, perspective_analysis, bias_assessment, stakeholder_impact
            )
            
            # Identify uncertainty factors
            uncertainty_factors = self._identify_power_dynamics_uncertainties(
                power_structures, perspective_analysis, bias_assessment
            )
            
            # Generate contextual notes
            contextual_notes = self._generate_power_dynamics_insights(
                claim, power_structures, perspective_analysis, bias_assessment
            )
            
            result = VerificationResult(
                framework_name="power_dynamics",
                confidence_score=overall_score,
                reasoning=reasoning,
                evidence_references=self._collect_power_dynamics_references(power_structures, perspective_analysis),
                uncertainty_factors=uncertainty_factors,
                contextual_notes=contextual_notes,
                metadata={
                    "power_structures_identified": len(power_structures),
                    "perspective_diversity": perspective_analysis.diversity_score,
                    "inclusion_score": perspective_analysis.inclusion_score,
                    "overall_bias_score": bias_assessment.overall_bias_score,
                    "bias_indicators_found": len(bias_assessment.bias_indicators),
                    "stakeholder_groups_affected": len(stakeholder_impact),
                    "verification_timestamp": datetime.utcnow().isoformat()
                }
            )
            
            # Cache result
            self.cache[cache_key] = result
            
            return result
            
        except Exception as e:
            self.logger.error(f"Power dynamics verification failed: {e}")
            return VerificationResult(
                framework_name="power_dynamics",
                confidence_score=0.1,
                reasoning=f"Power dynamics verification failed: {str(e)}",
                evidence_references=[],
                uncertainty_factors=["Technical failure", "Unable to analyze power structures"],
                contextual_notes="This result should be treated with extreme caution."
            )
    
    async def verify_consensus(self, proposal: ConsensusProposal, node_context: NodeContext) -> ConsensusVerificationResult:
        """Verify proposal in consensus mode"""
        # Extract claim from proposal
        claim_data = proposal.content.get("claim", {})
        claim = Claim(
            content=claim_data.get("content", ""),
            context=claim_data.get("context", {}),
            metadata=claim_data.get("metadata", {})
        )
        
        # Run individual verification
        individual_result = await self.verify_individual(claim)
        
        # Apply node-specific power dynamics adjustments
        node_adjusted_score = self._apply_power_dynamics_node_adjustments(
            individual_result.confidence_score, node_context, individual_result
        )
        
        return ConsensusVerificationResult(
            node_id=node_context.node_id,
            framework_name="power_dynamics",
            confidence_score=node_adjusted_score,
            reasoning=individual_result.reasoning,
            evidence_quality=self._assess_power_dynamics_evidence_quality(individual_result),
            consensus_readiness=node_adjusted_score > 0.6,
            suggested_refinements=self._suggest_power_dynamics_refinements(individual_result, node_context),
            metadata={
                **individual_result.metadata,
                "node_power_sensitivity": True,
                "consensus_mode": True,
                "node_power_awareness": node_context.philosophical_weights.get("power_dynamics", 1.0)
            }
        )
    
    async def _analyze_power_structures(self, claim: Claim) -> List[PowerStructure]:
        """Analyze power structures mentioned or implied in the claim"""
        content_lower = claim.content.lower()
        power_structures = []
        
        for structure_type, indicators in self.power_indicators.items():
            found_indicators = [indicator for indicator in indicators if indicator in content_lower]
            
            if found_indicators:
                # Calculate power level based on indicators
                power_level = min(len(found_indicators) / 3.0, 1.0)  # Normalize
                
                # Assess bias potential
                bias_potential = self._calculate_bias_potential(structure_type, found_indicators, content_lower)
                
                power_structure = PowerStructure(
                    structure_type=structure_type,
                    actors=found_indicators,
                    power_level=power_level,
                    influence_scope=self._determine_influence_scope(found_indicators, content_lower),
                    bias_potential=bias_potential
                )
                
                power_structures.append(power_structure)
        
        return power_structures
    
    def _calculate_bias_potential(self, structure_type: str, indicators: List[str], content: str) -> float:
        """Calculate potential for bias based on power structure type and context"""
        # High-bias indicators
        high_bias_words = ["official", "established", "mainstream", "standard", "normal"]
        moderate_bias_words = ["traditional", "conventional", "accepted", "recognized"]
        
        high_bias_count = sum(1 for word in high_bias_words if word in content)
        moderate_bias_count = sum(1 for word in moderate_bias_words if word in content)
        
        bias_score = (high_bias_count * 0.8 + moderate_bias_count * 0.5) / 5.0
        
        # Adjust based on structure type
        structure_bias_multipliers = {
            "institutional": 1.2,  # Higher bias potential
            "economic": 1.1,
            "political": 1.3,      # Highest bias potential
            "social": 0.9
        }
        
        multiplier = structure_bias_multipliers.get(structure_type, 1.0)
        return min(bias_score * multiplier, 1.0)
    
    def _determine_influence_scope(self, indicators: List[str], content: str) -> str:
        """Determine the scope of influence for identified power structures"""
        scope_indicators = {
            "global": ["global", "international", "worldwide", "universal"],
            "national": ["national", "country", "federal", "state"],
            "local": ["local", "community", "city", "town", "regional"],
            "institutional": ["organization", "company", "university", "agency"]
        }
        
        for scope, scope_words in scope_indicators.items():
            if any(word in content for word in scope_words):
                return scope
        
        return "unclear"
    
    async def _analyze_perspective_diversity(self, claim: Claim) -> PerspectiveAnalysis:
        """Analyze diversity and inclusion of perspectives"""
        content_lower = claim.content.lower()
        
        represented_perspectives = []
        missing_perspectives = []
        
        # Check for represented perspectives
        for category, perspectives in self.perspective_categories.items():
            found_perspectives = [p for p in perspectives if p in content_lower]
            represented_perspectives.extend(found_perspectives)
        
        # Identify potentially missing perspectives
        if "policy" in content_lower or "government" in content_lower:
            expected_perspectives = ["citizens", "taxpayers", "affected communities"]
            for perspective in expected_perspectives:
                if perspective not in content_lower:
                    missing_perspectives.append(perspective)
        
        if "economic" in content_lower or "business" in content_lower:
            expected_perspectives = ["workers", "consumers", "communities"]
            for perspective in expected_perspectives:
                if perspective not in content_lower:
                    missing_perspectives.append(perspective)
        
        # Calculate diversity and inclusion scores
        diversity_score = min(len(set(represented_perspectives)) / 5.0, 1.0)  # Normalize by 5 perspectives
        inclusion_score = max(1.0 - (len(missing_perspectives) / 5.0), 0.0)
        
        # Identify dominant perspective
        dominant_perspective = self._identify_dominant_perspective(content_lower, represented_perspectives)
        
        return PerspectiveAnalysis(
            represented_perspectives=list(set(represented_perspectives)),
            missing_perspectives=list(set(missing_perspectives)),
            dominant_perspective=dominant_perspective,
            diversity_score=diversity_score,
            inclusion_score=inclusion_score
        )
    
    def _identify_dominant_perspective(self, content: str, perspectives: List[str]) -> Optional[str]:
        """Identify which perspective dominates the claim"""
        perspective_counts = Counter(perspectives)
        
        if not perspective_counts:
            # Check for implicit dominant perspectives
            if any(word in content for word in ["expert", "professional", "academic"]):
                return "expert/professional"
            elif any(word in content for word in ["business", "market", "profit"]):
                return "business/economic"
            elif any(word in content for word in ["government", "policy", "official"]):
                return "institutional/governmental"
            else:
                return None
        
        most_common = perspective_counts.most_common(1)[0]
        if most_common[1] > 1:  # Appears more than once
            return most_common[0]
        
        return None
    
    async def _assess_bias_indicators(self, claim: Claim) -> BiasAssessment:
        """Assess various types of bias in the claim"""
        content_lower = claim.content.lower()
        
        bias_scores = {}
        all_bias_indicators = []
        
        for bias_type, patterns in self.bias_patterns.items():
            found_patterns = [pattern for pattern in patterns if pattern in content_lower]
            bias_score = len(found_patterns) / max(len(patterns) / 3, 1)  # Normalize
            bias_scores[f"{bias_type}_bias"] = min(bias_score, 1.0)
            
            if found_patterns:
                all_bias_indicators.extend([f"{bias_type}: {pattern}" for pattern in found_patterns])
        
        # Calculate overall bias score
        if bias_scores:
            overall_bias_score = statistics.mean(bias_scores.values())
        else:
            overall_bias_score = 0.0
        
        return BiasAssessment(
            institutional_bias=bias_scores.get("institutional_bias", 0.0),
            economic_bias=bias_scores.get("economic_bias", 0.0),
            cultural_bias=bias_scores.get("cultural_bias", 0.0),
            gender_bias=bias_scores.get("gender_bias", 0.0),
            racial_bias=bias_scores.get("racial_bias", 0.0),
            class_bias=bias_scores.get("class_bias", 0.0),
            overall_bias_score=overall_bias_score,
            bias_indicators=all_bias_indicators
        )
    
    async def _identify_stakeholder_impact(self, claim: Claim) -> Dict[str, float]:
        """Identify stakeholders and assess impact levels"""
        content_lower = claim.content.lower()
        stakeholder_impact = {}
        
        # Direct stakeholder mentions
        for category, stakeholders in self.perspective_categories.items():
            for stakeholder in stakeholders:
                if stakeholder in content_lower:
                    # Assess impact level based on context
                    impact_level = self._assess_stakeholder_impact_level(stakeholder, content_lower)
                    stakeholder_impact[stakeholder] = impact_level
        
        return stakeholder_impact
    
    def _assess_stakeholder_impact_level(self, stakeholder: str, content: str) -> float:
        """Assess the level of impact on a specific stakeholder"""
        high_impact_words = ["affected", "impacted", "harmed", "benefited", "threatened"]
        moderate_impact_words = ["involved", "concerned", "related", "connected"]
        
        impact_score = 0.0
        
        # Look for impact words in proximity to stakeholder
        stakeholder_pos = content.find(stakeholder)
        if stakeholder_pos != -1:
            # Check 50 characters around the stakeholder mention
            context_window = content[max(0, stakeholder_pos-50):stakeholder_pos+50]
            
            high_impact_count = sum(1 for word in high_impact_words if word in context_window)
            moderate_impact_count = sum(1 for word in moderate_impact_words if word in context_window)
            
            impact_score = min((high_impact_count * 0.8 + moderate_impact_count * 0.4), 1.0)
        
        return impact_score
    
    def _combine_power_dynamics_scores(self, power_structures: List[PowerStructure],
                                     perspective_analysis: PerspectiveAnalysis,
                                     bias_assessment: BiasAssessment,
                                     stakeholder_impact: Dict[str, float]) -> float:
        """Combine all power dynamics scores"""
        scores = []
        weights = []
        
        # Power structure transparency (lower is better for democracy)
        if power_structures:
            avg_bias_potential = statistics.mean([ps.bias_potential for ps in power_structures])
            power_transparency_score = 1.0 - avg_bias_potential  # Invert bias potential
            scores.append(power_transparency_score)
            weights.append(0.25)
        
        # Perspective diversity score
        scores.append(perspective_analysis.diversity_score)
        weights.append(0.25)
        
        # Inclusion score
        scores.append(perspective_analysis.inclusion_score)
        weights.append(0.25)
        
        # Bias mitigation score (lower bias = higher score)
        bias_mitigation_score = 1.0 - bias_assessment.overall_bias_score
        scores.append(bias_mitigation_score)
        weights.append(0.25)
        
        if not scores:
            return 0.5  # Neutral score if no analysis possible
        
        return sum(s * w for s, w in zip(scores, weights)) / sum(weights)
    
    def _generate_power_dynamics_reasoning(self, power_structures: List[PowerStructure],
                                         perspective_analysis: PerspectiveAnalysis,
                                         bias_assessment: BiasAssessment,
                                         stakeholder_impact: Dict[str, float]) -> str:
        """Generate reasoning for power dynamics analysis"""
        reasoning_parts = []
        
        # Power structures
        if power_structures:
            structure_types = [ps.structure_type for ps in power_structures]
            reasoning_parts.append(f"Identified {len(power_structures)} power structures: {', '.join(set(structure_types))}")
            
            high_bias_structures = [ps for ps in power_structures if ps.bias_potential > 0.7]
            if high_bias_structures:
                reasoning_parts.append(f"High bias potential detected in {len(high_bias_structures)} power structures")
        
        # Perspective analysis
        if perspective_analysis.represented_perspectives:
            reasoning_parts.append(f"Represents {len(perspective_analysis.represented_perspectives)} stakeholder perspectives")
        
        if perspective_analysis.missing_perspectives:
            reasoning_parts.append(f"Missing {len(perspective_analysis.missing_perspectives)} key perspectives")
        
        if perspective_analysis.dominant_perspective:
            reasoning_parts.append(f"Dominated by {perspective_analysis.dominant_perspective} perspective")
        
        # Bias assessment
        if bias_assessment.bias_indicators:
            reasoning_parts.append(f"Detected {len(bias_assessment.bias_indicators)} bias indicators")
        
        # Stakeholder impact
        if stakeholder_impact:
            high_impact_stakeholders = [s for s, impact in stakeholder_impact.items() if impact > 0.7]
            if high_impact_stakeholders:
                reasoning_parts.append(f"High impact on: {', '.join(high_impact_stakeholders)}")
        
        return ". ".join(reasoning_parts) + "." if reasoning_parts else "Basic power dynamics analysis completed."
    
    def _identify_power_dynamics_uncertainties(self, power_structures: List[PowerStructure],
                                             perspective_analysis: PerspectiveAnalysis,
                                             bias_assessment: BiasAssessment) -> List[str]:
        """Identify uncertainties in power dynamics analysis"""
        uncertainties = []
        
        if not power_structures:
            uncertainties.append("No clear power structures identified")
        
        if perspective_analysis.diversity_score < 0.3:
            uncertainties.append("Limited perspective diversity")
        
        if perspective_analysis.inclusion_score < 0.5:
            uncertainties.append("Potentially excluded stakeholder perspectives")
        
        if bias_assessment.overall_bias_score > 0.7:
            uncertainties.append("High potential for systemic bias")
        
        if bias_assessment.institutional_bias > 0.8:
            uncertainties.append("Strong institutional bias detected")
        
        return uncertainties
    
    def _generate_power_dynamics_insights(self, claim: Claim,
                                        power_structures: List[PowerStructure],
                                        perspective_analysis: PerspectiveAnalysis,
                                        bias_assessment: BiasAssessment) -> str:
        """Generate insights about power dynamics"""
        insights = []
        
        # Power structure insights
        if power_structures:
            democratic_structures = [ps for ps in power_structures if ps.bias_potential < 0.3]
            if democratic_structures:
                insights.append("Contains democratically accountable power structures")
            
            concentrated_power = [ps for ps in power_structures if ps.power_level > 0.8]
            if concentrated_power:
                insights.append("Involves highly concentrated power structures")
        
        # Perspective insights
        if perspective_analysis.diversity_score > 0.7:
            insights.append("Demonstrates strong perspective diversity")
        elif perspective_analysis.diversity_score < 0.3:
            insights.append("Consider incorporating additional stakeholder perspectives")
        
        # Bias insights
        if bias_assessment.overall_bias_score < 0.3:
            insights.append("Low bias indicators suggest inclusive framing")
        elif bias_assessment.cultural_bias > 0.7:
            insights.append("Consider cultural sensitivity and global perspectives")
        
        return ". ".join(insights) + "." if insights else "Standard power dynamics analysis completed."
    
    def _collect_power_dynamics_references(self, power_structures: List[PowerStructure],
                                         perspective_analysis: PerspectiveAnalysis) -> List[str]:
        """Collect references for power dynamics evidence"""
        references = []
        
        # Power structure references
        for ps in power_structures[:3]:  # Limit to top 3
            references.append(f"{ps.structure_type.title()} power structure: {', '.join(ps.actors[:3])}")
        
        # Perspective references
        if perspective_analysis.represented_perspectives:
            references.append(f"Represented perspectives: {', '.join(perspective_analysis.represented_perspectives[:3])}")
        
        if perspective_analysis.missing_perspectives:
            references.append(f"Missing perspectives: {', '.join(perspective_analysis.missing_perspectives[:3])}")
        
        return references
    
    def _apply_power_dynamics_node_adjustments(self, score: float, node_context: NodeContext,
                                              result: VerificationResult) -> float:
        """Apply node-specific power dynamics adjustments"""
        power_dynamics_weight = node_context.philosophical_weights.get("power_dynamics", 1.0)
        
        # Nodes that highly value power dynamics awareness get bonus for inclusive analysis
        if power_dynamics_weight > 0.8:
            diversity_score = result.metadata.get("perspective_diversity", 0.0)
            inclusion_score = result.metadata.get("inclusion_score", 0.0)
            
            if diversity_score > 0.7 and inclusion_score > 0.7:
                score = min(score + 0.1, 1.0)  # Bonus for inclusive analysis
        
        return score * power_dynamics_weight
    
    def _assess_power_dynamics_evidence_quality(self, result: VerificationResult) -> float:
        """Assess quality of power dynamics evidence"""
        quality_factors = []
        
        # Power structure identification
        power_structures = result.metadata.get("power_structures_identified", 0)
        structure_quality = min(power_structures / 3.0, 1.0)
        quality_factors.append(structure_quality)
        
        # Perspective diversity
        diversity_score = result.metadata.get("perspective_diversity", 0.0)
        quality_factors.append(diversity_score)
        
        # Inclusion score
        inclusion_score = result.metadata.get("inclusion_score", 0.0)
        quality_factors.append(inclusion_score)
        
        # Bias awareness (lower bias score = higher quality)
        bias_score = result.metadata.get("overall_bias_score", 0.0)
        bias_quality = 1.0 - bias_score
        quality_factors.append(bias_quality)
        
        return statistics.mean(quality_factors)
    
    def _suggest_power_dynamics_refinements(self, result: VerificationResult,
                                          node_context: NodeContext) -> List[str]:
        """Suggest refinements for power dynamics verification"""
        suggestions = []
        
        if result.confidence_score < 0.7:
            suggestions.append("Consider additional stakeholder perspectives and power structure analysis")
        
        if "Limited perspective diversity" in result.uncertainty_factors:
            suggestions.append("Include voices from marginalized or affected communities")
        
        if "High potential for systemic bias" in result.uncertainty_factors:
            suggestions.append("Address potential biases and ensure inclusive language")
        
        if "Strong institutional bias detected" in result.uncertainty_factors:
            suggestions.append("Balance institutional perspectives with community and grassroots viewpoints")
        
        return suggestions
    
    def _generate_cache_key(self, claim: Claim) -> str:
        """Generate cache key for power dynamics analysis"""
        import hashlib
        content_hash = hashlib.md5(claim.content.encode()).hexdigest()
        context_hash = hashlib.md5(str(sorted(claim.context.items())).encode()).hexdigest()
        return f"power_dynamics_{content_hash}_{context_hash}"
    
    def clear_cache(self):
        """Clear power dynamics analysis cache"""
        self.cache.clear()
    
    def get_power_dynamics_stats(self) -> Dict[str, Any]:
        """Get power dynamics analysis statistics"""
        return {
            "cached_analyses": len(self.cache),
            "power_structure_categories": len(self.power_indicators),
            "bias_pattern_categories": len(self.bias_patterns),
            "perspective_categories": len(self.perspective_categories)
        }