"""
Real Utility Verification Framework

Evaluates practical consequences, actionability, and cost-benefit analysis
of claims and decisions.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import statistics

from ..individual.components.unified_base import UnifiedVerificationComponent
from ...consensus_types import Claim, VerificationResult, ConsensusProposal, NodeContext, ConsensusVerificationResult


class UtilityType(Enum):
    """Types of utility to evaluate"""
    PRACTICAL = "practical"
    ECONOMIC = "economic"
    SOCIAL = "social"
    ENVIRONMENTAL = "environmental"
    TEMPORAL = "temporal"
    DECISION = "decision"


@dataclass
class CostBenefitAnalysis:
    """Cost-benefit analysis results"""
    monetary_costs: float = 0.0
    monetary_benefits: float = 0.0
    time_costs: float = 0.0  # In hours
    social_costs: float = 0.0  # 0.0 to 1.0
    social_benefits: float = 0.0  # 0.0 to 1.0
    environmental_impact: float = 0.0  # -1.0 to 1.0 (negative = harmful)
    net_utility: float = 0.0
    uncertainty_range: Tuple[float, float] = (0.0, 0.0)


@dataclass
class ActionabilityAssessment:
    """Assessment of how actionable a claim or recommendation is"""
    clarity_score: float = 0.0  # How clear are the action steps?
    feasibility_score: float = 0.0  # How feasible are the actions?
    resource_requirements: Dict[str, float] = field(default_factory=dict)
    time_to_implementation: float = 0.0  # In days
    success_probability: float = 0.0
    required_stakeholders: List[str] = field(default_factory=list)


@dataclass
class OutcomeSimulation:
    """Simulation of potential outcomes"""
    scenario_name: str
    probability: float
    positive_outcomes: List[str] = field(default_factory=list)
    negative_outcomes: List[str] = field(default_factory=list)
    affected_parties: List[str] = field(default_factory=list)
    time_horizon: str = "short-term"  # short-term, medium-term, long-term
    confidence: float = 0.0


class RealUtilityFramework(UnifiedVerificationComponent):
    """
    Real utility verification evaluating practical consequences and actionability
    """
    
    def __init__(self):
        self.cache = {}
        self.logger = logging.getLogger(__name__)
        
        # Utility indicators
        self.utility_indicators = {
            "practical": [
                "useful", "practical", "applicable", "actionable", "implementable",
                "workable", "feasible", "realistic", "effective", "efficient"
            ],
            "economic": [
                "cost", "benefit", "profit", "loss", "investment", "return",
                "economic", "financial", "savings", "expense", "revenue"
            ],
            "social": [
                "benefit", "harm", "help", "improve", "worsen", "impact",
                "affect", "community", "society", "people", "public"
            ],
            "environmental": [
                "sustainable", "green", "eco-friendly", "pollution", "carbon",
                "environment", "climate", "ecosystem", "renewable", "conservation"
            ],
            "temporal": [
                "immediate", "short-term", "long-term", "future", "delayed",
                "quick", "fast", "slow", "gradual", "permanent", "temporary"
            ]
        }
        
        # Action indicators
        self.action_indicators = {
            "clear_actions": [
                "should", "must", "need to", "implement", "adopt", "establish",
                "create", "develop", "build", "start", "begin", "initiate"
            ],
            "vague_actions": [
                "consider", "explore", "investigate", "study", "examine",
                "review", "assess", "evaluate", "think about", "look into"
            ],
            "resource_requirements": [
                "budget", "funding", "money", "staff", "personnel", "time",
                "resources", "equipment", "infrastructure", "technology"
            ]
        }
        
        # Outcome indicators
        self.outcome_patterns = {
            "positive": [
                "improve", "better", "enhance", "increase", "boost", "benefit",
                "help", "solve", "fix", "optimize", "strengthen", "advance"
            ],
            "negative": [
                "worsen", "worse", "harm", "damage", "decrease", "reduce",
                "hurt", "risk", "threaten", "undermine", "weaken", "deteriorate"
            ],
            "uncertain": [
                "might", "could", "may", "possibly", "potentially", "perhaps",
                "unclear", "uncertain", "unknown", "depends", "varies"
            ]
        }
        
        # Cost-benefit patterns
        self.cost_benefit_patterns = {
            "monetary": [
                r'\$\d+(?:,\d{3})*(?:\.\d{2})?',  # Dollar amounts
                r'\d+(?:,\d{3})* (?:dollars?|cents?)',
                r'cost.*\$\d+', r'save.*\$\d+', r'profit.*\$\d+'
            ],
            "time": [
                r'\d+ (?:hours?|days?|weeks?|months?|years?)',
                r'(?:within|in|after) \d+ (?:hours?|days?|weeks?)',
                r'takes? \d+ (?:hours?|days?|weeks?)'
            ],
            "percentage": [
                r'\d+(?:\.\d+)?%',
                r'\d+(?:\.\d+)? percent'
            ]
        }
    
    async def verify_individual(self, claim: Claim) -> VerificationResult:
        """Verify claim through utility analysis"""
        try:
            # Check cache
            cache_key = self._generate_cache_key(claim)
            if cache_key in self.cache:
                self.logger.info(f"Using cached utility result for claim: {claim.content[:50]}...")
                return self.cache[cache_key]
            
            # Perform parallel utility analyses
            cost_benefit_task = self._analyze_cost_benefit(claim)
            actionability_task = self._assess_actionability(claim)
            outcome_task = self._simulate_outcomes(claim)
            resource_task = self._analyze_resource_requirements(claim)
            
            cost_benefit, actionability, outcomes, resources = await asyncio.gather(
                cost_benefit_task, actionability_task, outcome_task, resource_task,
                return_exceptions=True
            )
            
            # Handle exceptions
            if isinstance(cost_benefit, Exception):
                self.logger.warning(f"Cost-benefit analysis failed: {cost_benefit}")
                cost_benefit = CostBenefitAnalysis()
            if isinstance(actionability, Exception):
                self.logger.warning(f"Actionability assessment failed: {actionability}")
                actionability = ActionabilityAssessment()
            if isinstance(outcomes, Exception):
                self.logger.warning(f"Outcome simulation failed: {outcomes}")
                outcomes = []
            if isinstance(resources, Exception):
                self.logger.warning(f"Resource analysis failed: {resources}")
                resources = {}
            
            # Combine utility scores
            overall_score = self._combine_utility_scores(
                cost_benefit, actionability, outcomes, resources
            )
            
            # Generate reasoning
            reasoning = self._generate_utility_reasoning(
                cost_benefit, actionability, outcomes, resources
            )
            
            # Identify uncertainty factors
            uncertainty_factors = self._identify_utility_uncertainties(
                cost_benefit, actionability, outcomes
            )
            
            # Generate contextual notes
            contextual_notes = self._generate_utility_insights(
                claim, cost_benefit, actionability, outcomes
            )
            
            result = VerificationResult(
                framework_name="utility",
                confidence_score=overall_score,
                reasoning=reasoning,
                evidence_references=self._collect_utility_references(cost_benefit, actionability, outcomes),
                uncertainty_factors=uncertainty_factors,
                contextual_notes=contextual_notes,
                metadata={
                    "net_utility": cost_benefit.net_utility,
                    "actionability_score": actionability.feasibility_score,
                    "success_probability": actionability.success_probability,
                    "time_to_implementation": actionability.time_to_implementation,
                    "outcome_scenarios": len(outcomes),
                    "resource_types_identified": len(resources),
                    "verification_timestamp": datetime.utcnow().isoformat()
                }
            )
            
            # Cache result
            self.cache[cache_key] = result
            
            return result
            
        except Exception as e:
            self.logger.error(f"Utility verification failed: {e}")
            return VerificationResult(
                framework_name="utility",
                confidence_score=0.1,
                reasoning=f"Utility verification failed: {str(e)}",
                evidence_references=[],
                uncertainty_factors=["Technical failure", "Unable to analyze utility"],
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
        
        # Apply node-specific utility adjustments
        node_adjusted_score = self._apply_utility_node_adjustments(
            individual_result.confidence_score, node_context, individual_result
        )
        
        return ConsensusVerificationResult(
            node_id=node_context.node_id,
            framework_name="utility",
            confidence_score=node_adjusted_score,
            reasoning=individual_result.reasoning,
            evidence_quality=self._assess_utility_evidence_quality(individual_result),
            consensus_readiness=node_adjusted_score > 0.6,
            suggested_refinements=self._suggest_utility_refinements(individual_result, node_context),
            metadata={
                **individual_result.metadata,
                "node_utility_focus": True,
                "consensus_mode": True,
                "node_utility_weight": node_context.philosophical_weights.get("utility", 1.0)
            }
        )
    
    async def _analyze_cost_benefit(self, claim: Claim) -> CostBenefitAnalysis:
        """Analyze costs and benefits mentioned in the claim"""
        content = claim.content
        content_lower = content.lower()
        
        # Extract monetary values
        monetary_costs, monetary_benefits = self._extract_monetary_values(content)
        
        # Extract time costs
        time_costs = self._extract_time_values(content)
        
        # Assess social costs and benefits
        social_costs, social_benefits = self._assess_social_impact(content_lower)
        
        # Assess environmental impact
        environmental_impact = self._assess_environmental_impact(content_lower)
        
        # Calculate net utility
        net_utility = self._calculate_net_utility(
            monetary_costs, monetary_benefits, time_costs,
            social_costs, social_benefits, environmental_impact
        )
        
        # Estimate uncertainty range
        uncertainty_range = self._estimate_uncertainty_range(net_utility, content_lower)
        
        return CostBenefitAnalysis(
            monetary_costs=monetary_costs,
            monetary_benefits=monetary_benefits,
            time_costs=time_costs,
            social_costs=social_costs,
            social_benefits=social_benefits,
            environmental_impact=environmental_impact,
            net_utility=net_utility,
            uncertainty_range=uncertainty_range
        )
    
    def _extract_monetary_values(self, content: str) -> Tuple[float, float]:
        """Extract monetary costs and benefits from text"""
        import re
        
        # Simple extraction - in real implementation would use more sophisticated NLP
        cost_patterns = [
            r'cost.*\$(\d+(?:,\d{3})*(?:\.\d{2})?)',
            r'expense.*\$(\d+(?:,\d{3})*(?:\.\d{2})?)',
            r'spend.*\$(\d+(?:,\d{3})*(?:\.\d{2})?)'
        ]
        
        benefit_patterns = [
            r'save.*\$(\d+(?:,\d{3})*(?:\.\d{2})?)',
            r'profit.*\$(\d+(?:,\d{3})*(?:\.\d{2})?)',
            r'benefit.*\$(\d+(?:,\d{3})*(?:\.\d{2})?)'
        ]
        
        costs = []
        benefits = []
        
        for pattern in cost_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            for match in matches:
                try:
                    amount = float(match.replace(',', ''))
                    costs.append(amount)
                except ValueError:
                    continue
        
        for pattern in benefit_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            for match in matches:
                try:
                    amount = float(match.replace(',', ''))
                    benefits.append(amount)
                except ValueError:
                    continue
        
        total_costs = sum(costs) if costs else 0.0
        total_benefits = sum(benefits) if benefits else 0.0
        
        return total_costs, total_benefits
    
    def _extract_time_values(self, content: str) -> float:
        """Extract time costs from text"""
        import re
        
        time_patterns = [
            r'(\d+)\s*(?:hours?|hrs?)',
            r'(\d+)\s*(?:days?)',
            r'(\d+)\s*(?:weeks?)',
            r'(\d+)\s*(?:months?)',
            r'(\d+)\s*(?:years?)'
        ]
        
        # Convert everything to hours for consistency
        time_multipliers = {
            'hour': 1, 'hr': 1,
            'day': 24,
            'week': 168,  # 24 * 7
            'month': 720,  # 24 * 30
            'year': 8760  # 24 * 365
        }
        
        total_hours = 0.0
        
        for pattern in time_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            for match in matches:
                try:
                    amount = float(match)
                    # Determine unit from pattern
                    if 'hour' in pattern or 'hr' in pattern:
                        total_hours += amount * 1
                    elif 'day' in pattern:
                        total_hours += amount * 24
                    elif 'week' in pattern:
                        total_hours += amount * 168
                    elif 'month' in pattern:
                        total_hours += amount * 720
                    elif 'year' in pattern:
                        total_hours += amount * 8760
                except ValueError:
                    continue
        
        return total_hours
    
    def _assess_social_impact(self, content: str) -> Tuple[float, float]:
        """Assess social costs and benefits"""
        positive_social_words = [
            "community", "benefit", "help", "improve", "support", "assist",
            "collaboration", "cooperation", "inclusion", "equity", "fairness"
        ]
        
        negative_social_words = [
            "harm", "damage", "hurt", "exclude", "discriminate", "inequality",
            "conflict", "division", "unfair", "burden", "stress", "anxiety"
        ]
        
        positive_count = sum(1 for word in positive_social_words if word in content)
        negative_count = sum(1 for word in negative_social_words if word in content)
        
        # Normalize to 0-1 scale
        social_benefits = min(positive_count / 5.0, 1.0)
        social_costs = min(negative_count / 5.0, 1.0)
        
        return social_costs, social_benefits
    
    def _assess_environmental_impact(self, content: str) -> float:
        """Assess environmental impact (-1 to 1, negative = harmful)"""
        positive_env_words = [
            "sustainable", "green", "eco-friendly", "renewable", "conservation",
            "protect", "preserve", "clean", "reduce emissions", "carbon neutral"
        ]
        
        negative_env_words = [
            "pollution", "contamination", "emissions", "waste", "toxic",
            "harmful", "destructive", "unsustainable", "fossil fuel"
        ]
        
        positive_count = sum(1 for word in positive_env_words if word in content)
        negative_count = sum(1 for word in negative_env_words if word in content)
        
        # Calculate net environmental impact
        if positive_count + negative_count == 0:
            return 0.0  # Neutral if no environmental mentions
        
        net_impact = (positive_count - negative_count) / max(positive_count + negative_count, 1)
        return max(-1.0, min(1.0, net_impact))
    
    def _calculate_net_utility(self, monetary_costs: float, monetary_benefits: float,
                              time_costs: float, social_costs: float, social_benefits: float,
                              environmental_impact: float) -> float:
        """Calculate overall net utility score"""
        # Normalize monetary values (simplified)
        if monetary_costs + monetary_benefits > 0:
            monetary_utility = (monetary_benefits - monetary_costs) / (monetary_benefits + monetary_costs)
        else:
            monetary_utility = 0.0
        
        # Normalize time costs (assume 40 hours = significant cost)
        time_utility = max(-1.0, min(1.0, -time_costs / 40.0))
        
        # Social utility
        social_utility = social_benefits - social_costs
        
        # Combine utilities with weights
        weights = [0.3, 0.2, 0.3, 0.2]  # monetary, time, social, environmental
        utilities = [monetary_utility, time_utility, social_utility, environmental_impact]
        
        net_utility = sum(u * w for u, w in zip(utilities, weights))
        
        # Normalize to 0-1 scale (shift from -1,1 to 0,1)
        return (net_utility + 1.0) / 2.0
    
    def _estimate_uncertainty_range(self, net_utility: float, content: str) -> Tuple[float, float]:
        """Estimate uncertainty range around net utility"""
        uncertainty_indicators = [
            "uncertain", "unclear", "depends", "varies", "might", "could",
            "estimate", "approximately", "roughly", "about"
        ]
        
        uncertainty_count = sum(1 for word in uncertainty_indicators if word in content)
        
        # Higher uncertainty = wider range
        uncertainty_factor = min(uncertainty_count / 5.0, 0.5)  # Max 50% uncertainty
        
        lower_bound = max(0.0, net_utility - uncertainty_factor)
        upper_bound = min(1.0, net_utility + uncertainty_factor)
        
        return (lower_bound, upper_bound)
    
    async def _assess_actionability(self, claim: Claim) -> ActionabilityAssessment:
        """Assess how actionable the claim or its recommendations are"""
        content_lower = claim.content.lower()
        
        # Assess clarity of actions
        clear_action_count = sum(1 for indicator in self.action_indicators["clear_actions"] 
                                if indicator in content_lower)
        vague_action_count = sum(1 for indicator in self.action_indicators["vague_actions"] 
                                if indicator in content_lower)
        
        total_actions = clear_action_count + vague_action_count
        clarity_score = clear_action_count / max(total_actions, 1) if total_actions > 0 else 0.0
        
        # Assess feasibility
        feasibility_score = self._assess_feasibility(content_lower)
        
        # Identify resource requirements
        resource_requirements = self._identify_resource_requirements(content_lower)
        
        # Estimate time to implementation
        time_to_implementation = self._estimate_implementation_time(content_lower)
        
        # Estimate success probability
        success_probability = self._estimate_success_probability(content_lower, feasibility_score)
        
        # Identify required stakeholders
        required_stakeholders = self._identify_required_stakeholders(content_lower)
        
        return ActionabilityAssessment(
            clarity_score=clarity_score,
            feasibility_score=feasibility_score,
            resource_requirements=resource_requirements,
            time_to_implementation=time_to_implementation,
            success_probability=success_probability,
            required_stakeholders=required_stakeholders
        )
    
    def _assess_feasibility(self, content: str) -> float:
        """Assess feasibility of proposed actions"""
        high_feasibility_words = [
            "simple", "easy", "straightforward", "feasible", "practical",
            "realistic", "achievable", "doable", "manageable"
        ]
        
        low_feasibility_words = [
            "complex", "difficult", "challenging", "impossible", "unrealistic",
            "impractical", "unfeasible", "overwhelming", "complicated"
        ]
        
        high_count = sum(1 for word in high_feasibility_words if word in content)
        low_count = sum(1 for word in low_feasibility_words if word in content)
        
        if high_count + low_count == 0:
            return 0.5  # Neutral if no feasibility indicators
        
        return high_count / (high_count + low_count)
    
    def _identify_resource_requirements(self, content: str) -> Dict[str, float]:
        """Identify resource requirements mentioned in content"""
        resources = {}
        
        resource_types = {
            "financial": ["money", "budget", "funding", "investment", "cost"],
            "human": ["staff", "personnel", "people", "team", "workforce"],
            "time": ["time", "schedule", "deadline", "duration"],
            "technical": ["technology", "equipment", "infrastructure", "system"],
            "expertise": ["expert", "specialist", "skill", "knowledge", "training"]
        }
        
        for resource_type, keywords in resource_types.items():
            mentions = sum(1 for keyword in keywords if keyword in content)
            if mentions > 0:
                # Simple scoring: more mentions = higher requirement
                resources[resource_type] = min(mentions / 3.0, 1.0)
        
        return resources
    
    def _estimate_implementation_time(self, content: str) -> float:
        """Estimate time to implementation in days"""
        import re
        
        # Look for time indicators
        immediate_patterns = ["immediate", "now", "today", "urgent"]
        short_term_patterns = ["days?", "weeks?", "soon", "quickly"]
        medium_term_patterns = ["months?", "quarter", "semester"]
        long_term_patterns = ["years?", "decade", "long.term"]
        
        if any(pattern in content for pattern in immediate_patterns):
            return 1.0  # 1 day
        elif any(pattern in content for pattern in short_term_patterns):
            return 30.0  # 1 month
        elif any(pattern in content for pattern in medium_term_patterns):
            return 180.0  # 6 months
        elif any(pattern in content for pattern in long_term_patterns):
            return 365.0  # 1 year
        else:
            return 90.0  # Default 3 months
    
    def _estimate_success_probability(self, content: str, feasibility_score: float) -> float:
        """Estimate probability of successful implementation"""
        confidence_indicators = ["certain", "sure", "confident", "proven", "tested"]
        uncertainty_indicators = ["uncertain", "risky", "experimental", "untested"]
        
        confidence_count = sum(1 for indicator in confidence_indicators if indicator in content)
        uncertainty_count = sum(1 for indicator in uncertainty_indicators if indicator in content)
        
        # Base probability on feasibility
        base_probability = feasibility_score
        
        # Adjust based on confidence indicators
        if confidence_count > uncertainty_count:
            adjustment = 0.2
        elif uncertainty_count > confidence_count:
            adjustment = -0.2
        else:
            adjustment = 0.0
        
        return max(0.1, min(0.9, base_probability + adjustment))
    
    def _identify_required_stakeholders(self, content: str) -> List[str]:
        """Identify stakeholders required for implementation"""
        stakeholder_keywords = {
            "government": ["government", "agency", "department", "official", "policy"],
            "business": ["company", "corporation", "industry", "business", "private"],
            "community": ["community", "public", "citizen", "resident", "local"],
            "experts": ["expert", "specialist", "researcher", "academic", "professional"],
            "funding": ["investor", "funder", "donor", "sponsor", "financier"]
        }
        
        required_stakeholders = []
        for stakeholder_type, keywords in stakeholder_keywords.items():
            if any(keyword in content for keyword in keywords):
                required_stakeholders.append(stakeholder_type)
        
        return required_stakeholders
    
    async def _simulate_outcomes(self, claim: Claim) -> List[OutcomeSimulation]:
        """Simulate potential outcomes of the claim or its recommendations"""
        content_lower = claim.content.lower()
        outcomes = []
        
        # Identify outcome indicators
        positive_outcomes = []
        negative_outcomes = []
        
        for outcome in self.outcome_patterns["positive"]:
            if outcome in content_lower:
                positive_outcomes.append(f"Potential for {outcome}ment")
        
        for outcome in self.outcome_patterns["negative"]:
            if outcome in content_lower:
                negative_outcomes.append(f"Risk of {outcome}ing")
        
        # Create optimistic scenario
        if positive_outcomes:
            outcomes.append(OutcomeSimulation(
                scenario_name="Optimistic",
                probability=0.3,
                positive_outcomes=positive_outcomes,
                negative_outcomes=[],
                time_horizon="medium-term",
                confidence=0.6
            ))
        
        # Create pessimistic scenario
        if negative_outcomes:
            outcomes.append(OutcomeSimulation(
                scenario_name="Pessimistic",
                probability=0.2,
                positive_outcomes=[],
                negative_outcomes=negative_outcomes,
                time_horizon="short-term",
                confidence=0.7
            ))
        
        # Create realistic scenario
        outcomes.append(OutcomeSimulation(
            scenario_name="Realistic",
            probability=0.5,
            positive_outcomes=positive_outcomes[:2],  # Limit optimism
            negative_outcomes=negative_outcomes[:1],  # Some challenges
            time_horizon="medium-term",
            confidence=0.8
        ))
        
        return outcomes
    
    async def _analyze_resource_requirements(self, claim: Claim) -> Dict[str, Any]:
        """Analyze resource requirements in detail"""
        content_lower = claim.content.lower()
        
        resources = {
            "complexity": "medium",
            "scalability": "unclear",
            "dependencies": [],
            "critical_resources": []
        }
        
        # Assess complexity
        complexity_indicators = {
            "simple": ["simple", "easy", "basic", "straightforward"],
            "medium": ["moderate", "standard", "typical", "normal"],
            "complex": ["complex", "complicated", "sophisticated", "advanced"]
        }
        
        for complexity_level, indicators in complexity_indicators.items():
            if any(indicator in content_lower for indicator in indicators):
                resources["complexity"] = complexity_level
                break
        
        # Identify dependencies
        dependency_words = ["requires", "depends", "needs", "relies on", "prerequisite"]
        if any(word in content_lower for word in dependency_words):
            resources["dependencies"] = ["external_systems", "stakeholder_cooperation"]
        
        return resources
    
    def _combine_utility_scores(self, cost_benefit: CostBenefitAnalysis,
                               actionability: ActionabilityAssessment,
                               outcomes: List[OutcomeSimulation],
                               resources: Dict[str, Any]) -> float:
        """Combine all utility scores into overall utility score"""
        scores = []
        weights = []
        
        # Cost-benefit score
        scores.append(cost_benefit.net_utility)
        weights.append(0.4)
        
        # Actionability score
        actionability_score = (actionability.clarity_score + actionability.feasibility_score + 
                              actionability.success_probability) / 3.0
        scores.append(actionability_score)
        weights.append(0.3)
        
        # Outcome positivity
        if outcomes:
            positive_outcomes = sum(len(o.positive_outcomes) for o in outcomes)
            negative_outcomes = sum(len(o.negative_outcomes) for o in outcomes)
            total_outcomes = positive_outcomes + negative_outcomes
            
            if total_outcomes > 0:
                outcome_score = positive_outcomes / total_outcomes
            else:
                outcome_score = 0.5
            
            scores.append(outcome_score)
            weights.append(0.2)
        
        # Resource efficiency (simpler = more efficient)
        resource_efficiency = 0.8 if resources.get("complexity") == "simple" else 0.5
        scores.append(resource_efficiency)
        weights.append(0.1)
        
        return sum(s * w for s, w in zip(scores, weights)) / sum(weights)
    
    def _generate_utility_reasoning(self, cost_benefit: CostBenefitAnalysis,
                                   actionability: ActionabilityAssessment,
                                   outcomes: List[OutcomeSimulation],
                                   resources: Dict[str, Any]) -> str:
        """Generate reasoning for utility analysis"""
        reasoning_parts = []
        
        # Cost-benefit summary
        if cost_benefit.net_utility > 0.7:
            reasoning_parts.append("Strong positive utility with favorable cost-benefit ratio")
        elif cost_benefit.net_utility < 0.3:
            reasoning_parts.append("Limited utility with concerning cost-benefit analysis")
        else:
            reasoning_parts.append("Moderate utility with balanced costs and benefits")
        
        # Actionability summary
        if actionability.clarity_score > 0.7 and actionability.feasibility_score > 0.7:
            reasoning_parts.append("Highly actionable with clear implementation path")
        elif actionability.clarity_score < 0.3 or actionability.feasibility_score < 0.3:
            reasoning_parts.append("Limited actionability requiring further specification")
        else:
            reasoning_parts.append("Moderately actionable with some implementation challenges")
        
        # Outcome summary
        if outcomes:
            positive_scenarios = [o for o in outcomes if len(o.positive_outcomes) > len(o.negative_outcomes)]
            if len(positive_scenarios) > len(outcomes) / 2:
                reasoning_parts.append("Outcome scenarios generally favorable")
            else:
                reasoning_parts.append("Mixed outcome scenarios with significant uncertainties")
        
        # Resource summary
        if actionability.resource_requirements:
            resource_types = list(actionability.resource_requirements.keys())
            reasoning_parts.append(f"Requires {', '.join(resource_types)} resources")
        
        return ". ".join(reasoning_parts) + "."
    
    def _identify_utility_uncertainties(self, cost_benefit: CostBenefitAnalysis,
                                       actionability: ActionabilityAssessment,
                                       outcomes: List[OutcomeSimulation]) -> List[str]:
        """Identify uncertainties in utility analysis"""
        uncertainties = []
        
        # Cost-benefit uncertainties
        uncertainty_range = cost_benefit.uncertainty_range[1] - cost_benefit.uncertainty_range[0]
        if uncertainty_range > 0.3:
            uncertainties.append("High uncertainty in cost-benefit calculations")
        
        # Actionability uncertainties
        if actionability.success_probability < 0.5:
            uncertainties.append("Low probability of successful implementation")
        
        if actionability.time_to_implementation > 365:
            uncertainties.append("Long implementation timeline increases uncertainty")
        
        # Resource uncertainties
        if not actionability.resource_requirements:
            uncertainties.append("Unclear resource requirements")
        
        # Outcome uncertainties
        if outcomes:
            low_confidence_outcomes = [o for o in outcomes if o.confidence < 0.6]
            if len(low_confidence_outcomes) > len(outcomes) / 2:
                uncertainties.append("Low confidence in outcome predictions")
        
        return uncertainties
    
    def _generate_utility_insights(self, claim: Claim,
                                  cost_benefit: CostBenefitAnalysis,
                                  actionability: ActionabilityAssessment,
                                  outcomes: List[OutcomeSimulation]) -> str:
        """Generate insights about utility analysis"""
        insights = []
        
        # Time horizon insights
        if actionability.time_to_implementation < 30:
            insights.append("Short implementation timeline enables rapid value delivery")
        elif actionability.time_to_implementation > 180:
            insights.append("Long implementation timeline requires sustained commitment")
        
        # Stakeholder insights
        if len(actionability.required_stakeholders) > 3:
            insights.append("Complex stakeholder coordination required for success")
        elif len(actionability.required_stakeholders) == 1:
            insights.append("Simple stakeholder structure facilitates implementation")
        
        # Environmental insights
        if cost_benefit.environmental_impact > 0.5:
            insights.append("Positive environmental impact adds long-term value")
        elif cost_benefit.environmental_impact < -0.5:
            insights.append("Environmental concerns may limit long-term utility")
        
        # Social insights
        if cost_benefit.social_benefits > cost_benefit.social_costs:
            insights.append("Social benefits outweigh costs, supporting public value")
        
        return ". ".join(insights) + "." if insights else "Standard utility analysis completed."
    
    def _collect_utility_references(self, cost_benefit: CostBenefitAnalysis,
                                   actionability: ActionabilityAssessment,
                                   outcomes: List[OutcomeSimulation]) -> List[str]:
        """Collect references for utility evidence"""
        references = []
        
        # Cost-benefit references
        if cost_benefit.monetary_benefits > 0 or cost_benefit.monetary_costs > 0:
            references.append(f"Monetary analysis: ${cost_benefit.monetary_benefits:.0f} benefits vs ${cost_benefit.monetary_costs:.0f} costs")
        
        # Actionability references
        references.append(f"Implementation feasibility: {actionability.feasibility_score:.2f}")
        references.append(f"Success probability: {actionability.success_probability:.2f}")
        
        # Outcome references
        for outcome in outcomes[:2]:  # Limit to top 2 scenarios
            references.append(f"{outcome.scenario_name} scenario: {outcome.probability:.1f} probability")
        
        return references
    
    def _apply_utility_node_adjustments(self, score: float, node_context: NodeContext,
                                       result: VerificationResult) -> float:
        """Apply node-specific utility adjustments"""
        utility_weight = node_context.philosophical_weights.get("utility", 1.0)
        
        # Nodes that highly value utility get bonus for practical analysis
        if utility_weight > 0.8:
            actionability_score = result.metadata.get("actionability_score", 0.0)
            net_utility = result.metadata.get("net_utility", 0.0)
            
            if actionability_score > 0.7 and net_utility > 0.7:
                score = min(score + 0.1, 1.0)  # Bonus for highly practical claims
        
        return score * utility_weight
    
    def _assess_utility_evidence_quality(self, result: VerificationResult) -> float:
        """Assess quality of utility evidence"""
        quality_factors = []
        
        # Cost-benefit clarity
        net_utility = result.metadata.get("net_utility", 0.0)
        if net_utility != 0.0:  # Has actual cost-benefit analysis
            quality_factors.append(0.8)
        else:
            quality_factors.append(0.3)
        
        # Actionability assessment
        actionability_score = result.metadata.get("actionability_score", 0.0)
        quality_factors.append(actionability_score)
        
        # Success probability
        success_probability = result.metadata.get("success_probability", 0.0)
        quality_factors.append(success_probability)
        
        # Outcome scenarios
        scenario_count = result.metadata.get("outcome_scenarios", 0)
        scenario_quality = min(scenario_count / 3.0, 1.0)
        quality_factors.append(scenario_quality)
        
        return statistics.mean(quality_factors)
    
    def _suggest_utility_refinements(self, result: VerificationResult,
                                    node_context: NodeContext) -> List[str]:
        """Suggest refinements for utility verification"""
        suggestions = []
        
        if result.confidence_score < 0.7:
            suggestions.append("Provide more specific cost-benefit analysis and implementation details")
        
        if "High uncertainty in cost-benefit calculations" in result.uncertainty_factors:
            suggestions.append("Include quantitative estimates and uncertainty ranges")
        
        if "Low probability of successful implementation" in result.uncertainty_factors:
            suggestions.append("Address implementation challenges and provide mitigation strategies")
        
        if "Unclear resource requirements" in result.uncertainty_factors:
            suggestions.append("Specify required resources, timeline, and stakeholder commitments")
        
        return suggestions
    
    def _generate_cache_key(self, claim: Claim) -> str:
        """Generate cache key for utility analysis"""
        import hashlib
        content_hash = hashlib.md5(claim.content.encode()).hexdigest()
        context_hash = hashlib.md5(str(sorted(claim.context.items())).encode()).hexdigest()
        return f"utility_{content_hash}_{context_hash}"
    
    def clear_cache(self):
        """Clear utility analysis cache"""
        self.cache.clear()
    
    def get_utility_stats(self) -> Dict[str, Any]:
        """Get utility analysis statistics"""
        return {
            "cached_analyses": len(self.cache),
            "utility_indicator_categories": len(self.utility_indicators),
            "action_indicator_categories": len(self.action_indicators),
            "outcome_pattern_categories": len(self.outcome_patterns)
        }