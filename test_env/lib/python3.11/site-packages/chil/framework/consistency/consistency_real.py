"""
Real Consistency Verification Framework

Implements logical consistency checking, contradiction detection, and formal reasoning validation.
"""

import asyncio
import logging
import re
from datetime import datetime
from typing import List, Dict, Any, Optional, Set, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import statistics

from .unified_base import UnifiedVerificationComponent
from ...consensus_types import Claim, VerificationResult, ConsensusProposal, NodeContext, ConsensusVerificationResult


class LogicalOperator(Enum):
    """Logical operators detected in claims"""
    AND = "and"
    OR = "or"  
    NOT = "not"
    IF_THEN = "if_then"
    IFF = "if_and_only_if"
    IMPLIES = "implies"
    BECAUSE = "because"
    THEREFORE = "therefore"


class ConsistencyIssueType(Enum):
    """Types of consistency issues"""
    LOGICAL_CONTRADICTION = "logical_contradiction"
    CIRCULAR_REASONING = "circular_reasoning"
    INVALID_INFERENCE = "invalid_inference"
    MISSING_PREMISE = "missing_premise"
    AMBIGUOUS_TERMS = "ambiguous_terms"
    FALSE_DICHOTOMY = "false_dichotomy"
    HASTY_GENERALIZATION = "hasty_generalization"
    AD_HOC_REASONING = "ad_hoc_reasoning"


@dataclass
class LogicalStructure:
    """Representation of logical structure in a claim"""
    premises: List[str] = field(default_factory=list)
    conclusions: List[str] = field(default_factory=list)
    operators: List[LogicalOperator] = field(default_factory=list)
    conditional_statements: List[Tuple[str, str]] = field(default_factory=list)  # (condition, consequence)
    negations: List[str] = field(default_factory=list)


@dataclass
class ConsistencyIssue:
    """Identified consistency issue"""
    issue_type: ConsistencyIssueType
    description: str
    severity: float  # 0.0 to 1.0, higher = more severe
    location: str  # Text where issue occurs
    suggested_fix: Optional[str] = None


@dataclass
class LogicalValidation:
    """Result of logical validation"""
    is_valid: bool
    validity_score: float  # 0.0 to 1.0
    logical_form: str
    validation_method: str
    issues_found: List[str] = field(default_factory=list)


@dataclass
class ContradictionAnalysis:
    """Analysis of potential contradictions"""
    contradictions_found: List[Tuple[str, str]] = field(default_factory=list)  # (statement1, statement2)
    contradiction_score: float = 0.0  # 0.0 = no contradictions, 1.0 = severe contradictions
    resolution_suggestions: List[str] = field(default_factory=list)


class RealConsistencyFramework(UnifiedVerificationComponent):
    """
    Real consistency verification using logical analysis and contradiction detection
    """
    
    def __init__(self, strict_mode: bool = False):
        self.strict_mode = strict_mode
        self.cache = {}
        self.logger = logging.getLogger(__name__)
        
        # Initialize logical pattern matching
        self._init_logical_patterns()
        
        # Common logical fallacies patterns
        self.fallacy_patterns = {
            "circular_reasoning": [
                r"because.*\1",  # X because X
                r"proves.*itself",
                r"by definition.*therefore"
            ],
            "false_dichotomy": [
                r"either.*or.*no other",
                r"only two.*options",
                r"must be.*or.*nothing else"
            ],
            "hasty_generalization": [
                r"all.*always",
                r"never.*any",
                r"every.*without exception",
                r"one.*therefore.*all"
            ],
            "ad_hominem": [
                r"says.*therefore wrong",
                r"biased.*so.*invalid",
                r"because.*who said it"
            ],
            "appeal_to_authority": [
                r"expert.*says.*so.*true",
                r"authority.*claims.*therefore",
                r"famous.*believes.*must be"
            ]
        }
        
        # Contradiction indicators
        self.contradiction_indicators = {
            "direct_negation": [
                "not", "isn't", "aren't", "doesn't", "don't", "won't", "can't", "shouldn't"
            ],
            "opposing_terms": [
                ("always", "never"), ("all", "none"), ("everything", "nothing"),
                ("increase", "decrease"), ("rise", "fall"), ("more", "less"),
                ("positive", "negative"), ("true", "false"), ("yes", "no")
            ],
            "temporal_contradiction": [
                ("before", "after"), ("early", "late"), ("past", "future"),
                ("previous", "next"), ("old", "new"), ("ancient", "modern")
            ]
        }
        
        # Logical coherence indicators
        self.coherence_indicators = {
            "causal_chains": [
                "because", "since", "due to", "caused by", "results in", "leads to",
                "therefore", "thus", "hence", "consequently", "as a result"
            ],
            "evidence_markers": [
                "evidence shows", "data indicates", "studies reveal", "research demonstrates",
                "statistics show", "experiments prove", "observations suggest"
            ],
            "qualification_markers": [
                "however", "although", "despite", "nevertheless", "on the other hand",
                "but", "yet", "still", "while", "whereas"
            ]
        }
    
    def _init_logical_patterns(self):
        """Initialize patterns for logical structure detection"""
        
        # Conditional patterns (if-then, etc.)
        self.conditional_patterns = {
            "if_then": re.compile(r'if\s+(.+?)\s+then\s+(.+?)(?:[.!?]|$)', re.IGNORECASE),
            "when_then": re.compile(r'when\s+(.+?)\s+then\s+(.+?)(?:[.!?]|$)', re.IGNORECASE),
            "implies": re.compile(r'(.+?)\s+implies\s+(.+?)(?:[.!?]|$)', re.IGNORECASE),
            "because": re.compile(r'(.+?)\s+because\s+(.+?)(?:[.!?]|$)', re.IGNORECASE),
            "since": re.compile(r'(.+?)\s+since\s+(.+?)(?:[.!?]|$)', re.IGNORECASE)
        }
        
        # Conclusion markers
        self.conclusion_patterns = [
            re.compile(r'therefore\s+(.+?)(?:[.!?]|$)', re.IGNORECASE),
            re.compile(r'thus\s+(.+?)(?:[.!?]|$)', re.IGNORECASE),
            re.compile(r'hence\s+(.+?)(?:[.!?]|$)', re.IGNORECASE),
            re.compile(r'consequently\s+(.+?)(?:[.!?]|$)', re.IGNORECASE),
            re.compile(r'as a result\s+(.+?)(?:[.!?]|$)', re.IGNORECASE)
        ]
        
        # Universal/existential quantifiers
        self.quantifier_patterns = {
            "universal": re.compile(r'\b(all|every|each|any)\s+(.+?)\s+(are|is|have|has)\s+(.+?)(?:[.!?]|$)', re.IGNORECASE),
            "existential": re.compile(r'\b(some|many|few|several)\s+(.+?)\s+(are|is|have|has)\s+(.+?)(?:[.!?]|$)', re.IGNORECASE),
            "negative_universal": re.compile(r'\b(no|none|nothing)\s+(.+?)(?:[.!?]|$)', re.IGNORECASE)
        }
    
    async def verify_individual(self, claim: Claim) -> VerificationResult:
        """Verify claim through logical consistency analysis"""
        try:
            # Check cache
            cache_key = self._generate_cache_key(claim)
            if cache_key in self.cache:
                self.logger.info(f"Using cached consistency result for claim: {claim.content[:50]}...")
                return self.cache[cache_key]
            
            # Perform parallel consistency analyses
            structure_task = self._analyze_logical_structure(claim)
            contradiction_task = self._detect_contradictions(claim)
            validation_task = self._validate_logical_form(claim)
            fallacy_task = self._detect_logical_fallacies(claim)
            
            logical_structure, contradiction_analysis, logical_validation, fallacies = await asyncio.gather(
                structure_task, contradiction_task, validation_task, fallacy_task,
                return_exceptions=True
            )
            
            # Handle exceptions
            if isinstance(logical_structure, Exception):
                self.logger.warning(f"Logical structure analysis failed: {logical_structure}")
                logical_structure = LogicalStructure()
            if isinstance(contradiction_analysis, Exception):
                self.logger.warning(f"Contradiction analysis failed: {contradiction_analysis}")
                contradiction_analysis = ContradictionAnalysis()
            if isinstance(logical_validation, Exception):
                self.logger.warning(f"Logical validation failed: {logical_validation}")
                logical_validation = LogicalValidation(False, 0.0, "", "")
            if isinstance(fallacies, Exception):
                self.logger.warning(f"Fallacy detection failed: {fallacies}")
                fallacies = []
            
            # Combine consistency scores
            overall_score = self._combine_consistency_scores(
                logical_structure, contradiction_analysis, logical_validation, fallacies
            )
            
            # Generate reasoning
            reasoning = self._generate_consistency_reasoning(
                logical_structure, contradiction_analysis, logical_validation, fallacies
            )
            
            # Identify uncertainty factors
            uncertainty_factors = self._identify_consistency_uncertainties(
                logical_structure, contradiction_analysis, logical_validation, fallacies
            )
            
            # Generate contextual notes
            contextual_notes = self._generate_consistency_insights(
                claim, logical_structure, contradiction_analysis, logical_validation, fallacies
            )
            
            result = VerificationResult(
                framework_name="consistency",
                confidence_score=overall_score,
                reasoning=reasoning,
                evidence_references=self._collect_consistency_references(logical_structure, logical_validation),
                uncertainty_factors=uncertainty_factors,
                contextual_notes=contextual_notes,
                metadata={
                    "logical_operators_found": len(logical_structure.operators),
                    "premises_identified": len(logical_structure.premises),
                    "conclusions_identified": len(logical_structure.conclusions),
                    "contradictions_found": len(contradiction_analysis.contradictions_found),
                    "logical_fallacies_detected": len(fallacies),
                    "logical_validity": logical_validation.is_valid,
                    "verification_timestamp": datetime.utcnow().isoformat(),
                    "strict_mode": self.strict_mode
                }
            )
            
            # Cache result
            self.cache[cache_key] = result
            
            return result
            
        except Exception as e:
            self.logger.error(f"Consistency verification failed: {e}")
            return VerificationResult(
                framework_name="consistency",
                confidence_score=0.1,
                reasoning=f"Consistency verification failed due to technical error: {str(e)}",
                evidence_references=[],
                uncertainty_factors=["Technical failure", "Unable to perform logical analysis"],
                contextual_notes="This result should be treated with extreme caution."
            )
    
    async def verify_consensus(self, proposal: ConsensusProposal, node_context: NodeContext) -> ConsensusVerificationResult:
        """Verify proposal in consensus mode with node-specific consistency requirements"""
        # Extract claim from proposal
        claim_data = proposal.content.get("claim", {})
        claim = Claim(
            content=claim_data.get("content", ""),
            context=claim_data.get("context", {}),
            metadata=claim_data.get("metadata", {})
        )
        
        # Run individual verification
        individual_result = await self.verify_individual(claim)
        
        # Apply node-specific consistency adjustments
        node_adjusted_score = self._apply_consistency_node_adjustments(
            individual_result.confidence_score, node_context, individual_result
        )
        
        # Assess consistency consensus readiness
        consensus_readiness = self._assess_consistency_consensus_readiness(
            individual_result, node_context
        )
        
        return ConsensusVerificationResult(
            node_id=node_context.node_id,
            framework_name="consistency",
            confidence_score=node_adjusted_score,
            reasoning=individual_result.reasoning,
            evidence_quality=self._assess_consistency_evidence_quality(individual_result),
            consensus_readiness=consensus_readiness,
            suggested_refinements=self._suggest_consistency_refinements(individual_result, node_context),
            metadata={
                **individual_result.metadata,
                "node_consistency_adjustments": True,
                "consensus_mode": True,
                "node_logical_strictness": node_context.philosophical_weights.get("consistency", 1.0)
            }
        )
    
    async def _analyze_logical_structure(self, claim: Claim) -> LogicalStructure:
        """Analyze the logical structure of the claim"""
        text = claim.content
        structure = LogicalStructure()
        
        # Extract conditional statements
        for pattern_name, pattern in self.conditional_patterns.items():
            matches = pattern.findall(text)
            for match in matches:
                if len(match) == 2:
                    condition, consequence = match[0].strip(), match[1].strip()
                    structure.conditional_statements.append((condition, consequence))
                    
                    # Identify the operator type
                    if pattern_name == "if_then":
                        structure.operators.append(LogicalOperator.IF_THEN)
                    elif pattern_name == "implies":
                        structure.operators.append(LogicalOperator.IMPLIES)
                    elif pattern_name == "because":
                        structure.operators.append(LogicalOperator.BECAUSE)
        
        # Extract conclusions
        for pattern in self.conclusion_patterns:
            matches = pattern.findall(text)
            for match in matches:
                structure.conclusions.append(match.strip())
                structure.operators.append(LogicalOperator.THEREFORE)
        
        # Extract premises (sentences that aren't conclusions)
        sentences = re.split(r'[.!?]+', text)
        conclusion_texts = set(structure.conclusions)
        
        for sentence in sentences:
            sentence = sentence.strip()
            if sentence and sentence not in conclusion_texts:
                # Check if this is a premise (not a conclusion marker)
                is_premise = not any(marker in sentence.lower() 
                                   for marker in ["therefore", "thus", "hence", "consequently"])
                if is_premise:
                    structure.premises.append(sentence)
        
        # Detect negations
        negation_pattern = re.compile(r'\b(not|no|never|none|nothing|isn\'t|aren\'t|doesn\'t|don\'t|won\'t|can\'t)\s+(.+?)(?:[.!?]|$)', re.IGNORECASE)
        negations = negation_pattern.findall(text)
        for negation in negations:
            structure.negations.append(negation[1].strip())
            structure.operators.append(LogicalOperator.NOT)
        
        # Detect logical connectives
        if re.search(r'\band\b', text, re.IGNORECASE):
            structure.operators.append(LogicalOperator.AND)
        if re.search(r'\bor\b', text, re.IGNORECASE):
            structure.operators.append(LogicalOperator.OR)
        
        return structure
    
    async def _detect_contradictions(self, claim: Claim) -> ContradictionAnalysis:
        """Detect logical contradictions in the claim"""
        text = claim.content.lower()
        analysis = ContradictionAnalysis()
        
        # Check for direct contradictions (X and not X)
        sentences = re.split(r'[.!?]+', claim.content)
        
        for i, sentence1 in enumerate(sentences):
            for j, sentence2 in enumerate(sentences[i+1:], i+1):
                contradiction_score = self._calculate_contradiction_score(sentence1, sentence2)
                if contradiction_score > 0.7:  # High contradiction threshold
                    analysis.contradictions_found.append((sentence1.strip(), sentence2.strip()))
        
        # Check for opposing terms in close proximity
        for term1, term2 in self.contradiction_indicators["opposing_terms"]:
            if term1 in text and term2 in text:
                # Check if they're in the same context (within 20 words)
                term1_pos = [m.start() for m in re.finditer(r'\b' + re.escape(term1) + r'\b', text)]
                term2_pos = [m.start() for m in re.finditer(r'\b' + re.escape(term2) + r'\b', text)]
                
                for pos1 in term1_pos:
                    for pos2 in term2_pos:
                        if abs(pos1 - pos2) < 100:  # Within ~20 words
                            context = text[max(0, min(pos1, pos2)-50):max(pos1, pos2)+50]
                            analysis.contradictions_found.append((term1, term2))
                            analysis.resolution_suggestions.append(
                                f"Clarify the relationship between '{term1}' and '{term2}' in context: ...{context}..."
                            )
        
        # Calculate overall contradiction score
        if analysis.contradictions_found:
            analysis.contradiction_score = min(len(analysis.contradictions_found) / 3.0, 1.0)
        else:
            analysis.contradiction_score = 0.0
        
        return analysis
    
    def _calculate_contradiction_score(self, sentence1: str, sentence2: str) -> float:
        """Calculate contradiction score between two sentences"""
        s1_lower = sentence1.lower().strip()
        s2_lower = sentence2.lower().strip()
        
        # Remove common words and focus on content words
        s1_words = set(word for word in s1_lower.split() if len(word) > 2)
        s2_words = set(word for word in s2_lower.split() if len(word) > 2)
        
        # Check for negation patterns
        s1_negated = any(neg in s1_lower for neg in self.contradiction_indicators["direct_negation"])
        s2_negated = any(neg in s2_lower for neg in self.contradiction_indicators["direct_negation"])
        
        # High overlap in content words + one negated = potential contradiction
        overlap = len(s1_words.intersection(s2_words))
        total_unique = len(s1_words.union(s2_words))
        
        if total_unique == 0:
            return 0.0
        
        overlap_ratio = overlap / total_unique
        
        # Higher contradiction score if high overlap with different negation states
        if overlap_ratio > 0.5 and s1_negated != s2_negated:
            return 0.8
        elif overlap_ratio > 0.3 and s1_negated != s2_negated:
            return 0.6
        else:
            return 0.0
    
    async def _validate_logical_form(self, claim: Claim) -> LogicalValidation:
        """Validate the logical form of the claim"""
        text = claim.content
        
        # Check for valid argument structure (premises → conclusion)
        has_premises = bool(re.search(r'\b(because|since|given|assuming)\b', text, re.IGNORECASE))
        has_conclusion = bool(re.search(r'\b(therefore|thus|hence|consequently)\b', text, re.IGNORECASE))
        
        # Check for logical connectives
        has_connectives = bool(re.search(r'\b(and|or|if|then|not)\b', text, re.IGNORECASE))
        
        # Check for quantifiers
        has_quantifiers = bool(re.search(r'\b(all|some|every|any|no|none)\b', text, re.IGNORECASE))
        
        # Simple validity scoring
        validity_factors = []
        
        # Argument structure validity
        if has_premises and has_conclusion:
            validity_factors.append(0.8)  # Good argument structure
        elif has_premises or has_conclusion:
            validity_factors.append(0.6)  # Partial structure
        else:
            validity_factors.append(0.4)  # No clear structure
        
        # Logical coherence
        if has_connectives:
            validity_factors.append(0.7)  # Has logical structure
        else:
            validity_factors.append(0.5)  # Simple statements
        
        # Quantifier use (shows logical thinking)
        if has_quantifiers:
            validity_factors.append(0.6)
        else:
            validity_factors.append(0.5)
        
        validity_score = statistics.mean(validity_factors)
        is_valid = validity_score > 0.6
        
        # Generate logical form description
        logical_form = self._describe_logical_form(text, has_premises, has_conclusion, has_connectives, has_quantifiers)
        
        return LogicalValidation(
            is_valid=is_valid,
            validity_score=validity_score,
            logical_form=logical_form,
            validation_method="pattern_based",
            issues_found=[] if is_valid else ["Weak logical structure", "Missing clear argument form"]
        )
    
    def _describe_logical_form(self, text: str, has_premises: bool, has_conclusion: bool,
                             has_connectives: bool, has_quantifiers: bool) -> str:
        """Describe the logical form of the claim"""
        form_elements = []
        
        if has_premises and has_conclusion:
            form_elements.append("Premise-Conclusion argument")
        elif has_premises:
            form_elements.append("Premise-based statement")
        elif has_conclusion:
            form_elements.append("Conclusion-oriented statement")
        else:
            form_elements.append("Simple declarative statement")
        
        if has_connectives:
            form_elements.append("with logical connectives")
        
        if has_quantifiers:
            form_elements.append("using quantified statements")
        
        return " ".join(form_elements)
    
    async def _detect_logical_fallacies(self, claim: Claim) -> List[ConsistencyIssue]:
        """Detect common logical fallacies"""
        text = claim.content.lower()
        fallacies = []
        
        for fallacy_type, patterns in self.fallacy_patterns.items():
            for pattern in patterns:
                if re.search(pattern, text):
                    severity = 0.8 if fallacy_type in ["circular_reasoning", "false_dichotomy"] else 0.6
                    
                    fallacies.append(ConsistencyIssue(
                        issue_type=ConsistencyIssueType(fallacy_type),
                        description=f"Potential {fallacy_type.replace('_', ' ')} detected",
                        severity=severity,
                        location=text[:100] + "..." if len(text) > 100 else text,
                        suggested_fix=self._get_fallacy_fix_suggestion(fallacy_type)
                    ))
        
        # Check for specific logical issues
        # Circular reasoning detection
        sentences = re.split(r'[.!?]+', claim.content)
        for sentence in sentences:
            if self._check_circular_reasoning(sentence):
                fallacies.append(ConsistencyIssue(
                    issue_type=ConsistencyIssueType.CIRCULAR_REASONING,
                    description="Potential circular reasoning detected",
                    severity=0.9,
                    location=sentence.strip(),
                    suggested_fix="Provide independent evidence or premises"
                ))
        
        return fallacies
    
    def _check_circular_reasoning(self, sentence: str) -> bool:
        """Check for circular reasoning patterns"""
        sentence_lower = sentence.lower().strip()
        
        # Simple check: if the conclusion appears to restate the premise
        words = sentence_lower.split()
        if len(words) < 4:
            return False
        
        # Look for patterns like "X because X" or "X proves X"
        for i, word in enumerate(words):
            if word in ["because", "since", "proves", "shows", "demonstrates"]:
                # Check if key terms repeat before and after the connector
                before_words = set(words[:i])
                after_words = set(words[i+1:])
                overlap = before_words.intersection(after_words)
                
                # Filter out common words
                meaningful_overlap = [w for w in overlap if len(w) > 3 and w not in ["this", "that", "they", "them"]]
                
                if len(meaningful_overlap) > 1:
                    return True
        
        return False
    
    def _get_fallacy_fix_suggestion(self, fallacy_type: str) -> str:
        """Get suggestion for fixing a specific fallacy"""
        suggestions = {
            "circular_reasoning": "Provide independent evidence that doesn't rely on the conclusion itself",
            "false_dichotomy": "Consider additional alternatives beyond the two options presented",
            "hasty_generalization": "Qualify statements with 'some', 'many', or 'often' instead of absolute terms",
            "ad_hominem": "Focus on the argument itself rather than the person making it",
            "appeal_to_authority": "Provide the reasoning behind the authority's position"
        }
        return suggestions.get(fallacy_type, "Review the logical structure of this argument")
    
    def _combine_consistency_scores(self, logical_structure: LogicalStructure,
                                  contradiction_analysis: ContradictionAnalysis,
                                  logical_validation: LogicalValidation,
                                  fallacies: List[ConsistencyIssue]) -> float:
        """Combine all consistency scores into overall consistency score"""
        
        scores = []
        weights = []
        
        # Logical structure score
        structure_score = self._score_logical_structure(logical_structure)
        scores.append(structure_score)
        weights.append(0.3)
        
        # Contradiction penalty
        contradiction_penalty = 1.0 - contradiction_analysis.contradiction_score
        scores.append(contradiction_penalty)
        weights.append(0.3)
        
        # Logical validation score
        scores.append(logical_validation.validity_score)
        weights.append(0.25)
        
        # Fallacy penalty
        fallacy_penalty = max(1.0 - (len(fallacies) / 5.0), 0.2)  # Max 5 fallacies for score calculation
        scores.append(fallacy_penalty)
        weights.append(0.15)
        
        # Calculate weighted average
        overall_score = sum(s * w for s, w in zip(scores, weights)) / sum(weights)
        
        # Apply strict mode penalty if enabled
        if self.strict_mode and (contradiction_analysis.contradictions_found or fallacies):
            overall_score *= 0.7  # 30% penalty in strict mode
        
        return overall_score
    
    def _score_logical_structure(self, structure: LogicalStructure) -> float:
        """Score the logical structure quality"""
        score_factors = []
        
        # Premise-conclusion structure
        if structure.premises and structure.conclusions:
            score_factors.append(0.9)  # Excellent structure
        elif structure.premises or structure.conclusions:
            score_factors.append(0.7)  # Partial structure
        else:
            score_factors.append(0.5)  # Simple statements
        
        # Logical operators usage
        operator_score = min(len(structure.operators) / 3.0, 1.0)  # Normalize by expected operators
        score_factors.append(operator_score)
        
        # Conditional statements (sophisticated reasoning)
        conditional_score = min(len(structure.conditional_statements) / 2.0, 1.0)
        score_factors.append(conditional_score)
        
        return statistics.mean(score_factors)
    
    def _generate_consistency_reasoning(self, logical_structure: LogicalStructure,
                                      contradiction_analysis: ContradictionAnalysis,
                                      logical_validation: LogicalValidation,
                                      fallacies: List[ConsistencyIssue]) -> str:
        """Generate human-readable reasoning for consistency verification"""
        reasoning_parts = []
        
        # Logical structure analysis
        if logical_structure.premises and logical_structure.conclusions:
            reasoning_parts.append(f"Well-structured argument with {len(logical_structure.premises)} premises and {len(logical_structure.conclusions)} conclusions")
        elif logical_structure.premises or logical_structure.conclusions:
            reasoning_parts.append("Partial argument structure identified")
        else:
            reasoning_parts.append("Simple declarative statements without clear logical structure")
        
        # Logical operators
        if logical_structure.operators:
            operator_types = list(set([op.value for op in logical_structure.operators]))
            reasoning_parts.append(f"Logical operators detected: {', '.join(operator_types)}")
        
        # Contradiction analysis
        if contradiction_analysis.contradictions_found:
            reasoning_parts.append(f"Found {len(contradiction_analysis.contradictions_found)} potential contradictions")
        else:
            reasoning_parts.append("No logical contradictions detected")
        
        # Logical validation
        if logical_validation.is_valid:
            reasoning_parts.append(f"Logical form validated: {logical_validation.logical_form}")
        else:
            reasoning_parts.append("Weak logical structure detected")
        
        # Fallacies
        if fallacies:
            fallacy_types = [f.issue_type.value.replace('_', ' ') for f in fallacies]
            reasoning_parts.append(f"Logical issues detected: {', '.join(set(fallacy_types))}")
        else:
            reasoning_parts.append("No major logical fallacies identified")
        
        return ". ".join(reasoning_parts) + "."
    
    def _identify_consistency_uncertainties(self, logical_structure: LogicalStructure,
                                          contradiction_analysis: ContradictionAnalysis,
                                          logical_validation: LogicalValidation,
                                          fallacies: List[ConsistencyIssue]) -> List[str]:
        """Identify factors contributing to consistency uncertainty"""
        uncertainties = []
        
        if not logical_structure.premises and not logical_structure.conclusions:
            uncertainties.append("Unclear logical structure")
        
        if contradiction_analysis.contradictions_found:
            uncertainties.append("Potential logical contradictions detected")
        
        if not logical_validation.is_valid:
            uncertainties.append("Invalid or weak logical form")
        
        if fallacies:
            high_severity_fallacies = [f for f in fallacies if f.severity > 0.7]
            if high_severity_fallacies:
                uncertainties.append("High-severity logical fallacies detected")
            else:
                uncertainties.append("Minor logical issues identified")
        
        if not logical_structure.operators:
            uncertainties.append("Limited logical connectives")
        
        if len(logical_structure.conditional_statements) == 0:
            uncertainties.append("No conditional reasoning detected")
        
        return uncertainties
    
    def _generate_consistency_insights(self, claim: Claim,
                                     logical_structure: LogicalStructure,
                                     contradiction_analysis: ContradictionAnalysis,
                                     logical_validation: LogicalValidation,
                                     fallacies: List[ConsistencyIssue]) -> str:
        """Generate insights and recommendations for consistency"""
        insights = []
        
        # Structure insights
        if logical_structure.conditional_statements:
            insights.append(f"Contains {len(logical_structure.conditional_statements)} conditional statements showing sophisticated reasoning")
        
        # Operator insights
        if LogicalOperator.IF_THEN in logical_structure.operators:
            insights.append("Uses conditional logic (if-then reasoning)")
        if LogicalOperator.BECAUSE in logical_structure.operators:
            insights.append("Provides causal reasoning")
        
        # Validation insights
        if logical_validation.validity_score > 0.8:
            insights.append("Strong logical form with clear argument structure")
        elif logical_validation.validity_score < 0.4:
            insights.append("Consider strengthening logical structure with clearer premises and conclusions")
        
        # Contradiction insights
        if contradiction_analysis.contradictions_found:
            insights.append("Review potential contradictions for clarity and resolution")
        
        # Fallacy insights
        if fallacies:
            unique_fallacies = set([f.issue_type.value for f in fallacies])
            if len(unique_fallacies) == 1:
                insights.append(f"Address {list(unique_fallacies)[0].replace('_', ' ')} to strengthen argument")
            else:
                insights.append(f"Multiple logical issues identified requiring attention")
        
        return ". ".join(insights) + "." if insights else "Standard logical consistency analysis completed."
    
    def _collect_consistency_references(self, logical_structure: LogicalStructure,
                                      logical_validation: LogicalValidation) -> List[str]:
        """Collect references for consistency evidence"""
        references = []
        
        # Logical structure references
        if logical_structure.premises:
            references.append(f"Premises identified: {'; '.join(logical_structure.premises[:3])}")
        
        if logical_structure.conclusions:
            references.append(f"Conclusions identified: {'; '.join(logical_structure.conclusions[:3])}")
        
        if logical_structure.conditional_statements:
            for condition, consequence in logical_structure.conditional_statements[:2]:
                references.append(f"Conditional: If {condition} then {consequence}")
        
        # Logical form reference
        references.append(f"Logical form: {logical_validation.logical_form}")
        
        return references
    
    def _apply_consistency_node_adjustments(self, score: float, node_context: NodeContext,
                                          result: VerificationResult) -> float:
        """Apply node-specific consistency adjustments"""
        # Get node's consistency framework weight
        consistency_weight = node_context.philosophical_weights.get("consistency", 1.0)
        
        # Adjust based on node's logical strictness
        adjusted_score = score * consistency_weight
        
        # Apply stricter penalties if node values consistency highly
        if consistency_weight > 0.8:
            fallacies_count = result.metadata.get("logical_fallacies_detected", 0)
            contradictions_count = result.metadata.get("contradictions_found", 0)
            
            if fallacies_count > 0 or contradictions_count > 0:
                adjusted_score *= 0.8  # 20% penalty for high-consistency nodes
        
        return min(adjusted_score, 1.0)
    
    def _assess_consistency_evidence_quality(self, result: VerificationResult) -> float:
        """Assess quality of consistency evidence for consensus"""
        quality_factors = []
        
        # Logical structure quality
        premises = result.metadata.get("premises_identified", 0)
        conclusions = result.metadata.get("conclusions_identified", 0)
        structure_quality = min((premises + conclusions) / 3.0, 1.0)
        quality_factors.append(structure_quality)
        
        # Validity quality
        logical_validity = result.metadata.get("logical_validity", False)
        validity_quality = 1.0 if logical_validity else 0.3
        quality_factors.append(validity_quality)
        
        # Contradiction penalty
        contradictions = result.metadata.get("contradictions_found", 0)
        contradiction_quality = max(1.0 - (contradictions / 3.0), 0.2)
        quality_factors.append(contradiction_quality)
        
        # Fallacy penalty
        fallacies = result.metadata.get("logical_fallacies_detected", 0)
        fallacy_quality = max(1.0 - (fallacies / 5.0), 0.2)
        quality_factors.append(fallacy_quality)
        
        return statistics.mean(quality_factors)
    
    def _assess_consistency_consensus_readiness(self, result: VerificationResult,
                                              node_context: NodeContext) -> bool:
        """Assess if consistency analysis is ready for consensus"""
        # Base readiness on confidence score
        base_readiness = result.confidence_score > 0.6
        
        # Check for critical logical issues
        critical_issues = [
            "Potential logical contradictions detected",
            "Invalid or weak logical form",
            "High-severity logical fallacies detected"
        ]
        
        has_critical_issues = any(issue in result.uncertainty_factors 
                                for issue in critical_issues)
        
        # In strict mode, require higher standards
        if self.strict_mode:
            return result.confidence_score > 0.8 and not has_critical_issues
        else:
            return base_readiness and not has_critical_issues
    
    def _suggest_consistency_refinements(self, result: VerificationResult,
                                       node_context: NodeContext) -> List[str]:
        """Suggest refinements for better consistency verification"""
        suggestions = []
        
        if result.confidence_score < 0.7:
            suggestions.append("Strengthen logical structure with clearer premises and conclusions")
        
        if "Unclear logical structure" in result.uncertainty_factors:
            suggestions.append("Add explicit logical connectives (if-then, because, therefore)")
        
        if "Potential logical contradictions detected" in result.uncertainty_factors:
            suggestions.append("Resolve contradictory statements or clarify apparent inconsistencies")
        
        if "Invalid or weak logical form" in result.uncertainty_factors:
            suggestions.append("Restructure argument with clear premises leading to conclusions")
        
        if "High-severity logical fallacies detected" in result.uncertainty_factors:
            suggestions.append("Address logical fallacies to strengthen argument validity")
        
        if "Limited logical connectives" in result.uncertainty_factors:
            suggestions.append("Use more sophisticated logical reasoning with conditional statements")
        
        return suggestions
    
    def _generate_cache_key(self, claim: Claim) -> str:
        """Generate cache key for consistency analysis"""
        import hashlib
        content_hash = hashlib.md5(claim.content.encode()).hexdigest()
        context_hash = hashlib.md5(str(sorted(claim.context.items())).encode()).hexdigest()
        mode_hash = hashlib.md5(str(self.strict_mode).encode()).hexdigest()
        return f"consistency_{content_hash}_{context_hash}_{mode_hash}"
    
    def clear_cache(self):
        """Clear consistency analysis cache"""
        self.cache.clear()
    
    def get_consistency_stats(self) -> Dict[str, Any]:
        """Get consistency analysis statistics"""
        return {
            "cached_analyses": len(self.cache),
            "strict_mode": self.strict_mode,
            "fallacy_patterns": len(self.fallacy_patterns),
            "contradiction_indicators": len(self.contradiction_indicators),
            "logical_patterns": len(self.conditional_patterns)
        }