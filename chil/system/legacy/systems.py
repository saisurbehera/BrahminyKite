"""
Adversarial Debate System

Implements multi-agent debates between philosophical frameworks
for enhanced verification through competing perspectives.
"""

import numpy as np
from typing import Dict, List, Any, Optional
from .frameworks import VerificationFramework, Claim, Domain


class DebateSystem:
    """
    Adversarial verification through multi-agent debate
    Implements different philosophical stances as competing agents
    """
    
    def __init__(self):
        self.agents = self._initialize_agents()
        self.debate_history = []
        self.argument_strategies = self._initialize_strategies()
        self.consensus_threshold = 0.7
        self.max_debate_rounds = 3
    
    def _initialize_agents(self) -> Dict[VerificationFramework, Dict[str, Any]]:
        """Initialize debate agents for each philosophical framework"""
        return {
            VerificationFramework.POSITIVIST: {
                "name": "Empirical Verifier",
                "stance": "Demands objective, measurable evidence",
                "strengths": ["empirical_data", "measurement", "scientific_method"],
                "weaknesses": ["subjective_domains", "cultural_context"],
                "argument_style": "data_driven"
            },
            VerificationFramework.INTERPRETIVIST: {
                "name": "Context Interpreter", 
                "stance": "Emphasizes meaning and cultural understanding",
                "strengths": ["cultural_context", "meaning_analysis", "interpretation"],
                "weaknesses": ["objective_measurement", "universal_claims"],
                "argument_style": "contextual"
            },
            VerificationFramework.PRAGMATIST: {
                "name": "Utility Assessor",
                "stance": "Focuses on practical consequences and effectiveness",
                "strengths": ["practical_outcomes", "effectiveness", "real_world_impact"],
                "weaknesses": ["abstract_theory", "immediate_measurement"],
                "argument_style": "outcome_focused"
            },
            VerificationFramework.CORRESPONDENCE: {
                "name": "Reality Matcher",
                "stance": "Truth must correspond to objective reality",
                "strengths": ["objective_facts", "reality_correspondence", "external_validation"],
                "weaknesses": ["subjective_experiences", "constructed_realities"],
                "argument_style": "reality_based"
            },
            VerificationFramework.COHERENCE: {
                "name": "Logic Validator",
                "stance": "Truth depends on internal consistency and logical coherence",
                "strengths": ["logical_consistency", "systematic_coherence", "internal_logic"],
                "weaknesses": ["empirical_grounding", "practical_application"],
                "argument_style": "logic_based"
            },
            VerificationFramework.CONSTRUCTIVIST: {
                "name": "Power Analyst",
                "stance": "Truth is shaped by power structures and social construction",
                "strengths": ["power_analysis", "bias_detection", "social_construction"],
                "weaknesses": ["objective_claims", "universal_standards"],
                "argument_style": "critical_analysis"
            }
        }
    
    def _initialize_strategies(self) -> Dict[str, Dict[str, Any]]:
        """Initialize argument strategies for different types of debates"""
        return {
            "data_driven": {
                "attack_patterns": ["lack_of_evidence", "measurement_issues", "statistical_problems"],
                "defense_patterns": ["empirical_support", "methodological_rigor", "replication"]
            },
            "contextual": {
                "attack_patterns": ["context_ignorance", "cultural_bias", "oversimplification"],
                "defense_patterns": ["rich_context", "cultural_sensitivity", "nuanced_understanding"]
            },
            "outcome_focused": {
                "attack_patterns": ["impractical", "ineffective", "poor_outcomes"],
                "defense_patterns": ["practical_success", "real_world_impact", "measurable_benefits"]
            },
            "reality_based": {
                "attack_patterns": ["unrealistic", "disconnected", "idealistic"],
                "defense_patterns": ["grounded_in_reality", "factual_basis", "objective_validation"]
            },
            "logic_based": {
                "attack_patterns": ["logical_inconsistency", "contradiction", "invalid_reasoning"],
                "defense_patterns": ["logical_coherence", "systematic_consistency", "valid_inference"]
            },
            "critical_analysis": {
                "attack_patterns": ["hidden_bias", "power_imbalance", "institutional_influence"],
                "defense_patterns": ["bias_awareness", "power_transparency", "critical_perspective"]
            }
        }
    
    def conduct_debate(self, claim: Claim, frameworks: List[VerificationFramework], 
                      initial_scores: Optional[Dict[VerificationFramework, float]] = None) -> Dict[str, Any]:
        """
        Conduct adversarial debate between different philosophical frameworks
        """
        if len(frameworks) < 2:
            return {"error": "Need at least 2 frameworks for debate"}
        
        debate_session = {
            "claim": claim.content,
            "domain": claim.domain.value,
            "participants": [f.value for f in frameworks],
            "rounds": [],
            "initial_scores": initial_scores or {},
            "consensus_evolution": []
        }
        
        # Initialize debate state
        current_scores = initial_scores or {f: 0.5 for f in frameworks}
        
        # Conduct debate rounds
        for round_num in range(self.max_debate_rounds):
            round_result = self._conduct_debate_round(claim, frameworks, current_scores, round_num + 1)
            debate_session["rounds"].append(round_result)
            
            # Update scores based on arguments
            current_scores = self._update_scores_from_arguments(current_scores, round_result)
            
            # Check for consensus
            consensus_level = self._calculate_consensus(current_scores)
            debate_session["consensus_evolution"].append(consensus_level)
            
            if consensus_level >= self.consensus_threshold:
                break
        
        # Final results
        final_consensus = self._calculate_consensus(current_scores)
        winner = max(current_scores, key=current_scores.get) if current_scores else frameworks[0]
        
        debate_result = {
            "session": debate_session,
            "final_scores": {f.value: score for f, score in current_scores.items()},
            "final_consensus": final_consensus,
            "winner": winner.value,
            "debate_quality": self._assess_debate_quality(debate_session),
            "argument_summary": self._summarize_arguments(debate_session)
        }
        
        # Record for learning
        self.debate_history.append(debate_result)
        
        return debate_result
    
    def _conduct_debate_round(self, claim: Claim, frameworks: List[VerificationFramework], 
                            current_scores: Dict[VerificationFramework, float], round_num: int) -> Dict[str, Any]:
        """Conduct a single round of debate"""
        round_result = {
            "round_number": round_num,
            "arguments": {},
            "attacks": {},
            "defenses": {},
            "score_changes": {}
        }
        
        # Each framework presents arguments
        for framework in frameworks:
            argument = self._generate_argument(claim, framework, current_scores)
            round_result["arguments"][framework.value] = argument
        
        # Each framework attacks others
        for attacker in frameworks:
            attacks = {}
            for target in frameworks:
                if target != attacker:
                    attack = self._generate_attack(claim, attacker, target, current_scores)
                    attacks[target.value] = attack
            round_result["attacks"][attacker.value] = attacks
        
        # Each framework defends against attacks
        for defender in frameworks:
            defenses = {}
            for attacker in frameworks:
                if attacker != defender:
                    defense = self._generate_defense(claim, defender, attacker, current_scores)
                    defenses[attacker.value] = defense
            round_result["defenses"][defender.value] = defenses
        
        return round_result
    
    def _generate_argument(self, claim: Claim, framework: VerificationFramework, 
                          current_scores: Dict[VerificationFramework, float]) -> Dict[str, Any]:
        """Generate an argument from a specific framework's perspective"""
        agent = self.agents[framework]
        
        # Assess framework's suitability for this claim
        suitability = self._assess_framework_suitability(framework, claim)
        
        # Generate argument based on framework strengths
        argument_strength = self._calculate_argument_strength(framework, claim, current_scores)
        
        # Create argument structure
        argument = {
            "stance": agent["stance"],
            "suitability_score": suitability,
            "argument_strength": argument_strength,
            "key_points": self._generate_key_points(framework, claim),
            "evidence_type": agent["strengths"],
            "confidence": min(suitability * argument_strength, 1.0)
        }
        
        return argument
    
    def _generate_attack(self, claim: Claim, attacker: VerificationFramework, 
                        target: VerificationFramework, current_scores: Dict[VerificationFramework, float]) -> Dict[str, Any]:
        """Generate an attack from one framework against another"""
        attacker_agent = self.agents[attacker]
        target_agent = self.agents[target]
        
        # Find target's weaknesses that align with attacker's strengths
        attack_vectors = self._find_attack_vectors(attacker, target, claim)
        
        # Calculate attack effectiveness
        attack_strength = self._calculate_attack_strength(attacker, target, claim, current_scores)
        
        attack = {
            "attack_type": attack_vectors[0] if attack_vectors else "general_critique",
            "attack_strength": attack_strength,
            "critique_points": self._generate_critique_points(attacker, target, claim),
            "target_weakness": target_agent["weaknesses"][0] if target_agent["weaknesses"] else "general",
            "effectiveness": attack_strength * self._get_domain_attack_multiplier(attacker, target, claim.domain)
        }
        
        return attack
    
    def _generate_defense(self, claim: Claim, defender: VerificationFramework, 
                         attacker: VerificationFramework, current_scores: Dict[VerificationFramework, float]) -> Dict[str, Any]:
        """Generate a defense from one framework against another's attack"""
        defender_agent = self.agents[defender]
        
        # Assess defense capability
        defense_strength = self._calculate_defense_strength(defender, attacker, claim, current_scores)
        
        # Generate defense strategy
        defense = {
            "defense_type": "strength_emphasis",
            "defense_strength": defense_strength,
            "counter_points": self._generate_counter_points(defender, attacker, claim),
            "framework_strengths": defender_agent["strengths"],
            "effectiveness": defense_strength * self._get_domain_defense_multiplier(defender, claim.domain)
        }
        
        return defense
    
    def _assess_framework_suitability(self, framework: VerificationFramework, claim: Claim) -> float:
        """Assess how suitable a framework is for verifying this type of claim"""
        # Domain-based suitability
        domain_suitability = {
            Domain.EMPIRICAL: {
                VerificationFramework.POSITIVIST: 1.0,
                VerificationFramework.CORRESPONDENCE: 0.9,
                VerificationFramework.COHERENCE: 0.7,
                VerificationFramework.PRAGMATIST: 0.6,
                VerificationFramework.INTERPRETIVIST: 0.3,
                VerificationFramework.CONSTRUCTIVIST: 0.2
            },
            Domain.AESTHETIC: {
                VerificationFramework.INTERPRETIVIST: 1.0,
                VerificationFramework.CONSTRUCTIVIST: 0.9,
                VerificationFramework.PRAGMATIST: 0.6,
                VerificationFramework.COHERENCE: 0.5,
                VerificationFramework.CORRESPONDENCE: 0.3,
                VerificationFramework.POSITIVIST: 0.2
            },
            Domain.ETHICAL: {
                VerificationFramework.COHERENCE: 1.0,
                VerificationFramework.PRAGMATIST: 0.8,
                VerificationFramework.CONSTRUCTIVIST: 0.7,
                VerificationFramework.INTERPRETIVIST: 0.6,
                VerificationFramework.CORRESPONDENCE: 0.4,
                VerificationFramework.POSITIVIST: 0.3
            }
        }
        
        # Get default suitability
        default_suitability = {f: 0.5 for f in VerificationFramework}
        
        suitability_map = domain_suitability.get(claim.domain, default_suitability)
        return suitability_map.get(framework, 0.5)
    
    def _calculate_argument_strength(self, framework: VerificationFramework, claim: Claim, 
                                   current_scores: Dict[VerificationFramework, float]) -> float:
        """Calculate the strength of a framework's argument"""
        base_strength = 0.5
        
        # Boost based on framework suitability
        suitability = self._assess_framework_suitability(framework, claim)
        base_strength += suitability * 0.3
        
        # Boost based on current performance
        current_score = current_scores.get(framework, 0.5)
        base_strength += current_score * 0.2
        
        # Random factor for debate dynamics
        base_strength += np.random.uniform(-0.1, 0.1)
        
        return np.clip(base_strength, 0.0, 1.0)
    
    def _find_attack_vectors(self, attacker: VerificationFramework, target: VerificationFramework, 
                           claim: Claim) -> List[str]:
        """Find effective attack vectors based on framework strengths/weaknesses"""
        attacker_agent = self.agents[attacker]
        target_agent = self.agents[target]
        
        # Find overlap between attacker's strengths and target's weaknesses
        attack_vectors = []
        
        for strength in attacker_agent["strengths"]:
            if any(weakness in strength for weakness in target_agent["weaknesses"]):
                attack_vectors.append(f"{strength}_vs_{target_agent['weaknesses'][0]}")
        
        # Add general attack patterns
        attack_style = attacker_agent["argument_style"]
        if attack_style in self.argument_strategies:
            attack_vectors.extend(self.argument_strategies[attack_style]["attack_patterns"])
        
        return attack_vectors or ["general_critique"]
    
    def _calculate_attack_strength(self, attacker: VerificationFramework, target: VerificationFramework, 
                                 claim: Claim, current_scores: Dict[VerificationFramework, float]) -> float:
        """Calculate the strength of an attack"""
        # Base attack strength
        base_strength = 0.4
        
        # Boost if attacker is strong in this domain
        attacker_suitability = self._assess_framework_suitability(attacker, claim)
        base_strength += attacker_suitability * 0.2
        
        # Boost if target is weak in this domain
        target_suitability = self._assess_framework_suitability(target, claim)
        base_strength += (1 - target_suitability) * 0.2
        
        # Performance differential
        attacker_score = current_scores.get(attacker, 0.5)
        target_score = current_scores.get(target, 0.5)
        if attacker_score > target_score:
            base_strength += (attacker_score - target_score) * 0.2
        
        return np.clip(base_strength, 0.0, 1.0)
    
    def _calculate_defense_strength(self, defender: VerificationFramework, attacker: VerificationFramework, 
                                  claim: Claim, current_scores: Dict[VerificationFramework, float]) -> float:
        """Calculate the strength of a defense"""
        # Base defense strength
        base_strength = 0.5
        
        # Boost if defender is suitable for this domain
        defender_suitability = self._assess_framework_suitability(defender, claim)
        base_strength += defender_suitability * 0.3
        
        # Boost based on current performance
        defender_score = current_scores.get(defender, 0.5)
        base_strength += defender_score * 0.2
        
        return np.clip(base_strength, 0.0, 1.0)
    
    def _get_domain_attack_multiplier(self, attacker: VerificationFramework, target: VerificationFramework, 
                                    domain: Domain) -> float:
        """Get domain-specific attack effectiveness multiplier"""
        # Some frameworks are naturally better at attacking others in certain domains
        multipliers = {
            Domain.EMPIRICAL: {
                (VerificationFramework.POSITIVIST, VerificationFramework.INTERPRETIVIST): 1.2,
                (VerificationFramework.CORRESPONDENCE, VerificationFramework.CONSTRUCTIVIST): 1.1
            },
            Domain.AESTHETIC: {
                (VerificationFramework.INTERPRETIVIST, VerificationFramework.POSITIVIST): 1.2,
                (VerificationFramework.CONSTRUCTIVIST, VerificationFramework.CORRESPONDENCE): 1.1
            }
        }
        
        return multipliers.get(domain, {}).get((attacker, target), 1.0)
    
    def _get_domain_defense_multiplier(self, defender: VerificationFramework, domain: Domain) -> float:
        """Get domain-specific defense effectiveness multiplier"""
        # Home field advantage for suitable frameworks
        suitability = self._assess_framework_suitability(defender, domain)
        return 0.8 + (suitability * 0.4)  # Range: 0.8 to 1.2
    
    def _generate_key_points(self, framework: VerificationFramework, claim: Claim) -> List[str]:
        """Generate key points for a framework's argument"""
        agent = self.agents[framework]
        
        # Generate points based on framework strengths
        points = []
        for strength in agent["strengths"][:3]:  # Top 3 strengths
            points.append(f"Emphasizes {strength} in verification")
        
        # Add domain-specific points
        if claim.domain == Domain.EMPIRICAL and framework == VerificationFramework.POSITIVIST:
            points.append("Demands measurable, objective evidence")
        elif claim.domain == Domain.AESTHETIC and framework == VerificationFramework.INTERPRETIVIST:
            points.append("Requires cultural and contextual understanding")
        
        return points
    
    def _generate_critique_points(self, attacker: VerificationFramework, target: VerificationFramework, 
                                claim: Claim) -> List[str]:
        """Generate critique points for an attack"""
        target_agent = self.agents[target]
        
        # Generate critiques based on target weaknesses
        critiques = []
        for weakness in target_agent["weaknesses"][:2]:  # Top 2 weaknesses
            critiques.append(f"Fails to adequately address {weakness}")
        
        return critiques
    
    def _generate_counter_points(self, defender: VerificationFramework, attacker: VerificationFramework, 
                               claim: Claim) -> List[str]:
        """Generate counter-points for a defense"""
        defender_agent = self.agents[defender]
        
        # Generate defenses based on defender strengths
        counter_points = []
        for strength in defender_agent["strengths"][:2]:  # Top 2 strengths
            counter_points.append(f"Provides robust {strength} analysis")
        
        return counter_points
    
    def _update_scores_from_arguments(self, current_scores: Dict[VerificationFramework, float], 
                                    round_result: Dict[str, Any]) -> Dict[VerificationFramework, float]:
        """Update framework scores based on debate round results"""
        new_scores = current_scores.copy()
        
        # Process arguments, attacks, and defenses
        for framework_name, argument in round_result["arguments"].items():
            framework = VerificationFramework(framework_name)
            # Boost score based on argument strength
            boost = argument["argument_strength"] * 0.1
            new_scores[framework] = min(1.0, new_scores[framework] + boost)
        
        # Process attacks (reduce target scores)
        for attacker_name, attacks in round_result["attacks"].items():
            for target_name, attack in attacks.items():
                target_framework = VerificationFramework(target_name)
                # Reduce score based on attack effectiveness
                reduction = attack["effectiveness"] * 0.05
                new_scores[target_framework] = max(0.0, new_scores[target_framework] - reduction)
        
        # Process defenses (restore some score)
        for defender_name, defenses in round_result["defenses"].items():
            defender_framework = VerificationFramework(defender_name)
            # Calculate average defense effectiveness
            if defenses:
                avg_defense = np.mean([d["effectiveness"] for d in defenses.values()])
                restoration = avg_defense * 0.03
                new_scores[defender_framework] = min(1.0, new_scores[defender_framework] + restoration)
        
        return new_scores
    
    def _calculate_consensus(self, scores: Dict[VerificationFramework, float]) -> float:
        """Calculate level of consensus between frameworks"""
        if len(scores) <= 1:
            return 1.0
        
        score_values = list(scores.values())
        # High consensus = low standard deviation
        std_dev = np.std(score_values)
        max_possible_std = 0.5  # Reasonable max for 0-1 range
        
        consensus = 1.0 - min(std_dev / max_possible_std, 1.0)
        return consensus
    
    def _assess_debate_quality(self, debate_session: Dict[str, Any]) -> Dict[str, float]:
        """Assess the quality of the debate"""
        return {
            "argument_diversity": len(set(debate_session["participants"])) / len(VerificationFramework),
            "rounds_completed": len(debate_session["rounds"]),
            "consensus_improvement": (debate_session["consensus_evolution"][-1] - 
                                    debate_session["consensus_evolution"][0] 
                                    if len(debate_session["consensus_evolution"]) > 1 else 0),
            "engagement_level": np.random.uniform(0.6, 0.9)  # Placeholder
        }
    
    def _summarize_arguments(self, debate_session: Dict[str, Any]) -> Dict[str, Any]:
        """Summarize the main arguments from the debate"""
        summary = {
            "main_arguments": {},
            "strongest_attacks": {},
            "best_defenses": {},
            "consensus_points": []
        }
        
        # Extract main arguments from each framework
        for round_data in debate_session["rounds"]:
            for framework, argument in round_data["arguments"].items():
                if framework not in summary["main_arguments"]:
                    summary["main_arguments"][framework] = []
                summary["main_arguments"][framework].extend(argument["key_points"])
        
        return summary
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get debate system statistics"""
        if not self.debate_history:
            return {"message": "No debate history available"}
        
        return {
            "total_debates": len(self.debate_history),
            "average_consensus": np.mean([d["final_consensus"] for d in self.debate_history]),
            "most_common_winner": self._get_most_common_winner(),
            "average_rounds": np.mean([len(d["session"]["rounds"]) for d in self.debate_history]),
            "framework_win_rates": self._calculate_win_rates()
        }
    
    def _get_most_common_winner(self) -> str:
        """Get the most commonly winning framework"""
        if not self.debate_history:
            return "No data"
        
        winners = [debate["winner"] for debate in self.debate_history]
        from collections import Counter
        most_common = Counter(winners).most_common(1)
        return most_common[0][0] if most_common else "No data"
    
    def _calculate_win_rates(self) -> Dict[str, float]:
        """Calculate win rates for each framework"""
        if not self.debate_history:
            return {}
        
        winners = [debate["winner"] for debate in self.debate_history]
        total_debates = len(winners)
        
        win_rates = {}
        for framework in VerificationFramework:
            wins = winners.count(framework.value)
            win_rates[framework.value] = wins / total_debates
        
        return win_rates