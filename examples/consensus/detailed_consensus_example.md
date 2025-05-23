# Detailed Consensus Example: Climate Change Claim Validation

This example demonstrates how Chil's philosophical consensus protocol works in practice, using a climate change claim validation scenario across multiple nodes.

## Scenario Setup

**Claim to Validate**: "Global temperature has increased by 1.1°C since pre-industrial times, primarily due to human activities"

**Network Configuration**:
- 5 nodes representing different research institutions
- Each node has different philosophical framework weightings
- Quorum requirement: 3 out of 5 nodes

**Participating Nodes**:
1. **Node A** (IPCC-style): High empirical weight, moderate contextual
2. **Node B** (Skeptical): High consistency requirements, power dynamics aware  
3. **Node C** (Pragmatic**: High utility focus, evolution-oriented
4. **Node D** (Interpretivist): High contextual weight, narrative-focused
5. **Node E** (Balanced): Equal weights across all frameworks

## Phase 1: Proposal Preparation

### Initial Proposal Creation
```python
# Node A (Proposer) creates the consensus proposal
proposal = ConsensusProposal(
    proposal_id="climate-consensus-001",
    proposal_type=ProposalType.CLAIM_VALIDATION,
    content={
        "claim": {
            "content": "Global temperature has increased by 1.1°C since pre-industrial times, primarily due to human activities",
            "context": {
                "domain": "climate_science",
                "urgency": "high",
                "timeframe": "1850-2023",
                "confidence_level": "very_high"
            },
            "metadata": {
                "sources": ["IPCC AR6", "NASA GISS", "HadCRUT5"],
                "measurement_methods": ["surface_stations", "satellites", "proxies"],
                "peer_reviewed": True
            }
        },
        "validation_criteria": [
            "empirical_evidence_strength",
            "contextual_relevance", 
            "logical_consistency",
            "power_dynamics_awareness",
            "practical_utility",
            "evolutionary_robustness"
        ]
    },
    metadata={
        "priority": "high",
        "deadline": "2024-01-15T00:00:00Z",
        "required_confidence": 0.85
    }
)
```

### Node Context Setup
Each node maintains its own verification context:

```python
# Node A Context (IPCC-style)
node_a_context = NodeContext(
    node_id="ipcc-node-a",
    role=PaxosRole.PROPOSER,
    philosophical_weights={
        "empirical": 0.35,
        "contextual": 0.20, 
        "consistency": 0.15,
        "power_dynamics": 0.10,
        "utility": 0.10,
        "evolution": 0.10
    },
    current_state=NodeState.ACTIVE,
    trust_scores={
        "ipcc-node-a": 1.0,
        "skeptical-node-b": 0.7,
        "pragmatic-node-c": 0.8,
        "interpretivist-node-d": 0.6,
        "balanced-node-e": 0.9
    }
)

# Node B Context (Skeptical)
node_b_context = NodeContext(
    node_id="skeptical-node-b", 
    role=PaxosRole.ACCEPTOR,
    philosophical_weights={
        "empirical": 0.25,
        "contextual": 0.15,
        "consistency": 0.30,  # High consistency requirements
        "power_dynamics": 0.20, # Aware of political influences
        "utility": 0.05,
        "evolution": 0.05
    },
    current_state=NodeState.ACTIVE,
    trust_scores={
        "ipcc-node-a": 0.7,
        "skeptical-node-b": 1.0,
        "pragmatic-node-c": 0.6,
        "interpretivist-node-d": 0.4,
        "balanced-node-e": 0.8
    }
)
```

## Phase 2: Prepare Phase (Modified Paxos Phase 1)

### Prepare Message Broadcasting

Node A (proposer) broadcasts prepare messages to all acceptors:

```python
prepare_msg = PaxosMessage(
    message_type=MessageType.PREPARE,
    proposal_number=ProposalNumber(round=1, node_id="ipcc-node-a"),
    proposal=proposal,
    sender_id="ipcc-node-a",
    philosophical_validation={
        "pre_validation_score": 0.92,
        "framework_scores": {
            "empirical": 0.95,  # Strong temperature data
            "contextual": 0.90, # Well-contextualized
            "consistency": 0.88, # Logically sound
            "power_dynamics": 0.85, # Some political considerations
            "utility": 0.95,     # High practical importance
            "evolution": 0.92    # Robust over time
        },
        "confidence_interval": (0.87, 0.97),
        "meta_verification": {
            "reflective_equilibrium": 0.91,
            "cross_framework_consistency": 0.89
        }
    }
)
```

### Individual Node Processing

Each acceptor node processes the prepare message through their philosophical frameworks:

#### Node B (Skeptical) Response:
```python
# Node B performs rigorous consistency checking
node_b_validation = {
    "empirical": {
        "score": 0.82,
        "reasoning": "Temperature data is strong, but some uncertainty in historical baselines",
        "evidence_quality": "high",
        "methodology_concerns": ["urban_heat_island", "station_coverage"]
    },
    "contextual": {
        "score": 0.75, 
        "reasoning": "Good scientific context, but lacks economic/political context",
        "missing_context": ["economic_implications", "regional_variations"]
    },
    "consistency": {
        "score": 0.78,  # Lower due to skeptical stance
        "reasoning": "Generally consistent but some logical gaps in attribution",
        "consistency_issues": ["natural_variability_attribution", "model_uncertainties"]
    },
    "power_dynamics": {
        "score": 0.70,  # Major concern for skeptical node
        "reasoning": "High potential for political/economic bias in climate research",
        "power_concerns": ["funding_bias", "institutional_pressure", "political_agenda"]
    },
    "utility": {
        "score": 0.60,
        "reasoning": "High claimed utility but uncertain cost-benefit analysis"
    },
    "evolution": {
        "score": 0.65,
        "reasoning": "Scientific consensus has evolved, but may be premature"
    }
}

# Node B's promise response
promise_msg_b = PaxosMessage(
    message_type=MessageType.PROMISE,
    proposal_number=ProposalNumber(round=1, node_id="ipcc-node-a"),
    sender_id="skeptical-node-b",
    philosophical_validation={
        "overall_score": 0.72,  # Below confidence threshold
        "framework_scores": node_b_validation,
        "decision": "conditional_promise",
        "conditions": [
            "require_additional_empirical_evidence",
            "address_power_dynamics_concerns", 
            "provide_uncertainty_quantification"
        ],
        "trust_adjustment": -0.1  # Slight decrease in trust for proposer
    }
)
```

#### Node C (Pragmatic) Response:
```python
node_c_validation = {
    "empirical": {
        "score": 0.88,
        "reasoning": "Sufficient empirical evidence for practical decision-making"
    },
    "utility": {
        "score": 0.95,  # Highest weight for pragmatic node
        "reasoning": "Critical for policy decisions and risk management",
        "practical_implications": ["adaptation_planning", "mitigation_strategies", "economic_planning"]
    },
    "evolution": {
        "score": 0.90,
        "reasoning": "Consensus has strengthened over time with accumulating evidence",
        "evolutionary_trend": "strengthening"
    },
    "contextual": {
        "score": 0.85,
        "reasoning": "Well-situated within broader climate science context"
    },
    "consistency": {
        "score": 0.83,
        "reasoning": "Consistent with broader scientific understanding"
    },
    "power_dynamics": {
        "score": 0.80,
        "reasoning": "Some political dimensions but scientific integrity maintained"
    }
}

promise_msg_c = PaxosMessage(
    message_type=MessageType.PROMISE,
    proposal_number=ProposalNumber(round=1, node_id="ipcc-node-a"),
    sender_id="pragmatic-node-c", 
    philosophical_validation={
        "overall_score": 0.87,  # Above threshold
        "framework_scores": node_c_validation,
        "decision": "strong_promise",
        "practical_endorsement": "recommend_immediate_action"
    }
)
```

## Phase 3: Accept Phase (Modified Paxos Phase 2)

### Quorum Analysis and Proposal Refinement

After receiving promises from nodes B, C, D, E (Node D and E responses similar detail), Node A analyzes the feedback:

```python
# Quorum achieved: 4/5 nodes responded with promises
# But concerns raised need addressing

refined_proposal = ConsensusProposal(
    proposal_id="climate-consensus-001-v2",
    proposal_type=ProposalType.CLAIM_VALIDATION,
    content={
        **original_proposal.content,
        "refinements": {
            "uncertainty_quantification": {
                "temperature_range": "1.0°C to 1.2°C",
                "confidence_interval": "95%",
                "attribution_confidence": "very_likely (>90%)"
            },
            "power_dynamics_disclosure": {
                "funding_sources": ["government", "international_organizations"],
                "institutional_independence": "high",
                "peer_review_process": "rigorous"
            },
            "additional_evidence": {
                "multiple_independent_datasets": True,
                "cross_validation_methods": ["instrumental", "paleoclimate", "physical_models"],
                "emerging_evidence": "attribution_studies_strengthening"
            }
        }
    }
)
```

### Accept Message Broadcasting

```python
accept_msg = PaxosMessage(
    message_type=MessageType.ACCEPT,
    proposal_number=ProposalNumber(round=1, node_id="ipcc-node-a"),
    proposal=refined_proposal,
    sender_id="ipcc-node-a",
    philosophical_validation={
        "refinement_score": 0.94,
        "addressed_concerns": [
            "uncertainty_quantification_added",
            "power_dynamics_transparency_improved",
            "additional_evidence_provided"
        ]
    }
)
```

### Final Acceptor Responses

#### Node B (Skeptical) - Final Accept:
```python
accepted_msg_b = PaxosMessage(
    message_type=MessageType.ACCEPTED,
    proposal_number=ProposalNumber(round=1, node_id="ipcc-node-a"),
    sender_id="skeptical-node-b",
    philosophical_validation={
        "final_score": 0.86,  # Now above threshold
        "score_improvement": +0.14,
        "reasoning": "Refinements adequately address major concerns",
        "remaining_reservations": ["model_uncertainties"],
        "conditional_acceptance": True
    }
)
```

## Phase 4: Consensus Achievement

### Final Consensus Result

```python
consensus_result = {
    "proposal_id": "climate-consensus-001-v2",
    "consensus_achieved": True,
    "final_score": 0.89,
    "participating_nodes": 5,
    "accepting_nodes": 5,
    "quorum_met": True,
    
    "philosophical_consensus": {
        "empirical": {
            "average_score": 0.88,
            "node_agreement": 0.92,
            "consensus_strength": "strong"
        },
        "contextual": {
            "average_score": 0.83,
            "node_agreement": 0.85,
            "consensus_strength": "moderate"
        },
        "consistency": {
            "average_score": 0.82,
            "node_agreement": 0.78,
            "consensus_strength": "moderate"
        },
        "power_dynamics": {
            "average_score": 0.78,
            "node_agreement": 0.71,
            "consensus_strength": "weak"
        },
        "utility": {
            "average_score": 0.91,
            "node_agreement": 0.89,
            "consensus_strength": "strong"
        },
        "evolution": {
            "average_score": 0.85,
            "node_agreement": 0.82,
            "consensus_strength": "moderate"
        }
    },
    
    "meta_verification": {
        "reflective_equilibrium": 0.87,
        "cross_framework_coherence": 0.84,
        "temporal_stability": 0.91,
        "robustness_analysis": "passed"
    },
    
    "trust_updates": {
        "ipcc-node-a": 0.95,      # Increased for good refinements
        "skeptical-node-b": 0.85,  # Increased for constructive criticism
        "pragmatic-node-c": 0.90,  # Maintained
        "interpretivist-node-d": 0.88, # Slight increase
        "balanced-node-e": 0.92    # Maintained
    },
    
    "actionable_outcomes": {
        "validation_status": "ACCEPTED_WITH_CONDITIONS",
        "confidence_level": 0.89,
        "recommended_actions": [
            "proceed_with_policy_implementation",
            "continue_monitoring_evidence",
            "address_remaining_uncertainties"
        ],
        "next_review_date": "2024-06-01T00:00:00Z"
    }
}
```

## Key Insights from This Consensus Process

### 1. **Multi-Framework Integration**
- Each node applied all six philosophical frameworks but with different weightings
- Consensus required satisfying multiple philosophical perspectives simultaneously
- The process revealed strengths and weaknesses across different frameworks

### 2. **Iterative Refinement**
- Initial proposal received conditional acceptance
- Feedback from skeptical nodes led to valuable refinements
- Final consensus was stronger due to addressing concerns

### 3. **Trust Evolution**
- Node trust scores updated based on constructive participation
- Skeptical but constructive criticism increased trust
- Good faith refinements by proposer increased their reputation

### 4. **Philosophical Transparency**
- Each validation included detailed reasoning
- Framework scores provided granular insight into decision basis
- Meta-verification ensured cross-framework coherence

### 5. **Practical Outcomes**
- Consensus resulted in actionable recommendations
- Conditions and reservations preserved intellectual honesty
- Future review dates ensured evolutionary adaptation

This example demonstrates how Chil's philosophical consensus protocol enables rigorous, multi-perspective validation while maintaining the flexibility to refine and improve proposals through collaborative deliberation.