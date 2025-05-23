# Chil Consensus Flow Diagrams

This document provides visual representations of how consensus is achieved in the Chil framework through Philosophical Paxos.

## Overview: Philosophical Paxos vs Traditional Paxos

```
Traditional Paxos:
┌─────────┐    ┌─────────┐    ┌─────────┐
│ Node A  │    │ Node B  │    │ Node C  │
│ "Value: │    │ "Value: │    │ "Value: │
│   42"   │    │   42"   │    │   42"   │
└─────────┘    └─────────┘    └─────────┘
     │              │              │
     └──────── Consensus: Value = 42 ──────┘

Philosophical Paxos:
┌─────────┐    ┌─────────┐    ┌─────────┐
│ Node A  │    │ Node B  │    │ Node C  │
│Framework│    │Framework│    │Framework│
│Weights: │    │Weights: │    │Weights: │
│E:0.4,C:0.3│  │E:0.2,C:0.4│  │E:0.3,C:0.3│
│U:0.3    │    │U:0.4    │    │U:0.4    │
└─────────┘    └─────────┘    └─────────┘
     │              │              │
     └─── Consensus: Claim Valid (0.87 confidence) ───┘
```

## Phase-by-Phase Consensus Flow

### Phase 1: Prepare with Philosophical Pre-validation

```
Time →

Proposer (Node A):                    Acceptor (Node B):                 Acceptor (Node C):
│                                     │                                  │
│ 1. Receive claim to validate        │                                  │
│ 2. Run local philosophical         │                                  │
│    framework evaluation:           │                                  │
│    ┌─────────────────────────┐     │                                  │
│    │ Empirical:      0.92    │     │                                  │
│    │ Contextual:     0.88    │     │                                  │
│    │ Consistency:    0.90    │     │                                  │
│    │ Power Dynamics: 0.85    │     │                                  │
│    │ Utility:        0.94    │     │                                  │
│    │ Evolution:      0.91    │     │                                  │
│    │ Overall:        0.90    │     │                                  │
│    └─────────────────────────┘     │                                  │
│                                     │                                  │
│ 3. Create PREPARE message          │                                  │
├─────PREPARE(n=1, claim, scores)────┼──────────────────────────────────┼─────►
│                                     │                                  │
│                                     │ 4. Receive PREPARE              │
│                                     │ 5. Run own philosophical eval:   │
│                                     │ ┌─────────────────────────┐     │
│                                     │ │ Empirical:      0.85    │     │
│                                     │ │ Contextual:     0.82    │     │
│                                     │ │ Consistency:    0.78    │     │
│                                     │ │ Power Dynamics: 0.70    │     │
│                                     │ │ Utility:        0.76    │     │
│                                     │ │ Evolution:      0.80    │     │
│                                     │ │ Overall:        0.78    │     │
│                                     │ └─────────────────────────┘     │
│                                     │                                  │
│                                     │ 6. Generate conditions/feedback  │
│◄────PROMISE(n=1, conditions)───────┼─                                 │
│                                     │                                  │
│                                     │                                  │ 4. Receive PREPARE
│                                     │                                  │ 5. Run philosophical eval
│                                     │                                  │ ┌─────────────────────────┐
│                                     │                                  │ │ Empirical:      0.90    │
│                                     │                                  │ │ Contextual:     0.86    │
│                                     │                                  │ │ Consistency:    0.88    │
│                                     │                                  │ │ Power Dynamics: 0.82    │
│                                     │                                  │ │ Utility:        0.92    │
│                                     │                                  │ │ Evolution:      0.89    │
│                                     │                                  │ │ Overall:        0.88    │
│                                     │                                  │ └─────────────────────────┘
│                                     │                                  │
│◄────────────────────────────────────┼─────PROMISE(n=1, minor_concerns)┼─
│                                     │                                  │
│ 7. Analyze feedback and quorum      │                                  │
│    - Node B: conditional promise    │                                  │
│    - Node C: strong promise         │                                  │
│    - Quorum achieved: 2/2           │                                  │
│                                     │                                  │
```

### Phase 2: Accept with Consensus Refinement

```
Time →

Proposer (Node A):                    Acceptor (Node B):                 Acceptor (Node C):
│                                     │                                  │
│ 8. Refine proposal based on         │                                  │
│    feedback from Phase 1:          │                                  │
│    ┌─────────────────────────┐     │                                  │
│    │ Added: uncertainty ranges│     │                                  │
│    │ Added: power dynamics    │     │                                  │
│    │        transparency      │     │                                  │
│    │ Added: additional sources│     │                                  │
│    └─────────────────────────┘     │                                  │
│                                     │                                  │
│ 9. Re-evaluate refined proposal     │                                  │
│    Overall: 0.90 → 0.93            │                                  │
│                                     │                                  │
├─────ACCEPT(n=1, refined_proposal)──┼──────────────────────────────────┼─────►
│                                     │                                  │
│                                     │ 10. Evaluate refinements        │
│                                     │     Score: 0.78 → 0.86          │
│                                     │     Conditions addressed: ✓      │
│                                     │                                  │
│◄────ACCEPTED(n=1, final_score)─────┼─                                 │
│                                     │                                  │
│                                     │                                  │ 10. Evaluate refinements
│                                     │                                  │     Score: 0.88 → 0.90
│                                     │                                  │     Minor improvements: ✓
│                                     │                                  │
│◄────────────────────────────────────┼─────ACCEPTED(n=1, final_score)─┼─
│                                     │                                  │
│ 11. Consensus achieved:             │                                  │
│     ┌─────────────────────────┐     │                                  │
│     │ Status: ACCEPTED        │     │                                  │
│     │ Confidence: 0.89        │     │                                  │
│     │ Participating: 3/3      │     │                                  │
│     │ Quorum: ✓               │     │                                  │
│     └─────────────────────────┘     │                                  │
│                                     │                                  │
```

## Multi-Round Deliberation Flow

For complex claims requiring extended philosophical debate:

```
Round 1: Initial Positions
┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐
│ Node A  │    │ Node B  │    │ Node C  │    │ Node D  │
│ 0.92    │    │ 0.71    │    │ 0.88    │    │ 0.65    │
│(Strong) │    │(Skeptical)   │(Moderate)    │(Weak)   │
└─────────┘    └─────────┘    └─────────┘    └─────────┘
     │              │              │              │
     └──────────────┼──────────────┼──────────────┘
                    │              │
                    └─ Disagreement threshold exceeded ─┘
                                   │
                                   ▼
                         Start Multi-Round Debate

Round 2: Argument Exchange
┌─────────┐ Arguments ┌─────────┐ Counter-args ┌─────────┐ Evidence ┌─────────┐
│ Node A  │──────────►│ Node B  │─────────────►│ Node C  │─────────►│ Node D  │
│ Provides│           │ Raises  │              │ Offers  │          │ Requests│
│ evidence│           │ concerns│              │ context │          │ clarity │
└─────────┘           └─────────┘              └─────────┘          └─────────┘
     ▲                     │                       ▲                     │
     │                     ▼                       │                     ▼
     │              ┌─────────┐              ┌─────────┐           ┌─────────┐
     │              │New Score│              │New Score│           │New Score│
     │              │  0.75   │              │  0.85   │           │  0.72   │
     │              │ (+0.04) │              │ (-0.03) │           │ (+0.07) │
     │              └─────────┘              └─────────┘           └─────────┘
     │                                                                     │
     └─────────────────────── Refined Arguments ──────────────────────────┘

Round 3: Convergence Analysis
┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐
│ Node A  │    │ Node B  │    │ Node C  │    │ Node D  │
│ 0.90    │    │ 0.82    │    │ 0.86    │    │ 0.79    │
│(-0.02)  │    │(+0.11)  │    │(-0.02)  │    │(+0.14)  │
└─────────┘    └─────────┘    └─────────┘    └─────────┘
     │              │              │              │
     └──────────────┼──────────────┼──────────────┘
                    │              │
        Convergence: All scores within 0.11 range
                    │              │
                    ▼              ▼
              Consensus: 0.84 ± 0.055
              Confidence: High (low variance)
```

## Trust Evolution During Consensus

```
Initial Trust Scores:
Node A: 0.80  Node B: 0.75  Node C: 0.85  Node D: 0.70

After Round 1:
┌─────────┐  +0.05   ┌─────────┐  +0.08   ┌─────────┐  +0.02   ┌─────────┐
│ Node A  │ (good    │ Node B  │ (constructive     │ Node C  │ (solid   │ Node D  │
│  0.85   │ evidence)│  0.83   │ criticism)        │  0.87   │ reasoning)│  0.70   │
└─────────┘          └─────────┘                  └─────────┘          └─────────┘
                                                                       (no change:
                                                                        weak input)

After Round 2:
┌─────────┐  +0.02   ┌─────────┐  +0.03   ┌─────────┐  +0.01   ┌─────────┐
│ Node A  │ (refined │ Node B  │ (adapted │ Node C  │ (balanced│ Node D  │
│  0.87   │ approach)│  0.86   │ position)│  0.88   │ mediation)│  0.72   │
└─────────┘          └─────────┘          └─────────┘          └─────────┘
                                                               +0.02
                                                               (improving)

Final Trust Scores:
Node A: 0.87 (+0.07)  Node B: 0.86 (+0.11)  Node C: 0.88 (+0.03)  Node D: 0.72 (+0.02)

Trust increased most for Node B due to constructive evolution of position
```

## Framework Integration Flow

How different philosophical frameworks contribute to consensus:

```
Claim: "Remote work increases productivity"

┌─────────────────────────────────────────────────────────────────────────────┐
│                          Node A (Empirical Focus)                          │
├─────────────────────────────────────────────────────────────────────────────┤
│ Empirical (0.4):    0.75  │ "Studies show mixed results"                   │
│ Contextual (0.15):  0.80  │ "Depends on job type"                          │
│ Consistency (0.2):  0.85  │ "Logically coherent claim"                     │
│ Power Dynamics (0.1): 0.70 │ "May benefit some workers more"               │
│ Utility (0.1):      0.90  │ "Practical implications clear"                 │
│ Evolution (0.05):   0.85  │ "Adapting to new evidence"                     │
│ ────────────────────────────────────────────────────────────────────────── │
│ Overall Score: 0.79                                                        │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                        Node B (Skeptical/Contextual)                       │
├─────────────────────────────────────────────────────────────────────────────┤
│ Empirical (0.2):    0.70  │ "Limited long-term studies"                    │
│ Contextual (0.3):   0.65  │ "Highly context-dependent"                     │
│ Consistency (0.25): 0.75  │ "Some logical gaps"                            │
│ Power Dynamics (0.15): 0.60 │ "Benefits employers vs employees unclear"    │
│ Utility (0.05):     0.80  │ "Some practical value"                         │
│ Evolution (0.05):   0.70  │ "Too early to conclude"                        │
│ ────────────────────────────────────────────────────────────────────────── │
│ Overall Score: 0.68                                                        │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                         Node C (Pragmatic/Utility)                         │
├─────────────────────────────────────────────────────────────────────────────┤
│ Empirical (0.15):   0.75  │ "Sufficient evidence for action"               │
│ Contextual (0.2):   0.85  │ "Clear practical context"                      │
│ Consistency (0.15): 0.80  │ "Reasonably consistent"                        │
│ Power Dynamics (0.1): 0.75 │ "Some power considerations"                   │
│ Utility (0.35):     0.90  │ "High practical utility"                       │
│ Evolution (0.05):   0.85  │ "Can adapt with experience"                    │
│ ────────────────────────────────────────────────────────────────────────── │
│ Overall Score: 0.83                                                        │
└─────────────────────────────────────────────────────────────────────────────┘

                                    ▼
                            Consensus Process
                                    │
    ┌───────────────────────────────┼───────────────────────────────┐
    │                               │                               │
    ▼                               ▼                               ▼
Framework Aggregation:         Trust Weighting:              Meta-Verification:
                              
Empirical: (0.75×0.87 +       Node A: 0.79 × 0.87 = 0.69   Reflective Equilibrium:
           0.70×0.86 +        Node B: 0.68 × 0.86 = 0.58   - Framework tensions: Low
           0.75×0.88) ÷ 3     Node C: 0.83 × 0.88 = 0.73   - Coherence score: 0.85
         = 0.73               
                              Weighted Average: 0.67         Pareto Analysis:
Contextual: 0.77                                            - No dominated solutions
Consistency: 0.80              Trust-Adjusted: 0.76         - Stable equilibrium
Power Dynamics: 0.68
Utility: 0.87                                               Final Validation: ✓
Evolution: 0.80

                                    ▼
                        Final Consensus: 0.76 ± 0.08
                        Status: ACCEPTED (above 0.70 threshold)
                        Confidence: Moderate (variance acceptable)
```

## Error Handling and Edge Cases

### Byzantine Node Detection

```
Normal Consensus Flow:
Node A: 0.85 ──┐
Node B: 0.82 ──┼── Average: 0.81, Variance: 0.02 (Low) ✓
Node C: 0.78 ──┘

Byzantine Node Detected:
Node A: 0.85 ──┐
Node B: 0.83 ──┼── Outlier Detection: Node D variance > 3σ
Node C: 0.80 ──┘    Potential Byzantine behavior detected
Node D: 0.15 ──┘    
                    ↓
               Isolation Protocol:
               - Flag Node D
               - Reduce trust score
               - Exclude from current consensus
               - Require proof of good faith
```

### Network Partition Handling

```
Before Partition:
┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐
│ Node A  │────│ Node B  │────│ Node C  │────│ Node D  │────│ Node E  │
└─────────┘    └─────────┘    └─────────┘    └─────────┘    └─────────┘

During Partition:
┌─────────┐    ┌─────────┐         ║         ┌─────────┐    ┌─────────┐
│ Node A  │────│ Node B  │         ║         │ Node D  │────│ Node E  │
└─────────┘    └─────────┘         ║         └─────────┘    └─────────┘
                                   ║              
           Partition A (2 nodes)   ║   Partition B (2 nodes)
           Quorum: NO (need 3)     ║   Quorum: NO (need 3)
                                   ║
                        ┌─────────┐║
                        │ Node C  ││ Isolated node
                        └─────────┘║ (awareness of partition)

Resolution Strategy:
1. Detect partition through timeout/heartbeat failure
2. Enter "partition mode" - no new consensus until reunion
3. Continue individual verification mode
4. On partition healing, sync philosophical states
5. Resume consensus with updated trust scores
```

This comprehensive flow documentation shows how Chil's philosophical consensus operates in practice, handling both normal operations and edge cases while maintaining philosophical rigor throughout the process.

<function_calls>
<invoke name="TodoWrite">
<parameter name="todos">[{"id": "doc1", "content": "Create detailed consensus example in examples/", "status": "completed", "priority": "high"}, {"id": "doc2", "content": "Create comprehensive design document", "status": "completed", "priority": "high"}, {"id": "doc3", "content": "Add consensus flow diagrams and explanations", "status": "completed", "priority": "medium"}]