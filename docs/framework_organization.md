# Chil Framework Organization

*Organized Component Structure for Unified Philosophical Verification*

The Chil framework has been reorganized into dedicated folders for each verification component, providing clear separation of concerns and modular development.

## 📁 **New Directory Structure**

```
chil/framework/
├── __init__.py                 # Main framework exports
├── consensus_types.py          # Shared type definitions
│
├── empirical/                  # Empirical Verification Framework
│   ├── __init__.py
│   └── empirical_real.py       # Fact-checking APIs, academic sources
│
├── contextual/                 # Contextual Verification Framework  
│   ├── __init__.py
│   └── contextual_real.py      # NLP, cultural analysis, bias detection
│
├── consistency/                # Consistency Verification Framework
│   ├── __init__.py
│   ├── consistency_real.py     # Pattern-based logic checking
│   └── consistency_formal.py   # Formal theorem proving (Lean, Z3, Coq)
│
├── power_dynamics/             # Power Dynamics Verification Framework
│   ├── __init__.py
│   └── power_dynamics_real.py  # Bias detection, perspective diversity
│
├── utility/                    # Utility Verification Framework
│   ├── __init__.py
│   └── utility_real.py         # Cost-benefit, actionability analysis
│
├── evolution/                  # Evolution Verification Framework
│   ├── __init__.py
│   └── evolutionary.py         # Temporal robustness, adaptation
│
└── llm_integration/            # LLM + Formal Verification Integration
    ├── __init__.py
    └── llm_formal_integration.py  # LLM-guided formal verification
```

## 🔧 **Component Specifications**

### **1. Empirical Framework** (`chil.framework.empirical`)

**Purpose**: Objective, evidence-based verification through external data sources

**Key Features**:
- **Fact-checking APIs**: PolitiFact, Snopes, FactCheck.org integration
- **Academic databases**: PubMed, ArXiv, CrossRef search and validation
- **Statistical validation**: Numerical claim verification and confidence intervals
- **Source credibility**: Assessment of evidence quality and reliability
- **Caching system**: 24-hour result caching for performance

**Usage**:
```python
from chil.framework.empirical import RealEmpiricalFramework

empirical = RealEmpiricalFramework(api_keys={"snopes_api_key": "..."})
result = await empirical.verify_individual(claim)
# Returns empirical evidence with confidence scores and source references
```

### **2. Contextual Framework** (`chil.framework.contextual`)

**Purpose**: Situational and cultural context analysis

**Key Features**:
- **NLP processing**: Entity extraction, semantic analysis, coherence scoring
- **Cultural bias detection**: Analysis of cultural indicators and perspectives
- **Stakeholder analysis**: Identification of affected parties and viewpoints
- **Context completeness**: Assessment of provided contextual information
- **Perspective diversity**: Measurement of inclusive representation

**Usage**:
```python
from chil.framework.contextual import RealContextualFramework

contextual = RealContextualFramework(enable_advanced_nlp=True)
result = await contextual.verify_individual(claim)
# Returns contextual analysis with bias indicators and diversity scores
```

### **3. Consistency Framework** (`chil.framework.consistency`)

**Purpose**: Logical consistency through multiple approaches

**Dual Implementation**:
- **Pattern-based** (`RealConsistencyFramework`): Heuristic contradiction detection
- **Formal logic** (`FormalConsistencyFramework`): Mathematical theorem proving

**Key Features**:
- **Theorem provers**: Lean 4, Coq integration for mathematical rigor
- **SMT solvers**: Z3 for decidable logical fragments
- **Logic programming**: SWI-Prolog for constraint satisfaction
- **Contradiction detection**: Automated inconsistency identification
- **Fallacy detection**: Common logical fallacy recognition

**Usage**:
```python
from chil.framework.consistency import FormalConsistencyFramework

formal_consistency = FormalConsistencyFramework(
    preferred_systems=[FormalSystem.LEAN, FormalSystem.Z3]
)
result = await formal_consistency.verify_individual(claim)
# Returns formal proof results with mathematical guarantees
```

### **4. Power Dynamics Framework** (`chil.framework.power_dynamics`)

**Purpose**: Analysis of power structures and institutional biases

**Key Features**:
- **Power structure mapping**: Identification of institutional, economic, political power
- **Bias assessment**: Multi-dimensional bias detection (institutional, cultural, gender, etc.)
- **Perspective diversity**: Analysis of represented vs. missing viewpoints
- **Stakeholder impact**: Assessment of effects on different groups
- **Inclusion scoring**: Measurement of inclusive representation

**Usage**:
```python
from chil.framework.power_dynamics import RealPowerDynamicsFramework

power_dynamics = RealPowerDynamicsFramework()
result = await power_dynamics.verify_individual(claim)
# Returns power analysis with bias indicators and diversity metrics
```

### **5. Utility Framework** (`chil.framework.utility`)

**Purpose**: Practical consequences and actionability evaluation

**Key Features**:
- **Cost-benefit analysis**: Monetary, time, social, environmental factors
- **Actionability assessment**: Clarity, feasibility, success probability
- **Outcome simulation**: Multiple scenario modeling with probabilities
- **Resource analysis**: Requirements for implementation
- **Stakeholder coordination**: Required parties for successful execution

**Usage**:
```python
from chil.framework.utility import RealUtilityFramework

utility = RealUtilityFramework()
result = await utility.verify_individual(claim)
# Returns utility analysis with cost-benefit and actionability scores
```

### **6. Evolution Framework** (`chil.framework.evolution`)

**Purpose**: Temporal robustness and adaptive learning

**Key Features**:
- **Temporal stability**: Analysis of claim robustness over time
- **Learning mechanisms**: Adaptation based on new evidence
- **Historical consistency**: Validation against past knowledge
- **Future robustness**: Prediction of long-term validity
- **Framework optimization**: Adaptive weight adjustment

**Usage**:
```python
from chil.framework.evolution import EvolutionaryVerifier

evolution = EvolutionaryVerifier()
result = await evolution.verify_individual(claim)
# Returns temporal analysis and adaptation recommendations
```

### **7. LLM Integration Framework** (`chil.framework.llm_integration`)

**Purpose**: Synergy between Large Language Models and formal verification

**Key Features**:
- **Natural language translation**: Claims → formal logic (Lean, Z3, Prolog)
- **Multi-step reasoning**: Complex argument breakdown and verification
- **Proof explanation**: Formal results → human-readable explanations
- **Iterative refinement**: LLM-guided improvement of formal translations
- **Cross-system coordination**: Unified verification across multiple formal systems

**Usage**:
```python
from chil.framework.llm_integration import LLMFormalIntegration

llm_formal = LLMFormalIntegration(
    llm_provider="anthropic",
    model_name="claude-3-sonnet"
)
result = await llm_formal.verify_individual(claim)
# Returns LLM+formal verification with explanations and refinements
```

## 🔄 **Unified Framework Integration**

All frameworks can be used together through the main system orchestrator:

```python
from chil import UnifiedIdealVerifier, VerificationMode
from chil.framework import (
    RealEmpiricalFramework,
    RealContextualFramework, 
    FormalConsistencyFramework,
    RealPowerDynamicsFramework,
    RealUtilityFramework,
    LLMFormalIntegration
)

# Create unified verifier with all frameworks
verifier = UnifiedIdealVerifier(
    mode=VerificationMode.HYBRID,
    frameworks={
        "empirical": RealEmpiricalFramework(),
        "contextual": RealContextualFramework(),
        "consistency": FormalConsistencyFramework(),
        "power_dynamics": RealPowerDynamicsFramework(),
        "utility": RealUtilityFramework(),
        "llm_integration": LLMFormalIntegration()
    }
)

# Verify claim across all frameworks
result = await verifier.verify(claim)
# Returns unified verification result combining all framework insights
```

## 🎯 **Benefits of Organization**

### **1. Modularity**
- Each framework can be developed, tested, and deployed independently
- Easy to add new frameworks or replace implementations
- Clear separation of concerns reduces complexity

### **2. Maintainability**  
- Framework-specific bugs isolated to their respective modules
- Easier code review and collaborative development
- Clear ownership and responsibility boundaries

### **3. Extensibility**
- New verification approaches can be added as separate frameworks
- Domain-specific implementations can extend base frameworks
- Plugin architecture supports community contributions

### **4. Testing**
- Each framework can have comprehensive unit tests
- Integration testing focuses on framework interactions
- Easier to mock dependencies for isolated testing

### **5. Performance**
- Selective framework loading based on use case
- Independent caching strategies per framework
- Parallel execution of framework verifications

### **6. Configuration**
- Framework-specific configuration files
- Easy to enable/disable specific frameworks
- Fine-tuned parameters for different domains

## 📦 **Import Patterns**

### **Individual Framework Import**
```python
# Import specific framework
from chil.framework.empirical import RealEmpiricalFramework

# Use directly
empirical = RealEmpiricalFramework(api_keys=config.api_keys)
result = await empirical.verify_individual(claim)
```

### **Multiple Framework Import**
```python
# Import multiple frameworks
from chil.framework import (
    RealEmpiricalFramework,
    FormalConsistencyFramework,
    LLMFormalIntegration
)

# Use in combination
frameworks = {
    "empirical": RealEmpiricalFramework(),
    "consistency": FormalConsistencyFramework(), 
    "llm": LLMFormalIntegration()
}
```

### **Full Framework Import**
```python
# Import everything from framework
from chil.framework import *

# All frameworks available
empirical = RealEmpiricalFramework()
contextual = RealContextualFramework()
# ... etc
```

### **Consensus Mode Import**
```python
# Import for consensus verification
from chil.framework import ConsensusProposal, NodeContext
from chil.framework.empirical import RealEmpiricalFramework

# Use in consensus
framework = RealEmpiricalFramework()
consensus_result = await framework.verify_consensus(proposal, node_context)
```

## 🔮 **Future Framework Additions**

The organized structure makes it easy to add new frameworks:

### **Potential New Frameworks**
- `chil.framework.legal`: Legal precedent and statutory analysis
- `chil.framework.medical`: Evidence-based medicine verification
- `chil.framework.financial`: Economic modeling and risk assessment
- `chil.framework.environmental`: Sustainability and ecological impact
- `chil.framework.ethical`: Moral and ethical consideration analysis
- `chil.framework.historical`: Historical accuracy and context verification

### **Adding a New Framework**
1. Create directory: `chil/framework/new_framework/`
2. Implement: `new_framework_real.py` with `UnifiedVerificationComponent` base
3. Create: `__init__.py` with proper exports
4. Update: `chil/framework/__init__.py` to include new framework
5. Add: Tests in `tests/unit/test_new_framework.py`

This organization provides a solid foundation for the continued evolution and expansion of the Chil verification framework while maintaining clean architecture and ease of use.

---

*For implementation details of each framework, see their respective directories and documentation files.*