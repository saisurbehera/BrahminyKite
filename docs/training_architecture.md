# BrahminyKite Training Architecture

## Vision
Transform Chil into a library for training LLMs using GRPO (Group Relative Policy Optimization) with specialized mini-LLMs and backing tools for each philosophical framework.

## Framework Component → Tool Mapping (High-Performance, All-Local)

### 1. **Empirical Framework**
- **Mini-LLM**: Fact-checking model (e.g., fine-tuned T5-small)
- **Backing Tools**:
  - **Z3 SMT Solver** (already integrated) - formal verification
  - **Lean Theorem Prover** (already integrated) - mathematical proofs
  - **DuckDB** - local analytical database for fact storage
  - **Local knowledge graphs** - RDF/SPARQL with Oxigraph
- **Training Signal**: Factual accuracy rewards

### 2. **Contextual Framework**
- **Mini-LLM**: Cultural/context-aware model (e.g., mBERT-mini)
- **Backing Tools**:
  - **spaCy** (local) - linguistic analysis
  - **Gensim** - topic modeling and word embeddings
  - **FAISS** - vector similarity search
  - **Local sentiment models** - TextBlob or custom models
- **Training Signal**: Context appropriateness scores

### 3. **Consistency Framework**
- **Mini-LLM**: Logic-specialized model (e.g., RoBERTa-small)
- **Backing Tools**:
  - **Z3** (already integrated) - logical consistency
  - **Prolog** (already integrated) - rule-based reasoning
  - **SQLite with FTS5** - fast local graph patterns
  - **Datalog** (Souffle) - high-performance inference
- **Training Signal**: Logical coherence metrics

### 4. **Power Dynamics Framework**
- **Mini-LLM**: Bias detection model (e.g., DeBERTa-small)
- **Backing Tools**:
  - **Local fairness metrics** - custom NumPy/PyTorch implementations
  - **NetworkX** - in-memory graph analysis
  - **scikit-learn** - clustering for perspective analysis
  - **Rust-based analyzers** - for performance-critical bias detection
- **Training Signal**: Fairness and representation metrics

### 5. **Utility Framework**
- **Mini-LLM**: Outcome prediction model (e.g., ALBERT-tiny)
- **Backing Tools**:
  - **XGBoost/LightGBM** - fast gradient boosting
  - **OR-Tools** (C++ backend) - optimization
  - **Game theory** - custom Rust/C++ implementations
  - **NumPy/Numba** - JIT-compiled calculations
- **Training Signal**: Utility maximization scores

### 6. **Evolution Framework**
- **Mini-LLM**: Temporal adaptation model (e.g., Longformer-small)
- **Backing Tools**:
  - **DEAP** with Numba acceleration - genetic algorithms
  - **JAX** - high-performance evolutionary strategies
  - **Ray RLlib** - distributed RL (local mode)
  - **Rust implementations** - custom evolutionary algorithms
- **Training Signal**: Adaptation effectiveness over time

## GRPO Integration Architecture

### Core Components

```python
# Proposed structure
chil/
├── training/
│   ├── grpo/
│   │   ├── optimizer.py       # GRPO implementation (PyTorch JIT)
│   │   ├── reward_model.py    # Multi-framework reward aggregation
│   │   └── policy.py          # Policy network management
│   ├── models/
│   │   ├── empirical_llm.py   # Mini-LLM for empirical
│   │   ├── contextual_llm.py  # Mini-LLM for contextual
│   │   └── ...                # Other domain-specific LLMs
│   ├── pipelines/
│   │   ├── data_loader.py     # High-performance data loading
│   │   ├── trainer.py         # Main training loop (multiprocessing)
│   │   └── evaluator.py       # Performance metrics
│   └── accelerators/
│       ├── rust_tools/        # Rust implementations
│       ├── cuda_kernels/      # Custom CUDA kernels
│       └── cache.py           # LRU cache for tool results
```

### Performance Optimizations

1. **Tool Caching**
   - LRU cache for expensive computations
   - Memoization of tool outputs
   - Persistent cache with LMDB

2. **Parallel Processing**
   - Framework-level parallelism (multiprocessing)
   - Tool-level async I/O (asyncio)
   - Batch processing for mini-LLMs

3. **Memory Management**
   - Memory-mapped datasets
   - Gradient checkpointing
   - Tool result pooling

4. **Compilation**
   - PyTorch JIT for hot paths
   - Numba for numerical computations
   - Cython for critical loops

### Training Pipeline

1. **Input Processing**
   - Zero-copy data loading
   - Parallel tokenization
   - Cached embeddings

2. **Reward Computation**
   - Vectorized framework rewards
   - SIMD operations where possible
   - Batched tool calls

3. **GRPO Update**
   - Fused optimizer operations
   - Mixed precision training
   - Gradient accumulation

4. **Feedback Loop**
   - Asynchronous checkpointing
   - Incremental metric updates
   - Ring buffer for history

## Implementation Priorities

### Phase 1: Foundation (Weeks 1-2)
- [ ] Set up high-performance data pipeline
- [ ] Implement GRPO with PyTorch optimizations
- [ ] Create tool caching layer
- [ ] Integrate first mini-LLM with local tools

### Phase 2: Multi-Framework (Weeks 3-4)
- [ ] Add remaining mini-LLMs
- [ ] Implement parallel tool execution
- [ ] Build batched reward computation
- [ ] Create performance benchmarks

### Phase 3: Optimization (Weeks 5-6)
- [ ] Profile and optimize bottlenecks
- [ ] Add Rust/C++ accelerators
- [ ] Implement distributed training (local cluster)
- [ ] Fine-tune memory usage

## Key Design Decisions Needed

1. **Mini-LLM Selection**
   - Quantized models (INT8/INT4)
   - Distilled architectures
   - ONNX Runtime optimization

2. **Tool Integration Patterns**
   - Process pools for isolation
   - Shared memory for large data
   - Lock-free queues for communication

3. **Storage Strategy**
   - LMDB for embeddings
   - Arrow/Parquet for datasets
   - Memory-mapped checkpoints

4. **Compute Optimization**
   - BLAS library selection
   - Thread pool configuration
   - NUMA awareness

## Performance Targets

- **Throughput**: 1000+ claims/second during training
- **Latency**: <100ms per framework evaluation
- **Memory**: <32GB for full pipeline
- **Scaling**: Linear up to 8 GPUs

## Next Steps

1. Benchmark local tool alternatives
2. Design cache-friendly data structures
3. Implement core GRPO with optimizations
4. Profile baseline performance
5. Create Rust accelerators for bottlenecks