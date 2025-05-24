# BrahminyKite Tools Layer

High-performance, local tool integration for framework-specific verification and training.

## Architecture

```
tools/
├── protos/          # gRPC service definitions
├── services/        # Server implementations
├── clients/         # Client libraries with caching
└── configs/         # Service configurations
```

## Services

### 1. Empirical Tools (Port 50051)
- **Z3 SMT Solver**: Logical consistency checking
- **Lean Theorem Prover**: Mathematical proof verification
- **DuckDB**: High-performance analytical queries
- **Oxigraph**: RDF/SPARQL knowledge graphs

### 2. Contextual Tools (Port 50052)
- **spaCy**: NLP analysis (entities, POS, dependencies)
- **Gensim**: Topic modeling and embeddings
- **FAISS**: Vector similarity search
- **Local sentiment**: TextBlob/VADER integration

### 3. Consistency Tools (Port 50053)
- **SQLite FTS5**: Pattern matching and full-text search
- **Souffle Datalog**: High-performance logical inference

### 4. Power Dynamics Tools (Port 50054)
- **Fairness metrics**: Demographic parity, equalized odds
- **NetworkX**: Graph analysis and community detection
- **scikit-learn**: Clustering for perspective analysis
- **Rust bias detector**: High-performance bias detection

### 5. Utility Tools (Port 50055)
- **XGBoost/LightGBM**: Gradient boosting predictions
- **OR-Tools**: Constraint optimization
- **Game theory**: Equilibrium computation
- **Numba utilities**: JIT-compiled calculations

### 6. Evolution Tools (Port 50056)
- **DEAP**: Genetic algorithms with Numba acceleration
- **JAX**: Evolutionary strategies (CMA-ES, etc.)
- **Ray RLlib**: Local reinforcement learning
- **Rust evolution**: Custom high-performance algorithms

## Usage

### Starting Services

```python
from chil.tools.services import serve_empirical, serve_contextual

# Start individual services
empirical_server = serve_empirical(port=50051)
contextual_server = serve_contextual(port=50052)

# Or use the service manager
from chil.tools import ServiceManager

manager = ServiceManager(config_path='configs/service_config.yaml')
manager.start_all()
```

### Client Usage

```python
from chil.tools.clients import EmpiricalToolsClient

# Basic usage
with EmpiricalToolsClient() as client:
    # Check logical consistency
    result = client.check_logical_consistency(
        formula="(and (> x 0) (< x 10))",
        constraints=["(= (* x x) 25)"]
    )
    
    # Query facts
    facts = client.query_facts(
        "SELECT * FROM facts WHERE confidence > ?",
        parameters={"1": "0.8"}
    )

# Async batch processing
import asyncio

async def batch_verify():
    client = EmpiricalToolsClient()
    formulas = ["(> x 0)", "(< y 10)", "(= (+ x y) 15)"]
    results = await client.batch_check_consistency(formulas)
    return results
```

## Performance Features

### 1. Caching
- LRU cache for expensive computations
- Configurable TTL and size limits
- Shared cache across requests

### 2. Batching
- Automatic request batching
- Configurable batch size and timeout
- Async batch processing

### 3. Connection Pooling
- Persistent gRPC connections
- Keepalive for long-running services
- Automatic reconnection

### 4. Local Optimization
- Zero-copy data transfer
- Memory-mapped files for large datasets
- Process isolation for stability

## Configuration

Edit `configs/service_config.yaml` to customize:
- Port assignments
- Resource limits
- Cache settings
- Model paths
- Performance tuning

## Monitoring

- Metrics exposed on port 9090
- OpenTelemetry tracing support
- Structured logging with context

## Development

### Adding New Tools

1. Update `protos/tools.proto` with service definition
2. Generate Python code: `python -m grpc_tools.protoc ...`
3. Implement service in `services/`
4. Create client in `clients/`
5. Add configuration to `service_config.yaml`

### Testing

```bash
# Unit tests
pytest tests/tools/

# Integration tests
pytest tests/tools/integration/

# Performance benchmarks
python benchmarks/tools_benchmark.py
```