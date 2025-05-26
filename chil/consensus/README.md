# Consensus Network Layer

This module implements the distributed consensus network layer for BrahminyKite, enabling framework services to coordinate and reach consensus on verification decisions.

## Architecture

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   Node A        │     │   Node B        │     │   Node C        │
│ ┌─────────────┐ │     │ ┌─────────────┐ │     │ ┌─────────────┐ │
│ │  Services   │ │     │ │  Services   │ │     │ │  Services   │ │
│ └──────┬──────┘ │     │ └──────┬──────┘ │     │ └──────┬──────┘ │
│        │        │     │        │        │     │        │        │
│ ┌──────▼──────┐ │     │ ┌──────▼──────┐ │     │ ┌──────▼──────┐ │
│ │  Consensus  │ │     │ │  Consensus  │ │     │ │  Consensus  │ │
│ │    Layer    │ │◄────┼─┤    Layer    │ │◄────┼─┤    Layer    │ │
│ └─────────────┘ │     │ └─────────────┘ │     │ └─────────────┘ │
│                 │     │                 │     │                 │
│   gRPC Transport│     │   gRPC Transport│     │   gRPC Transport│
└─────────────────┘     └─────────────────┘     └─────────────────┘
         ▲                       ▲                       ▲
         │                       │                       │
         └───────────────────────┴───────────────────────┘
                      Peer-to-Peer Network
```

## Components

### 1. **Network Transport** (`network/`)
- gRPC-based peer-to-peer communication
- TLS/mTLS for secure connections
- Connection pooling and management
- Message routing and delivery guarantees

### 2. **Peer Discovery** (`discovery/`)
- Static peer configuration
- Dynamic peer discovery (DNS, Kubernetes)
- Membership management
- Health checking and liveness detection

### 3. **Consensus Protocol** (`protocol/`)
- Integration with existing Paxos implementation
- Message types: Prepare, Promise, Accept, Accepted
- State machine replication
- Leader election

### 4. **State Management** (`state/`)
- Distributed state storage
- State synchronization
- Snapshot and recovery
- Conflict resolution

### 5. **Fault Tolerance** (`fault/`)
- Failure detection
- Network partition handling
- Byzantine fault tolerance (optional)
- Recovery mechanisms

## Usage

### Starting a Consensus Node

```python
from chil.consensus import ConsensusNode

# Create and configure node
node = ConsensusNode(
    node_id="node-1",
    bind_address="0.0.0.0:7000",
    peers=[
        "node-2.cluster.local:7000",
        "node-3.cluster.local:7000"
    ]
)

# Register state machine
node.register_state_machine(my_state_machine)

# Start the node
await node.start()
```

### Proposing Values

```python
# Propose a value for consensus
result = await node.propose(
    key="verification_decision",
    value={
        "claim_id": "claim-123",
        "framework_results": {...},
        "consensus_needed": True
    }
)

if result.success:
    print(f"Consensus reached: {result.value}")
```

## Configuration

### Environment Variables

- `CONSENSUS_NODE_ID`: Unique identifier for this node
- `CONSENSUS_BIND_ADDRESS`: Address to bind the gRPC server
- `CONSENSUS_PEERS`: Comma-separated list of peer addresses
- `CONSENSUS_TLS_CERT`: Path to TLS certificate
- `CONSENSUS_TLS_KEY`: Path to TLS private key
- `CONSENSUS_TLS_CA`: Path to CA certificate for mutual TLS

### Configuration File

```yaml
consensus:
  node_id: "node-1"
  bind_address: "0.0.0.0:7000"
  
  peers:
    - address: "node-2.cluster.local:7000"
      weight: 1
    - address: "node-3.cluster.local:7000"
      weight: 1
  
  protocol:
    type: "paxos"
    timeout: 5s
    retries: 3
    
  security:
    tls_enabled: true
    cert_file: "/certs/node.crt"
    key_file: "/certs/node.key"
    ca_file: "/certs/ca.crt"
    
  fault_tolerance:
    failure_detector: "phi_accrual"
    suspect_threshold: 8
    heartbeat_interval: 1s
```

## Development

### Running Tests

```bash
# Unit tests
pytest chil/consensus/tests/

# Integration tests
pytest chil/consensus/tests/integration/ --integration

# Chaos tests (requires Docker)
pytest chil/consensus/tests/chaos/ --chaos
```

### Building Proto Files

```bash
cd chil/consensus
./compile_protos.sh
```

## Performance Considerations

- Connection pooling for efficient peer communication
- Batching of consensus proposals
- Asynchronous message processing
- Configurable timeouts and retries
- Rate limiting for proposal submissions

## Security

- TLS/mTLS for all peer connections
- Message authentication and integrity
- Node identity verification
- Rate limiting and DoS protection
- Audit logging for all consensus decisions