# Chil

A unified philosophical verification and distributed consensus framework named after the Chilika Chil (Brahminy Kite). Combines individual claim verification with distributed consensus protocols.

## Overview

Chil integrates six philosophical verification frameworks with modified Paxos consensus to create a dual-mode system:

- **Individual Mode**: Traditional claim verification using empirical, contextual, consistency, power dynamics, utility, and evolution frameworks
- **Consensus Mode**: Distributed verification across multiple nodes using philosophical Paxos protocol  
- **Hybrid Mode**: Both capabilities active simultaneously

## Quick Start

```bash
# Install chil package
pip install -e .

# Run individual verification
python scripts/run_verifier.py --mode individual

# Run consensus verification  
python scripts/run_verifier.py --mode consensus

# Run tests
python tests/run_tests.py

# Use in Python
python -c "import chil; v = chil.create_verifier(); print('Chil ready!')"
```

## Architecture

```
chil/                   # Main package
├── framework/          # Core verification frameworks
│   ├── individual/     # Individual verification components  
│   ├── consensus/      # Consensus protocol implementation
│   ├── meta/          # Meta-verification capabilities
│   └── consensus_types.py  # Shared type definitions
├── system/            # System orchestration
│   ├── orchestration/ # Core orchestrator and mode bridging
│   └── compatibility/ # Backward compatibility layer
└── config/            # Configuration management

docs/                  # Documentation
├── philosophy/        # Philosophical foundations
├── architecture/      # System design documents  
└── examples/          # Usage examples

tests/                 # Comprehensive test suite
├── unit/             # Unit tests
├── integration/      # Integration tests
└── compatibility/    # Backward compatibility tests

scripts/               # Utility scripts
```


## Why the name ?
As many of you know, I have a deep love for birds. All my project names are inspired by birds observed near Chilika, Odisha.

This one draws inspiration from the ଚିଲିକା ଚିଲ (Chilika Chil), the Brahminy Kite—a majestic hunter whose wings carve arcs across sunlit skies.

Just like them, I hope this project treads gracefully in its domain. 🌿✨


<img src="https://inaturalist-open-data.s3.amazonaws.com/photos/782915/large.jpg" alt="drawing" width="400"/>
