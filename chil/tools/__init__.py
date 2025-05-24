"""
High-performance tool integration layer for BrahminyKite.

This module provides gRPC-based services for framework-specific tools,
optimized for local execution with minimal latency.
"""

from .services import (
    EmpiricalToolsService,
    ContextualToolsService,
    ConsistencyToolsService,
    PowerDynamicsToolsService,
    UtilityToolsService,
    EvolutionToolsService
)

from .clients import (
    EmpiricalToolsClient,
    ContextualToolsClient,
    ConsistencyToolsClient,
    PowerDynamicsToolsClient,
    UtilityToolsClient,
    EvolutionToolsClient
)

__all__ = [
    # Services
    'EmpiricalToolsService',
    'ContextualToolsService',
    'ConsistencyToolsService',
    'PowerDynamicsToolsService',
    'UtilityToolsService',
    'EvolutionToolsService',
    
    # Clients
    'EmpiricalToolsClient',
    'ContextualToolsClient',
    'ConsistencyToolsClient',
    'PowerDynamicsToolsClient',
    'UtilityToolsClient',
    'EvolutionToolsClient',
]