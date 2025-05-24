"""
High-performance client for Empirical Framework Tools.
"""

import grpc
import asyncio
from typing import Dict, List, Optional, Any
from functools import lru_cache

from ..protos import tools_pb2
from ..protos import tools_pb2_grpc


class EmpiricalToolsClient:
    """Client for empirical framework tools with caching and batching."""
    
    def __init__(self, host: str = 'localhost', port: int = 50051):
        self.channel = grpc.insecure_channel(
            f'{host}:{port}',
            options=[
                ('grpc.max_send_message_length', 50 * 1024 * 1024),
                ('grpc.max_receive_message_length', 50 * 1024 * 1024),
                ('grpc.keepalive_time_ms', 10000),
            ]
        )
        self.stub = tools_pb2_grpc.EmpiricalToolsStub(self.channel)
        self._cache = {}
    
    @lru_cache(maxsize=1000)
    def check_logical_consistency(self, formula: str, 
                                constraints: List[str] = None,
                                timeout_ms: int = 5000) -> Dict[str, Any]:
        """Check logical consistency with caching."""
        request = tools_pb2.Z3Request(
            formula=formula,
            constraints=constraints or [],
            timeout_ms=timeout_ms
        )
        
        try:
            response = self.stub.CheckLogicalConsistency(request)
            return {
                'satisfiable': response.satisfiable,
                'model': dict(response.model),
                'proof': response.proof
            }
        except grpc.RpcError as e:
            return {'error': str(e), 'satisfiable': False}
    
    def prove_theorem(self, theorem: str, axioms: List[str], 
                     tactic: str = 'auto') -> Dict[str, Any]:
        """Prove theorem using Lean."""
        request = tools_pb2.LeanRequest(
            theorem=theorem,
            axioms=axioms,
            tactic=tactic
        )
        
        try:
            response = self.stub.ProveTheorem(request)
            return {
                'proven': response.proven,
                'proof': response.proof,
                'steps': list(response.steps)
            }
        except grpc.RpcError as e:
            return {'error': str(e), 'proven': False}
    
    def query_facts(self, query: str, parameters: Dict[str, str] = None) -> List[Dict[str, str]]:
        """Query fact database."""
        request = tools_pb2.DuckDBRequest(
            query=query,
            parameters=parameters or {}
        )
        
        try:
            response = self.stub.QueryFactDatabase(request)
            return [dict(row.columns) for row in response.results]
        except grpc.RpcError as e:
            return []
    
    def query_knowledge_graph(self, sparql: str, graph_uri: str = None) -> List[Dict[str, str]]:
        """Query knowledge graph."""
        request = tools_pb2.SparqlRequest(
            query=sparql,
            graph_uri=graph_uri or ''
        )
        
        try:
            response = self.stub.QueryKnowledgeGraph(request)
            return [dict(binding.variables) for binding in response.bindings]
        except grpc.RpcError as e:
            return []
    
    async def batch_check_consistency(self, formulas: List[str]) -> List[Dict[str, Any]]:
        """Batch check multiple formulas asynchronously."""
        tasks = [
            asyncio.create_task(
                asyncio.to_thread(self.check_logical_consistency, formula)
            )
            for formula in formulas
        ]
        return await asyncio.gather(*tasks)
    
    def close(self):
        """Close the gRPC channel."""
        self.channel.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()