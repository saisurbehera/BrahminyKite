"""
External API Integrations for BrahminyKite

Provides connectors to real-world data sources and services.
"""

from .fact_checking import FactCheckingHub, SnopesAPI, PolitiFactAPI, FactCheckOrgAPI
from .academic import AcademicHub, PubMedAPI, ArXivAPI, CrossRefAPI, SemanticScholarAPI
from .nlp import NLPHub, SpacyProcessor, TransformerProcessor
from .logic import LogicHub, Z3Solver, MiniSATSolver

__all__ = [
    # Fact Checking
    'FactCheckingHub',
    'SnopesAPI',
    'PolitiFactAPI',
    'FactCheckOrgAPI',
    
    # Academic
    'AcademicHub',
    'PubMedAPI',
    'ArXivAPI',
    'CrossRefAPI',
    'SemanticScholarAPI',
    
    # NLP
    'NLPHub',
    'SpacyProcessor',
    'TransformerProcessor',
    
    # Logic
    'LogicHub',
    'Z3Solver',
    'MiniSATSolver'
]