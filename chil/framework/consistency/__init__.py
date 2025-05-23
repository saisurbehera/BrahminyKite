"""
Consistency Verification Framework

Handles logical consistency through multiple approaches:
- Pattern-based consistency checking (consistency_real.py)
- Formal logic verification (consistency_formal.py)
- Theorem proving (Lean, Coq)
- SMT solving (Z3)
- Logic programming (Prolog)
- Contradiction detection
- Logical fallacy identification
"""

from .consistency_real import RealConsistencyFramework
from .consistency_formal import FormalConsistencyFramework

__all__ = ["RealConsistencyFramework", "FormalConsistencyFramework"]