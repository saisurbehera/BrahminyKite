"""
Logic Solver Integrations

Provides integration with SAT solvers and formal logic systems.
"""

import logging
from typing import List, Dict, Any, Optional, Tuple, Set
from dataclasses import dataclass
from abc import ABC, abstractmethod
from enum import Enum
import tempfile
import subprocess
import os

# Try to import Z3
try:
    import z3
    HAS_Z3 = True
except ImportError:
    HAS_Z3 = False
    z3 = None

# Try to import PySAT
try:
    from pysat.solvers import Glucose3, Minisat22
    from pysat.formula import CNF
    HAS_PYSAT = True
except ImportError:
    HAS_PYSAT = False


class LogicalOperator(Enum):
    """Logical operators"""
    AND = "and"
    OR = "or"
    NOT = "not"
    IMPLIES = "implies"
    IFF = "iff"  # if and only if
    XOR = "xor"  # exclusive or


@dataclass
class LogicalVariable:
    """Logical variable/proposition"""
    name: str
    value: Optional[bool] = None
    
    def __hash__(self):
        return hash(self.name)
    
    def __eq__(self, other):
        return isinstance(other, LogicalVariable) and self.name == other.name


@dataclass
class LogicalFormula:
    """Logical formula representation"""
    operator: Optional[LogicalOperator]
    operands: List[Any]  # Can be LogicalVariable or nested LogicalFormula
    
    def __str__(self):
        if self.operator is None:
            return str(self.operands[0])
        elif self.operator == LogicalOperator.NOT:
            return f"¬{self.operands[0]}"
        else:
            op_symbol = {
                LogicalOperator.AND: "∧",
                LogicalOperator.OR: "∨",
                LogicalOperator.IMPLIES: "→",
                LogicalOperator.IFF: "↔",
                LogicalOperator.XOR: "⊕"
            }
            symbol = op_symbol.get(self.operator, self.operator.value)
            return f"({' ' + symbol + ' '.join(str(op) for op in self.operands)})"


@dataclass
class SATResult:
    """SAT solver result"""
    satisfiable: bool
    model: Optional[Dict[str, bool]] = None  # Variable assignments if SAT
    proof: Optional[str] = None  # Unsatisfiability proof if UNSAT
    time_ms: Optional[float] = None


@dataclass
class ProofStep:
    """Step in a logical proof"""
    formula: str
    justification: str
    dependencies: List[int] = None  # Indices of previous steps
    
    def __post_init__(self):
        if self.dependencies is None:
            self.dependencies = []


class LogicSolver(ABC):
    """Abstract base class for logic solvers"""
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    @abstractmethod
    def solve_sat(self, formula: LogicalFormula) -> SATResult:
        """Solve a SAT problem"""
        pass
    
    @abstractmethod
    def check_consistency(self, formulas: List[LogicalFormula]) -> bool:
        """Check if a set of formulas is consistent"""
        pass
    
    @abstractmethod
    def prove_entailment(self, premises: List[LogicalFormula], conclusion: LogicalFormula) -> Tuple[bool, Optional[List[ProofStep]]]:
        """Check if premises entail conclusion"""
        pass


class Z3Solver(LogicSolver):
    """Z3 theorem prover integration"""
    
    def __init__(self):
        super().__init__()
        if not HAS_Z3:
            raise ImportError("Z3 not installed. Install with: pip install z3-solver")
    
    def solve_sat(self, formula: LogicalFormula) -> SATResult:
        """Solve SAT using Z3"""
        import time
        start_time = time.time()
        
        # Convert to Z3 formula
        z3_formula, var_map = self._to_z3(formula)
        
        # Create solver
        solver = z3.Solver()
        solver.add(z3_formula)
        
        # Check satisfiability
        result = solver.check()
        
        elapsed_ms = (time.time() - start_time) * 1000
        
        if result == z3.sat:
            # Get model
            model = solver.model()
            assignments = {}
            
            for var_name, z3_var in var_map.items():
                val = model.eval(z3_var)
                if val is not None:
                    assignments[var_name] = bool(val)
                else:
                    assignments[var_name] = False  # Default
            
            return SATResult(
                satisfiable=True,
                model=assignments,
                time_ms=elapsed_ms
            )
        else:
            # Get unsat core if available
            proof = None
            if result == z3.unsat:
                try:
                    proof = str(solver.unsat_core())
                except:
                    pass
            
            return SATResult(
                satisfiable=False,
                proof=proof,
                time_ms=elapsed_ms
            )
    
    def check_consistency(self, formulas: List[LogicalFormula]) -> bool:
        """Check consistency of formula set"""
        solver = z3.Solver()
        
        # Add all formulas
        for formula in formulas:
            z3_formula, _ = self._to_z3(formula)
            solver.add(z3_formula)
        
        return solver.check() == z3.sat
    
    def prove_entailment(self, premises: List[LogicalFormula], conclusion: LogicalFormula) -> Tuple[bool, Optional[List[ProofStep]]]:
        """Prove that premises entail conclusion"""
        solver = z3.Solver()
        
        # Add premises
        for premise in premises:
            z3_premise, _ = self._to_z3(premise)
            solver.add(z3_premise)
        
        # Add negation of conclusion
        z3_conclusion, _ = self._to_z3(conclusion)
        solver.add(z3.Not(z3_conclusion))
        
        # If UNSAT, then premises entail conclusion
        result = solver.check()
        
        if result == z3.unsat:
            # Generate proof steps
            proof_steps = self._generate_proof_steps(premises, conclusion)
            return True, proof_steps
        else:
            return False, None
    
    def _to_z3(self, formula: LogicalFormula) -> Tuple[Any, Dict[str, Any]]:
        """Convert LogicalFormula to Z3 expression"""
        var_map = {}
        
        def convert(f):
            if isinstance(f, LogicalVariable):
                if f.name not in var_map:
                    var_map[f.name] = z3.Bool(f.name)
                return var_map[f.name]
            elif isinstance(f, LogicalFormula):
                if f.operator == LogicalOperator.NOT:
                    return z3.Not(convert(f.operands[0]))
                elif f.operator == LogicalOperator.AND:
                    return z3.And([convert(op) for op in f.operands])
                elif f.operator == LogicalOperator.OR:
                    return z3.Or([convert(op) for op in f.operands])
                elif f.operator == LogicalOperator.IMPLIES:
                    return z3.Implies(convert(f.operands[0]), convert(f.operands[1]))
                elif f.operator == LogicalOperator.IFF:
                    a = convert(f.operands[0])
                    b = convert(f.operands[1])
                    return z3.And(z3.Implies(a, b), z3.Implies(b, a))
                elif f.operator == LogicalOperator.XOR:
                    return z3.Xor(convert(f.operands[0]), convert(f.operands[1]))
                else:
                    # Treat as variable if no operator
                    return convert(f.operands[0])
            else:
                raise ValueError(f"Unknown formula type: {type(f)}")
        
        return convert(formula), var_map
    
    def _generate_proof_steps(self, premises: List[LogicalFormula], conclusion: LogicalFormula) -> List[ProofStep]:
        """Generate natural deduction style proof steps"""
        steps = []
        
        # Add premises
        for i, premise in enumerate(premises):
            steps.append(ProofStep(
                formula=str(premise),
                justification="Premise"
            ))
        
        # Add simplified proof steps (in real implementation, would use Z3's proof objects)
        steps.append(ProofStep(
            formula=str(conclusion),
            justification="Follows from premises (Z3)",
            dependencies=list(range(len(premises)))
        ))
        
        return steps
    
    def simplify_formula(self, formula: LogicalFormula) -> LogicalFormula:
        """Simplify a logical formula"""
        z3_formula, var_map = self._to_z3(formula)
        simplified = z3.simplify(z3_formula)
        
        # Convert back (simplified version)
        # Note: This is a simplified conversion, real implementation would be more complex
        return formula  # For now, return original


class MiniSATSolver(LogicSolver):
    """MiniSAT solver integration using PySAT"""
    
    def __init__(self):
        super().__init__()
        if not HAS_PYSAT:
            raise ImportError("PySAT not installed. Install with: pip install python-sat")
    
    def solve_sat(self, formula: LogicalFormula) -> SATResult:
        """Solve SAT using MiniSAT"""
        import time
        start_time = time.time()
        
        # Convert to CNF
        cnf, var_map = self._to_cnf(formula)
        
        # Create solver
        solver = Minisat22()
        
        # Add clauses
        for clause in cnf.clauses:
            solver.add_clause(clause)
        
        # Solve
        satisfiable = solver.solve()
        
        elapsed_ms = (time.time() - start_time) * 1000
        
        if satisfiable:
            # Get model
            model = solver.get_model()
            assignments = {}
            
            # Reverse var_map to get variable names
            reverse_map = {v: k for k, v in var_map.items()}
            
            for lit in model:
                var_num = abs(lit)
                if var_num in reverse_map:
                    var_name = reverse_map[var_num]
                    assignments[var_name] = lit > 0
            
            solver.delete()
            return SATResult(
                satisfiable=True,
                model=assignments,
                time_ms=elapsed_ms
            )
        else:
            solver.delete()
            return SATResult(
                satisfiable=False,
                time_ms=elapsed_ms
            )
    
    def check_consistency(self, formulas: List[LogicalFormula]) -> bool:
        """Check consistency using MiniSAT"""
        cnf = CNF()
        var_counter = 1
        var_map = {}
        
        # Convert all formulas to CNF
        for formula in formulas:
            formula_cnf, formula_var_map = self._to_cnf(formula, var_counter, var_map)
            var_counter = max(formula_var_map.values()) + 1 if formula_var_map else var_counter
            var_map.update(formula_var_map)
            
            # Add clauses
            for clause in formula_cnf.clauses:
                cnf.append(clause)
        
        # Solve
        solver = Minisat22()
        for clause in cnf.clauses:
            solver.add_clause(clause)
        
        result = solver.solve()
        solver.delete()
        
        return result
    
    def prove_entailment(self, premises: List[LogicalFormula], conclusion: LogicalFormula) -> Tuple[bool, Optional[List[ProofStep]]]:
        """Prove entailment using MiniSAT"""
        cnf = CNF()
        var_counter = 1
        var_map = {}
        
        # Add premises
        for premise in premises:
            premise_cnf, premise_var_map = self._to_cnf(premise, var_counter, var_map)
            var_counter = max(premise_var_map.values()) + 1 if premise_var_map else var_counter
            var_map.update(premise_var_map)
            
            for clause in premise_cnf.clauses:
                cnf.append(clause)
        
        # Add negation of conclusion
        neg_conclusion = LogicalFormula(LogicalOperator.NOT, [conclusion])
        neg_cnf, neg_var_map = self._to_cnf(neg_conclusion, var_counter, var_map)
        var_map.update(neg_var_map)
        
        for clause in neg_cnf.clauses:
            cnf.append(clause)
        
        # Solve
        solver = Minisat22()
        for clause in cnf.clauses:
            solver.add_clause(clause)
        
        result = not solver.solve()  # UNSAT means entailment holds
        solver.delete()
        
        if result:
            # Generate simple proof
            steps = []
            for i, premise in enumerate(premises):
                steps.append(ProofStep(
                    formula=str(premise),
                    justification="Premise"
                ))
            
            steps.append(ProofStep(
                formula=str(conclusion),
                justification="Follows from premises (MiniSAT)",
                dependencies=list(range(len(premises)))
            ))
            
            return True, steps
        else:
            return False, None
    
    def _to_cnf(self, formula: LogicalFormula, var_counter: int = 1, 
                existing_vars: Optional[Dict[str, int]] = None) -> Tuple[CNF, Dict[str, int]]:
        """Convert formula to CNF format"""
        if existing_vars is None:
            existing_vars = {}
        
        var_map = existing_vars.copy()
        cnf = CNF()
        
        # Simplified CNF conversion (Tseitin transformation would be used in practice)
        # This is a basic implementation
        def get_var_num(var_name: str) -> int:
            if var_name not in var_map:
                nonlocal var_counter
                var_map[var_name] = var_counter
                var_counter += 1
            return var_map[var_name]
        
        def to_clauses(f, polarity=True):
            if isinstance(f, LogicalVariable):
                var_num = get_var_num(f.name)
                return [[var_num if polarity else -var_num]]
            
            elif isinstance(f, LogicalFormula):
                if f.operator == LogicalOperator.NOT:
                    return to_clauses(f.operands[0], not polarity)
                
                elif f.operator == LogicalOperator.AND:
                    if polarity:
                        # All must be true
                        clauses = []
                        for op in f.operands:
                            clauses.extend(to_clauses(op, True))
                        return clauses
                    else:
                        # At least one must be false
                        clause = []
                        for op in f.operands:
                            sub_clauses = to_clauses(op, False)
                            if len(sub_clauses) == 1 and len(sub_clauses[0]) == 1:
                                clause.append(sub_clauses[0][0])
                        return [clause] if clause else []
                
                elif f.operator == LogicalOperator.OR:
                    if polarity:
                        # At least one must be true
                        clause = []
                        for op in f.operands:
                            sub_clauses = to_clauses(op, True)
                            if len(sub_clauses) == 1 and len(sub_clauses[0]) == 1:
                                clause.append(sub_clauses[0][0])
                        return [clause] if clause else []
                    else:
                        # All must be false
                        clauses = []
                        for op in f.operands:
                            clauses.extend(to_clauses(op, False))
                        return clauses
                
                # For other operators, use Tseitin variables
                else:
                    # Simplified handling
                    return []
            
            return []
        
        # Convert formula to clauses
        clauses = to_clauses(formula, True)
        for clause in clauses:
            if clause:  # Don't add empty clauses
                cnf.append(clause)
        
        return cnf, var_map


class LogicHub:
    """Hub for coordinating multiple logic solvers"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.solvers = {}
        
        # Initialize available solvers
        if HAS_Z3:
            try:
                self.solvers['z3'] = Z3Solver()
                self.logger.info("Z3 solver initialized")
            except Exception as e:
                self.logger.warning(f"Failed to initialize Z3: {e}")
        
        if HAS_PYSAT:
            try:
                self.solvers['minisat'] = MiniSATSolver()
                self.logger.info("MiniSAT solver initialized")
            except Exception as e:
                self.logger.warning(f"Failed to initialize MiniSAT: {e}")
        
        if not self.solvers:
            self.logger.warning("No logic solvers available!")
    
    def solve_sat(self, formula: LogicalFormula, solver: str = 'z3') -> Optional[SATResult]:
        """Solve SAT problem using specified solver"""
        if solver in self.solvers:
            return self.solvers[solver].solve_sat(formula)
        elif self.solvers:
            # Fallback to any available solver
            fallback = list(self.solvers.keys())[0]
            self.logger.warning(f"Solver {solver} not available, using {fallback}")
            return self.solvers[fallback].solve_sat(formula)
        else:
            return None
    
    def check_consistency(self, formulas: List[LogicalFormula], solver: str = 'z3') -> Optional[bool]:
        """Check consistency using specified solver"""
        if solver in self.solvers:
            return self.solvers[solver].check_consistency(formulas)
        elif self.solvers:
            fallback = list(self.solvers.keys())[0]
            return self.solvers[fallback].check_consistency(formulas)
        else:
            return None
    
    def prove_entailment(self, premises: List[LogicalFormula], conclusion: LogicalFormula, 
                        solver: str = 'z3') -> Tuple[Optional[bool], Optional[List[ProofStep]]]:
        """Prove entailment using specified solver"""
        if solver in self.solvers:
            return self.solvers[solver].prove_entailment(premises, conclusion)
        elif self.solvers:
            fallback = list(self.solvers.keys())[0]
            return self.solvers[fallback].prove_entailment(premises, conclusion)
        else:
            return None, None
    
    def parse_formula(self, formula_str: str) -> Optional[LogicalFormula]:
        """Parse a string into a logical formula"""
        # Simple parser for basic formulas
        # In practice, would use a proper parser
        
        # Remove spaces
        formula_str = formula_str.strip()
        
        # Handle basic cases
        if formula_str.startswith('NOT ') or formula_str.startswith('¬'):
            inner = formula_str[4:] if formula_str.startswith('NOT ') else formula_str[1:]
            inner_formula = self.parse_formula(inner)
            if inner_formula:
                return LogicalFormula(LogicalOperator.NOT, [inner_formula])
        
        # Check for binary operators
        for op_text, op_enum in [
            (' AND ', LogicalOperator.AND),
            (' OR ', LogicalOperator.OR),
            (' IMPLIES ', LogicalOperator.IMPLIES),
            (' IFF ', LogicalOperator.IFF),
            (' XOR ', LogicalOperator.XOR)
        ]:
            if op_text in formula_str:
                parts = formula_str.split(op_text)
                if len(parts) == 2:
                    left = self.parse_formula(parts[0])
                    right = self.parse_formula(parts[1])
                    if left and right:
                        return LogicalFormula(op_enum, [left, right])
        
        # Must be a variable
        if formula_str.isalnum():
            return LogicalFormula(None, [LogicalVariable(formula_str)])
        
        return None
    
    def create_formula_builder(self):
        """Create a formula builder for easier formula construction"""
        return FormulaBuilder()


class FormulaBuilder:
    """Helper class for building logical formulas"""
    
    def var(self, name: str) -> LogicalFormula:
        """Create a variable"""
        return LogicalFormula(None, [LogicalVariable(name)])
    
    def and_(self, *operands) -> LogicalFormula:
        """Create AND formula"""
        return LogicalFormula(LogicalOperator.AND, list(operands))
    
    def or_(self, *operands) -> LogicalFormula:
        """Create OR formula"""
        return LogicalFormula(LogicalOperator.OR, list(operands))
    
    def not_(self, operand) -> LogicalFormula:
        """Create NOT formula"""
        return LogicalFormula(LogicalOperator.NOT, [operand])
    
    def implies(self, antecedent, consequent) -> LogicalFormula:
        """Create implication"""
        return LogicalFormula(LogicalOperator.IMPLIES, [antecedent, consequent])
    
    def iff(self, left, right) -> LogicalFormula:
        """Create if-and-only-if"""
        return LogicalFormula(LogicalOperator.IFF, [left, right])
    
    def xor(self, left, right) -> LogicalFormula:
        """Create exclusive-or"""
        return LogicalFormula(LogicalOperator.XOR, [left, right])