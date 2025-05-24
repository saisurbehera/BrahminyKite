"""
Utility Framework Tools Service Implementation.
"""

import grpc
import time
import pickle
from concurrent import futures
from typing import Dict, List, Any

import numpy as np
import xgboost as xgb
from ortools.linear_solver import pywraplp
from numba import jit

from ..protos import tools_pb2
from ..protos import tools_pb2_grpc


class UtilityToolsServicer(tools_pb2_grpc.UtilityToolsServicer):
    """gRPC service for utility framework tools."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.models = {}
        self._load_models()
    
    def _load_models(self):
        """Load pre-trained XGBoost/LightGBM models."""
        import os
        models_path = self.config.get('xgboost_models', './models/xgboost/')
        if os.path.exists(models_path):
            for model_file in os.listdir(models_path):
                if model_file.endswith('.json'):
                    model_name = model_file.replace('.json', '')
                    model = xgb.Booster()
                    model.load_model(os.path.join(models_path, model_file))
                    self.models[model_name] = model
    
    def PredictOutcome(self, request, context):
        """Predict outcomes using gradient boosting."""
        try:
            # Prepare features
            feature_data = []
            feature_names = []
            
            for feature in request.features:
                row = []
                for name, value in sorted(feature.values.items()):
                    if name not in feature_names:
                        feature_names.append(name)
                    row.append(value)
                feature_data.append(row)
            
            # Convert to DMatrix
            dmatrix = xgb.DMatrix(
                np.array(feature_data),
                feature_names=feature_names
            )
            
            # Get model
            model = self.models.get(request.model_name)
            if not model:
                # Create default model if not found
                model = xgb.XGBRegressor(n_estimators=100, max_depth=3)
                # In production, this would be pre-trained
            
            # Predict
            predictions = model.predict(dmatrix)
            
            response = tools_pb2.GBMResponse()
            response.predictions.extend(predictions.tolist())
            
            # Get feature importance
            if hasattr(model, 'get_score'):
                importance = model.get_score(importance_type='gain')
                for feat, imp in importance.items():
                    fi = response.importances.add()
                    fi.feature = feat
                    fi.importance = float(imp)
            
            return response
            
        except Exception as e:
            context.abort(grpc.StatusCode.INTERNAL, f"XGBoost error: {str(e)}")
    
    def Optimize(self, request, context):
        """Solve optimization problem using OR-Tools."""
        try:
            # Create solver
            solver = pywraplp.Solver.CreateSolver('GLOP')
            if not solver:
                raise Exception("Could not create solver")
            
            # Create variables
            variables = {}
            for var in request.variables:
                variables[var.name] = solver.NumVar(
                    var.min, var.max, var.name
                )
            
            # Add constraints
            for constraint in request.constraints:
                # Parse constraint expression
                # In production, use proper expression parser
                expr = constraint.expression
                for var_name, var in variables.items():
                    expr = expr.replace(var_name, f"variables['{var_name}']")
                
                if constraint.type == "<=":
                    solver.Add(eval(expr) <= constraint.value)
                elif constraint.type == ">=":
                    solver.Add(eval(expr) >= constraint.value)
                else:  # "=="
                    solver.Add(eval(expr) == constraint.value)
            
            # Set objective
            objective_expr = request.objective
            for var_name, var in variables.items():
                objective_expr = objective_expr.replace(
                    var_name, f"variables['{var_name}']"
                )
            
            if request.maximize:
                solver.Maximize(eval(objective_expr))
            else:
                solver.Minimize(eval(objective_expr))
            
            # Solve
            status = solver.Solve()
            
            response = tools_pb2.OptimizationResponse()
            
            if status == pywraplp.Solver.OPTIMAL:
                response.status = "OPTIMAL"
                response.objective_value = solver.Objective().Value()
                
                for var_name, var in variables.items():
                    response.solution[var_name] = var.solution_value()
            else:
                response.status = "INFEASIBLE"
            
            return response
            
        except Exception as e:
            context.abort(grpc.StatusCode.INTERNAL, f"OR-Tools error: {str(e)}")
    
    def SolveGame(self, request, context):
        """Solve game theory problems."""
        try:
            # Simple Nash equilibrium solver
            # In production, use more sophisticated algorithms
            
            response = tools_pb2.GameResponse()
            
            # Convert payoff matrix to numpy
            payoff_matrix = np.array(request.payoff_matrix)
            
            # Find pure strategy Nash equilibria
            n_players = len(request.players)
            n_strategies = [len(p.strategies) for p in request.players]
            
            # For 2-player games, find equilibria
            if n_players == 2:
                for i in range(n_strategies[0]):
                    for j in range(n_strategies[1]):
                        # Check if (i,j) is Nash equilibrium
                        is_nash = True
                        
                        # Check player 1's best response
                        for i_prime in range(n_strategies[0]):
                            if payoff_matrix[i_prime][j][0] > payoff_matrix[i][j][0]:
                                is_nash = False
                                break
                        
                        # Check player 2's best response
                        if is_nash:
                            for j_prime in range(n_strategies[1]):
                                if payoff_matrix[i][j_prime][1] > payoff_matrix[i][j][1]:
                                    is_nash = False
                                    break
                        
                        if is_nash:
                            eq = response.equilibria.add()
                            eq.mixed_strategies[request.players[0].id] = float(i)
                            eq.mixed_strategies[request.players[1].id] = float(j)
                            eq.payoff = payoff_matrix[i][j][0]
            
            return response
            
        except Exception as e:
            context.abort(grpc.StatusCode.INTERNAL, f"Game theory error: {str(e)}")
    
    @staticmethod
    @jit(nopython=True)
    def _compute_expected_value(values, probabilities):
        """Numba-accelerated expected value computation."""
        return np.sum(values * probabilities)
    
    @staticmethod
    @jit(nopython=True)
    def _compute_risk_adjusted_utility(values, probabilities, risk_aversion):
        """Numba-accelerated risk-adjusted utility."""
        utilities = np.exp(-risk_aversion * values)
        expected_utility = np.sum(utilities * probabilities)
        return -np.log(expected_utility) / risk_aversion
    
    def ComputeUtility(self, request, context):
        """Compute utility using Numba-accelerated functions."""
        try:
            inputs = np.array(request.inputs)
            
            response = tools_pb2.UtilityResponse()
            
            if request.function == "expected_value":
                probabilities = np.array([
                    request.parameters.get(f"p{i}", 1.0/len(inputs))
                    for i in range(len(inputs))
                ])
                response.utility = float(
                    self._compute_expected_value(inputs, probabilities)
                )
                
            elif request.function == "risk_adjusted":
                risk_aversion = request.parameters.get("risk_aversion", 1.0)
                probabilities = np.array([
                    request.parameters.get(f"p{i}", 1.0/len(inputs))
                    for i in range(len(inputs))
                ])
                response.utility = float(
                    self._compute_risk_adjusted_utility(
                        inputs, probabilities, risk_aversion
                    )
                )
                
            else:
                # Default: mean utility
                response.utility = float(np.mean(inputs))
            
            # Add components breakdown
            response.components["mean"] = float(np.mean(inputs))
            response.components["std"] = float(np.std(inputs))
            response.components["min"] = float(np.min(inputs))
            response.components["max"] = float(np.max(inputs))
            
            return response
            
        except Exception as e:
            context.abort(grpc.StatusCode.INTERNAL, f"Utility computation error: {str(e)}")


def serve(port: int = 50055, config: Dict[str, Any] = None):
    """Start the utility tools gRPC server."""
    config = config or {}
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    tools_pb2_grpc.add_UtilityToolsServicer_to_server(
        UtilityToolsServicer(config), server
    )
    server.add_insecure_port(f'[::]:{port}')
    server.start()
    return server


if __name__ == '__main__':
    server = serve()
    server.wait_for_termination()