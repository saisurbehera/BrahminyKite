"""
Repository for evolutionary framework database operations.
"""

import logging
from typing import List, Optional, Dict, Any, Tuple
from uuid import UUID
from datetime import datetime, timedelta
import json
import random

from sqlalchemy import select, func, and_, or_, desc
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from .base import BaseRepository, RepositoryError
from ..models.evolution import (
    EvolutionaryAgent,
    EvolutionGeneration,
    EvolutionMetrics,
    EvolutionState,
    AgentStatus,
    SelectionStrategy,
    MutationType
)

logger = logging.getLogger(__name__)


class EvolutionaryAgentRepository(BaseRepository[EvolutionaryAgent]):
    """Repository for evolutionary agents."""
    
    def __init__(self, session: AsyncSession):
        super().__init__(EvolutionaryAgent, session)
    
    async def get_by_identifier(self, agent_identifier: str) -> Optional[EvolutionaryAgent]:
        """Get agent by its unique identifier."""
        return await self.get_by(agent_identifier=agent_identifier)
    
    async def create_agent(
        self,
        agent_identifier: str,
        generation_id: Union[UUID, str],
        genome: Dict[str, Any],
        phenotype: Dict[str, Any],
        parent_agents: Optional[List[Union[UUID, str]]] = None,
        mutations: Optional[List[Dict[str, Any]]] = None,
        capabilities: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> EvolutionaryAgent:
        """Create a new evolutionary agent."""
        # Calculate lineage depth
        lineage_depth = 0
        if parent_agents:
            # Get parent with maximum depth
            parent_depths = []
            for parent_id in parent_agents:
                parent = await self.get(parent_id)
                if parent:
                    parent_depths.append(parent.lineage_depth)
            
            if parent_depths:
                lineage_depth = max(parent_depths) + 1
        
        return await self.create(
            agent_identifier=agent_identifier,
            generation_id=generation_id,
            parent_agents=parent_agents,
            lineage_depth=lineage_depth,
            genome=genome,
            phenotype=phenotype,
            mutations=mutations or [],
            fitness_score=0.0,
            status=AgentStatus.ACTIVE,
            capabilities=capabilities or [],
            metadata=metadata or {}
        )
    
    async def update_fitness(
        self,
        agent_id: Union[UUID, str],
        fitness_score: float,
        task_success_rate: Optional[float] = None,
        adaptability_score: Optional[float] = None,
        innovation_score: Optional[float] = None,
        stability_score: Optional[float] = None
    ) -> Optional[EvolutionaryAgent]:
        """Update agent fitness and performance scores."""
        updates = {"fitness_score": fitness_score}
        
        if task_success_rate is not None:
            updates["task_success_rate"] = task_success_rate
        if adaptability_score is not None:
            updates["adaptability_score"] = adaptability_score
        if innovation_score is not None:
            updates["innovation_score"] = innovation_score
        if stability_score is not None:
            updates["stability_score"] = stability_score
        
        return await self.update(agent_id, **updates)
    
    async def get_generation_agents(
        self,
        generation_id: Union[UUID, str],
        status: Optional[AgentStatus] = None,
        min_fitness: Optional[float] = None
    ) -> List[EvolutionaryAgent]:
        """Get all agents in a generation."""
        filters = {"generation_id": generation_id}
        
        if status:
            filters["status"] = status
        if min_fitness is not None:
            filters["fitness_score"] = {"gte": min_fitness}
        
        return await self.list(
            filters=filters,
            order_by="fitness_score",
            order_desc=True
        )
    
    async def get_elite_agents(
        self,
        generation_id: Union[UUID, str],
        elite_size: int
    ) -> List[EvolutionaryAgent]:
        """Get the top performing agents in a generation."""
        return await self.list(
            filters={
                "generation_id": generation_id,
                "status": AgentStatus.ACTIVE
            },
            order_by="fitness_score",
            order_desc=True,
            limit=elite_size
        )
    
    async def mark_evolved(
        self,
        agent_id: Union[UUID, str]
    ) -> Optional[EvolutionaryAgent]:
        """Mark an agent as evolved (selected for next generation)."""
        agent = await self.get(agent_id)
        if not agent:
            return None
        
        return await self.update(
            agent_id,
            status=AgentStatus.EVOLVED,
            selection_count=agent.selection_count + 1
        )
    
    async def get_lineage(
        self,
        agent_id: Union[UUID, str],
        max_depth: Optional[int] = None
    ) -> List[EvolutionaryAgent]:
        """Get the lineage of an agent (ancestors)."""
        agent = await self.get(agent_id)
        if not agent or not agent.parent_agents:
            return []
        
        lineage = []
        to_process = [(agent, 0)]
        
        while to_process:
            current_agent, depth = to_process.pop(0)
            
            if max_depth and depth >= max_depth:
                continue
            
            if current_agent.parent_agents:
                parents = await self.get_many(current_agent.parent_agents)
                lineage.extend(parents)
                
                for parent in parents:
                    to_process.append((parent, depth + 1))
        
        return lineage


class EvolutionGenerationRepository(BaseRepository[EvolutionGeneration]):
    """Repository for evolution generations."""
    
    def __init__(self, session: AsyncSession):
        super().__init__(EvolutionGeneration, session)
    
    async def create_generation(
        self,
        experiment_id: str,
        generation_number: int,
        population_size: int,
        selection_strategy: SelectionStrategy,
        mutation_rate: float = 0.1,
        mutation_types: Optional[List[MutationType]] = None,
        elite_size: int = 0,
        selection_params: Optional[Dict[str, Any]] = None
    ) -> EvolutionGeneration:
        """Create a new generation."""
        return await self.create(
            generation_number=generation_number,
            experiment_id=experiment_id,
            population_size=population_size,
            elite_size=elite_size,
            selection_strategy=selection_strategy,
            selection_params=selection_params or {},
            mutation_rate=mutation_rate,
            mutation_types=mutation_types or [MutationType.PARAMETER],
            started_at=datetime.utcnow()
        )
    
    async def complete_generation(
        self,
        generation_id: Union[UUID, str],
        avg_fitness: float,
        max_fitness: float,
        min_fitness: float,
        fitness_variance: float,
        genetic_diversity: Optional[float] = None,
        phenotypic_diversity: Optional[float] = None,
        improvement_rate: Optional[float] = None,
        convergence_score: Optional[float] = None
    ) -> Optional[EvolutionGeneration]:
        """Complete a generation with statistics."""
        return await self.update(
            generation_id,
            avg_fitness=avg_fitness,
            max_fitness=max_fitness,
            min_fitness=min_fitness,
            fitness_variance=fitness_variance,
            genetic_diversity=genetic_diversity,
            phenotypic_diversity=phenotypic_diversity,
            improvement_rate=improvement_rate,
            convergence_score=convergence_score,
            completed_at=datetime.utcnow()
        )
    
    async def get_latest_generation(
        self,
        experiment_id: str
    ) -> Optional[EvolutionGeneration]:
        """Get the latest generation for an experiment."""
        return await self.get_by(
            experiment_id=experiment_id,
            order_by="generation_number",
            order_desc=True
        )
    
    async def get_experiment_generations(
        self,
        experiment_id: str,
        limit: Optional[int] = None
    ) -> List[EvolutionGeneration]:
        """Get all generations for an experiment."""
        return await self.list(
            filters={"experiment_id": experiment_id},
            order_by="generation_number",
            order_desc=True,
            limit=limit,
            load_relationships=["agents"]
        )
    
    async def get_fitness_trajectory(
        self,
        experiment_id: str
    ) -> List[Dict[str, Any]]:
        """Get fitness trajectory across generations."""
        generations = await self.list(
            filters={"experiment_id": experiment_id},
            order_by="generation_number"
        )
        
        return [
            {
                "generation": gen.generation_number,
                "avg_fitness": gen.avg_fitness,
                "max_fitness": gen.max_fitness,
                "min_fitness": gen.min_fitness,
                "variance": gen.fitness_variance,
                "improvement_rate": gen.improvement_rate
            }
            for gen in generations
            if gen.avg_fitness is not None
        ]


class EvolutionMetricsRepository(BaseRepository[EvolutionMetrics]):
    """Repository for evolution metrics."""
    
    def __init__(self, session: AsyncSession):
        super().__init__(EvolutionMetrics, session)
    
    async def record_agent_metrics(
        self,
        agent_id: Union[UUID, str],
        evaluation_round: int,
        tasks_completed: int = 0,
        tasks_failed: int = 0,
        avg_task_time_ms: Optional[float] = None,
        exploration_rate: Optional[float] = None,
        exploitation_rate: Optional[float] = None,
        cooperation_score: Optional[float] = None,
        compute_efficiency: Optional[float] = None,
        memory_efficiency: Optional[float] = None,
        energy_efficiency: Optional[float] = None,
        learning_rate: Optional[float] = None,
        knowledge_retention: Optional[float] = None,
        generalization_score: Optional[float] = None,
        environment_fitness: Optional[Dict[str, float]] = None,
        robustness_score: Optional[float] = None
    ) -> EvolutionMetrics:
        """Record detailed metrics for an agent."""
        return await self.create(
            agent_id=agent_id,
            evaluation_round=evaluation_round,
            tasks_completed=tasks_completed,
            tasks_failed=tasks_failed,
            avg_task_time_ms=avg_task_time_ms,
            exploration_rate=exploration_rate,
            exploitation_rate=exploitation_rate,
            cooperation_score=cooperation_score,
            compute_efficiency=compute_efficiency,
            memory_efficiency=memory_efficiency,
            energy_efficiency=energy_efficiency,
            learning_rate=learning_rate,
            knowledge_retention=knowledge_retention,
            generalization_score=generalization_score,
            environment_fitness=environment_fitness,
            robustness_score=robustness_score
        )
    
    async def get_agent_performance_history(
        self,
        agent_id: Union[UUID, str],
        limit: Optional[int] = None
    ) -> List[EvolutionMetrics]:
        """Get performance history for an agent."""
        return await self.list(
            filters={"agent_id": agent_id},
            order_by="evaluation_round",
            order_desc=True,
            limit=limit
        )
    
    async def compare_agents(
        self,
        agent_ids: List[Union[UUID, str]],
        evaluation_round: int
    ) -> Dict[str, Dict[str, Any]]:
        """Compare metrics for multiple agents at a specific round."""
        metrics_list = await self.list(
            filters={
                "agent_id": agent_ids,
                "evaluation_round": evaluation_round
            }
        )
        
        comparison = {}
        for metrics in metrics_list:
            comparison[str(metrics.agent_id)] = {
                "task_success_rate": metrics.tasks_completed / (metrics.tasks_completed + metrics.tasks_failed) if (metrics.tasks_completed + metrics.tasks_failed) > 0 else 0,
                "avg_task_time_ms": metrics.avg_task_time_ms,
                "exploration_rate": metrics.exploration_rate,
                "exploitation_rate": metrics.exploitation_rate,
                "efficiency": {
                    "compute": metrics.compute_efficiency,
                    "memory": metrics.memory_efficiency,
                    "energy": metrics.energy_efficiency
                },
                "learning": {
                    "rate": metrics.learning_rate,
                    "retention": metrics.knowledge_retention,
                    "generalization": metrics.generalization_score
                },
                "robustness": metrics.robustness_score
            }
        
        return comparison


class EvolutionStateRepository(BaseRepository[EvolutionState]):
    """Repository for evolution state management."""
    
    def __init__(self, session: AsyncSession):
        super().__init__(EvolutionState, session)
    
    async def save_generation_state(
        self,
        generation_id: Union[UUID, str],
        experiment_id: str,
        population_snapshot: Dict[str, Any],
        elite_pool: List[Dict[str, Any]],
        current_params: Dict[str, Any],
        fitness_trajectory: List[float],
        diversity_trajectory: List[float],
        checkpoint_path: Optional[str] = None,
        bottlenecks: Optional[List[Dict[str, Any]]] = None,
        breakthrough_events: Optional[List[Dict[str, Any]]] = None
    ) -> EvolutionState:
        """Save the state of a generation."""
        # Get previous state for param history
        previous_state = await self.get_by(
            experiment_id=experiment_id,
            order_by="created_at",
            order_desc=True
        )
        
        param_history = []
        if previous_state:
            param_history = previous_state.param_history + [current_params]
        else:
            param_history = [current_params]
        
        return await self.create(
            generation_id=generation_id,
            experiment_id=experiment_id,
            population_snapshot=population_snapshot,
            elite_pool=elite_pool,
            current_params=current_params,
            param_history=param_history,
            fitness_trajectory=fitness_trajectory,
            diversity_trajectory=diversity_trajectory,
            checkpoint_path=checkpoint_path,
            is_recoverable=checkpoint_path is not None,
            bottlenecks=bottlenecks,
            breakthrough_events=breakthrough_events
        )
    
    async def get_latest_state(
        self,
        experiment_id: str
    ) -> Optional[EvolutionState]:
        """Get the latest state for an experiment."""
        query = select(self.model).where(
            self.model.experiment_id == experiment_id
        ).order_by(
            self.model.created_at.desc()
        ).limit(1)
        
        result = await self.session.execute(query)
        return result.scalar_one_or_none()
    
    async def identify_bottlenecks(
        self,
        experiment_id: str,
        window_size: int = 5
    ) -> List[Dict[str, Any]]:
        """Identify evolutionary bottlenecks in an experiment."""
        states = await self.list(
            filters={"experiment_id": experiment_id},
            order_by="created_at"
        )
        
        bottlenecks = []
        
        if len(states) < window_size:
            return bottlenecks
        
        for i in range(window_size, len(states)):
            window = states[i-window_size:i]
            fitness_values = []
            
            for state in window:
                if state.fitness_trajectory:
                    fitness_values.append(state.fitness_trajectory[-1])
            
            if fitness_values:
                # Check for stagnation
                fitness_std = np.std(fitness_values) if len(fitness_values) > 1 else 0
                if fitness_std < 0.01:  # Very low variance
                    bottlenecks.append({
                        "type": "stagnation",
                        "generation_range": [window[0].generation_id, window[-1].generation_id],
                        "fitness_range": [min(fitness_values), max(fitness_values)],
                        "duration": window_size
                    })
        
        return bottlenecks


class EvolutionRepository:
    """
    Composite repository for all evolutionary framework operations.
    
    Provides a unified interface for evolution data access.
    """
    
    def __init__(self, session: AsyncSession):
        self.session = session
        self.agents = EvolutionaryAgentRepository(session)
        self.generations = EvolutionGenerationRepository(session)
        self.metrics = EvolutionMetricsRepository(session)
        self.states = EvolutionStateRepository(session)
    
    async def start_experiment(
        self,
        experiment_id: str,
        initial_population_size: int,
        selection_strategy: SelectionStrategy,
        mutation_rate: float = 0.1,
        elite_percentage: float = 0.1,
        initial_genomes: Optional[List[Dict[str, Any]]] = None
    ) -> Tuple[EvolutionGeneration, List[EvolutionaryAgent]]:
        """Start a new evolutionary experiment."""
        # Create first generation
        generation = await self.generations.create_generation(
            experiment_id=experiment_id,
            generation_number=1,
            population_size=initial_population_size,
            selection_strategy=selection_strategy,
            mutation_rate=mutation_rate,
            elite_size=int(initial_population_size * elite_percentage)
        )
        
        # Create initial population
        agents = []
        for i in range(initial_population_size):
            genome = initial_genomes[i] if initial_genomes and i < len(initial_genomes) else self._generate_random_genome()
            phenotype = self._express_phenotype(genome)
            
            agent = await self.agents.create_agent(
                agent_identifier=f"{experiment_id}_gen1_agent{i}",
                generation_id=generation.id,
                genome=genome,
                phenotype=phenotype,
                capabilities=self._derive_capabilities(phenotype)
            )
            agents.append(agent)
        
        return generation, agents
    
    async def evolve_generation(
        self,
        experiment_id: str,
        fitness_evaluator: callable
    ) -> Tuple[EvolutionGeneration, List[EvolutionaryAgent]]:
        """Evolve to the next generation."""
        # Get current generation
        current_gen = await self.generations.get_latest_generation(experiment_id)
        if not current_gen:
            raise RepositoryError("No generation found for experiment")
        
        # Get all agents in current generation
        current_agents = await self.agents.get_generation_agents(current_gen.id)
        
        # Evaluate fitness if not already done
        for agent in current_agents:
            if agent.fitness_score == 0.0:
                fitness = await fitness_evaluator(agent)
                await self.agents.update_fitness(agent.id, fitness)
        
        # Refresh agents with updated fitness
        current_agents = await self.agents.get_generation_agents(current_gen.id)
        
        # Calculate generation statistics
        fitness_scores = [a.fitness_score for a in current_agents]
        avg_fitness = sum(fitness_scores) / len(fitness_scores)
        max_fitness = max(fitness_scores)
        min_fitness = min(fitness_scores)
        fitness_variance = np.var(fitness_scores) if len(fitness_scores) > 1 else 0
        
        # Calculate improvement rate
        improvement_rate = None
        if current_gen.generation_number > 1:
            prev_gens = await self.generations.get_experiment_generations(
                experiment_id,
                limit=2
            )
            if len(prev_gens) >= 2 and prev_gens[1].avg_fitness:
                improvement_rate = (avg_fitness - prev_gens[1].avg_fitness) / prev_gens[1].avg_fitness
        
        # Complete current generation
        await self.generations.complete_generation(
            current_gen.id,
            avg_fitness=avg_fitness,
            max_fitness=max_fitness,
            min_fitness=min_fitness,
            fitness_variance=fitness_variance,
            improvement_rate=improvement_rate
        )
        
        # Select parents for next generation
        parents = await self._select_parents(
            current_agents,
            current_gen.selection_strategy,
            current_gen.population_size,
            current_gen.elite_size
        )
        
        # Create next generation
        next_gen = await self.generations.create_generation(
            experiment_id=experiment_id,
            generation_number=current_gen.generation_number + 1,
            population_size=current_gen.population_size,
            selection_strategy=current_gen.selection_strategy,
            mutation_rate=current_gen.mutation_rate,
            mutation_types=current_gen.mutation_types,
            elite_size=current_gen.elite_size,
            selection_params=current_gen.selection_params
        )
        
        # Create offspring
        offspring = []
        for i in range(current_gen.population_size):
            if i < current_gen.elite_size:
                # Elite agents are copied directly
                parent = parents[i]
                genome = parent.genome.copy()
                phenotype = parent.phenotype.copy()
                parent_ids = [parent.id]
            else:
                # Create offspring through crossover and mutation
                parent1, parent2 = random.sample(parents[current_gen.elite_size:], 2)
                genome = self._crossover(parent1.genome, parent2.genome)
                mutations = self._mutate(genome, current_gen.mutation_rate, current_gen.mutation_types)
                phenotype = self._express_phenotype(genome)
                parent_ids = [parent1.id, parent2.id]
            
            agent = await self.agents.create_agent(
                agent_identifier=f"{experiment_id}_gen{next_gen.generation_number}_agent{i}",
                generation_id=next_gen.id,
                genome=genome,
                phenotype=phenotype,
                parent_agents=parent_ids,
                mutations=mutations if i >= current_gen.elite_size else [],
                capabilities=self._derive_capabilities(phenotype)
            )
            offspring.append(agent)
        
        # Mark selected parents as evolved
        for parent in parents:
            await self.agents.mark_evolved(parent.id)
        
        # Save generation state
        elite_pool = [
            {
                "agent_id": str(a.id),
                "fitness": a.fitness_score,
                "genome": a.genome
            }
            for a in parents[:current_gen.elite_size]
        ]
        
        await self.states.save_generation_state(
            generation_id=next_gen.id,
            experiment_id=experiment_id,
            population_snapshot={
                "size": len(offspring),
                "avg_fitness": avg_fitness,
                "diversity": self._calculate_diversity(current_agents)
            },
            elite_pool=elite_pool,
            current_params={
                "mutation_rate": current_gen.mutation_rate,
                "selection_strategy": current_gen.selection_strategy
            },
            fitness_trajectory=[avg_fitness],
            diversity_trajectory=[self._calculate_diversity(offspring)]
        )
        
        return next_gen, offspring
    
    async def _select_parents(
        self,
        agents: List[EvolutionaryAgent],
        strategy: SelectionStrategy,
        population_size: int,
        elite_size: int
    ) -> List[EvolutionaryAgent]:
        """Select parents based on strategy."""
        # Sort by fitness
        sorted_agents = sorted(agents, key=lambda a: a.fitness_score, reverse=True)
        
        # Always include elites
        parents = sorted_agents[:elite_size]
        
        # Select additional parents based on strategy
        remaining_slots = population_size - elite_size
        
        if strategy == SelectionStrategy.FITNESS_PROPORTIONATE:
            # Roulette wheel selection
            fitness_sum = sum(a.fitness_score for a in sorted_agents[elite_size:])
            if fitness_sum > 0:
                for _ in range(remaining_slots):
                    r = random.uniform(0, fitness_sum)
                    cumsum = 0
                    for agent in sorted_agents[elite_size:]:
                        cumsum += agent.fitness_score
                        if cumsum >= r:
                            parents.append(agent)
                            break
        
        elif strategy == SelectionStrategy.TOURNAMENT:
            # Tournament selection
            tournament_size = 3
            for _ in range(remaining_slots):
                tournament = random.sample(sorted_agents[elite_size:], tournament_size)
                winner = max(tournament, key=lambda a: a.fitness_score)
                parents.append(winner)
        
        elif strategy == SelectionStrategy.RANK_BASED:
            # Rank-based selection
            ranks = list(range(len(sorted_agents), 0, -1))
            rank_sum = sum(ranks[elite_size:])
            for _ in range(remaining_slots):
                r = random.uniform(0, rank_sum)
                cumsum = 0
                for i, agent in enumerate(sorted_agents[elite_size:], elite_size):
                    cumsum += ranks[i]
                    if cumsum >= r:
                        parents.append(agent)
                        break
        
        return parents
    
    def _generate_random_genome(self) -> Dict[str, Any]:
        """Generate a random genome."""
        return {
            "layer_sizes": [random.randint(16, 128) for _ in range(random.randint(2, 5))],
            "activation": random.choice(["relu", "tanh", "sigmoid"]),
            "learning_rate": random.uniform(0.0001, 0.1),
            "dropout_rate": random.uniform(0.0, 0.5),
            "optimizer": random.choice(["adam", "sgd", "rmsprop"])
        }
    
    def _express_phenotype(self, genome: Dict[str, Any]) -> Dict[str, Any]:
        """Express genome as phenotype."""
        return {
            "architecture": {
                "type": "neural_network",
                "layers": genome.get("layer_sizes", [64, 32]),
                "activation": genome.get("activation", "relu")
            },
            "training": {
                "learning_rate": genome.get("learning_rate", 0.001),
                "dropout": genome.get("dropout_rate", 0.1),
                "optimizer": genome.get("optimizer", "adam")
            },
            "behavior": {
                "exploration_tendency": random.uniform(0.1, 0.9),
                "risk_tolerance": random.uniform(0.1, 0.9)
            }
        }
    
    def _derive_capabilities(self, phenotype: Dict[str, Any]) -> List[str]:
        """Derive capabilities from phenotype."""
        capabilities = ["basic_reasoning"]
        
        # Add capabilities based on architecture
        if len(phenotype["architecture"]["layers"]) > 3:
            capabilities.append("deep_learning")
        
        if phenotype["training"]["learning_rate"] < 0.01:
            capabilities.append("fine_tuning")
        
        if phenotype["behavior"]["exploration_tendency"] > 0.7:
            capabilities.append("exploration")
        
        return capabilities
    
    def _crossover(self, genome1: Dict[str, Any], genome2: Dict[str, Any]) -> Dict[str, Any]:
        """Perform crossover between two genomes."""
        offspring = {}
        
        for key in genome1:
            if random.random() < 0.5:
                offspring[key] = genome1[key]
            else:
                offspring[key] = genome2.get(key, genome1[key])
        
        return offspring
    
    def _mutate(
        self,
        genome: Dict[str, Any],
        mutation_rate: float,
        mutation_types: List[MutationType]
    ) -> List[Dict[str, Any]]:
        """Apply mutations to genome."""
        mutations = []
        
        for key, value in genome.items():
            if random.random() < mutation_rate:
                mutation_type = random.choice(mutation_types)
                
                if mutation_type == MutationType.PARAMETER:
                    if isinstance(value, (int, float)):
                        # Gaussian mutation
                        mutated_value = value * (1 + random.gauss(0, 0.1))
                        genome[key] = type(value)(mutated_value)
                        mutations.append({
                            "type": "parameter",
                            "gene": key,
                            "old_value": value,
                            "new_value": genome[key]
                        })
                
                elif mutation_type == MutationType.STRUCTURAL:
                    if key == "layer_sizes" and isinstance(value, list):
                        # Add or remove layer
                        if random.random() < 0.5 and len(value) > 1:
                            # Remove layer
                            idx = random.randint(0, len(value) - 1)
                            value.pop(idx)
                            mutations.append({
                                "type": "structural",
                                "action": "remove_layer",
                                "position": idx
                            })
                        else:
                            # Add layer
                            idx = random.randint(0, len(value))
                            size = random.randint(16, 128)
                            value.insert(idx, size)
                            mutations.append({
                                "type": "structural",
                                "action": "add_layer",
                                "position": idx,
                                "size": size
                            })
        
        return mutations
    
    def _calculate_diversity(self, agents: List[EvolutionaryAgent]) -> float:
        """Calculate genetic diversity of population."""
        if len(agents) < 2:
            return 0.0
        
        # Simple diversity measure based on genome differences
        total_distance = 0
        comparisons = 0
        
        for i in range(len(agents)):
            for j in range(i + 1, len(agents)):
                distance = self._genome_distance(agents[i].genome, agents[j].genome)
                total_distance += distance
                comparisons += 1
        
        return total_distance / comparisons if comparisons > 0 else 0.0
    
    def _genome_distance(self, genome1: Dict[str, Any], genome2: Dict[str, Any]) -> float:
        """Calculate distance between two genomes."""
        distance = 0.0
        
        for key in set(genome1.keys()) | set(genome2.keys()):
            val1 = genome1.get(key)
            val2 = genome2.get(key)
            
            if val1 is None or val2 is None:
                distance += 1.0
            elif isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
                distance += abs(val1 - val2) / max(abs(val1), abs(val2), 1)
            elif val1 != val2:
                distance += 1.0
        
        return distance

# Import numpy if available, otherwise use basic calculations
try:
    import numpy as np
except ImportError:
    logger.warning("NumPy not available, using basic calculations for variance")
    
    class np:
        @staticmethod
        def var(values):
            if len(values) < 2:
                return 0
            mean = sum(values) / len(values)
            return sum((x - mean) ** 2 for x in values) / len(values)
        
        @staticmethod
        def std(values):
            return np.var(values) ** 0.5