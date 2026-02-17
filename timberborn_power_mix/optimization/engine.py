import logging
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from concurrent.futures import ProcessPoolExecutor

from timberborn_power_mix.simulation.models import SimulationConfig, EnergyMixConfig
from timberborn_power_mix.optimization.models import OptimizationConfig
from timberborn_power_mix.simulation.engine import run_simulation_singlethread
from timberborn_power_mix.machines import PRODUCER_DATABASE, BatteryName
from timberborn_power_mix import consts, helpers
import timberborn_power_mix.optimization.helpers as opt_helpers
import timberborn_power_mix.simulation.helpers as sim_helpers

logger = logging.getLogger(__name__)


class Individual:
    """Represents a single power grid configuration in the population."""

    def __init__(self, mix: EnergyMixConfig):
        self.mix = mix
        self.cost: float = 0.0
        self.battery_stress: float = 0.0  # Objective 1: Minimize
        self.hours_empty_pct: float = 0.0  # Selection criteria

        self.rank: int = 0
        self.crowding_distance: float = 0.0
        self.domination_count: int = 0
        self.dominated_solutions: List["Individual"] = []

    def set_results(self, cost: float, battery_stress: float, hours_empty_pct: float):
        """Sets the evaluation results calculated externally."""
        self.cost = cost
        self.battery_stress = battery_stress
        self.hours_empty_pct = hours_empty_pct


def evaluate_individual_task(
    mix: EnergyMixConfig, sim_config_base: Dict[str, Any], seed: int
) -> Tuple[float, float, float]:
    """
    Standalone task for parallel evaluation.
    Returns (cost, battery_stress, hours_empty_pct).
    """
    config = SimulationConfig(**sim_config_base, energy_mix=mix, seed=seed)

    # Use singlethread here because we are parallelizing at the individual level
    result = run_simulation_singlethread(config)

    cost = opt_helpers.calculate_total_wood_cost(mix)
    capacity = sim_helpers.calculate_total_battery_capacity(mix)
    battery_stress = sim_helpers.calculate_battery_stress(
        result.worst_sample.battery_charge, capacity
    )

    total_hours = sim_config_base["days"] * consts.HOURS_PER_DAY
    avg_hours_empty = np.mean(result.aggregated_samples.hours_empty_results)
    hours_empty_pct = avg_hours_empty / total_hours

    return cost, battery_stress, hours_empty_pct


def evaluate_population(
    population: List[Individual],
    sim_config_base: Dict[str, Any],
    rng: np.random.Generator,
    executor: ProcessPoolExecutor,
):
    """Evaluates a list of individuals in parallel."""
    tasks = []
    for ind in population:
        seed = int(rng.integers(0, 2**32 - 1))
        tasks.append(
            executor.submit(evaluate_individual_task, ind.mix, sim_config_base, seed)
        )

    for ind, task in zip(population, tasks):
        cost, stress, pct = task.result()
        ind.set_results(cost, stress, pct)


def fast_non_dominated_sort(population: List[Individual]) -> List[List[Individual]]:
    """Groups individuals into Pareto fronts based on Cost and Battery Stress."""
    fronts: List[List[Individual]] = [[]]
    for p in population:
        p.domination_count = 0
        p.dominated_solutions = []
        for q in population:
            if (p.cost <= q.cost and p.battery_stress <= q.battery_stress) and (
                p.cost < q.cost or p.battery_stress < q.battery_stress
            ):
                p.dominated_solutions.append(q)
            elif (q.cost <= p.cost and q.battery_stress <= p.battery_stress) and (
                q.cost < p.cost or q.battery_stress < p.battery_stress
            ):
                p.domination_count += 1
        if p.domination_count == 0:
            p.rank = 1
            fronts[0].append(p)

    i = 0
    while len(fronts[i]) > 0:
        next_front = []
        for p in fronts[i]:
            for q in p.dominated_solutions:
                q.domination_count -= 1
                if q.domination_count == 0:
                    q.rank = i + 2
                    next_front.append(q)
        i += 1
        fronts.append(next_front)
    return fronts[:-1]


def calculate_crowding_distance(front: List[Individual]):
    """Calculates diversity score for individuals in a front."""
    size = len(front)
    if size == 0:
        return
    if size <= 2:
        for ind in front:
            ind.crowding_distance = float("inf")
        return

    for ind in front:
        ind.crowding_distance = 0.0
    for obj in ["cost", "battery_stress"]:
        front.sort(key=lambda x: getattr(x, obj))
        front[0].crowding_distance = float("inf")
        front[-1].crowding_distance = float("inf")
        obj_range = getattr(front[-1], obj) - getattr(front[0], obj)
        if obj_range > 0:
            for i in range(1, size - 1):
                front[i].crowding_distance += (
                    getattr(front[i + 1], obj) - getattr(front[i - 1], obj)
                ) / obj_range


def tournament_selection(
    population: List[Individual], rng: np.random.Generator
) -> Individual:
    """Selects the best individual from a random subset."""
    participants = rng.choice(population, size=2, replace=False)
    a, b = participants[0], participants[1]
    if a.rank < b.rank:
        return a
    if b.rank < a.rank:
        return b
    return a if a.crowding_distance > b.crowding_distance else b


def crossover(p1: Individual, p2: Individual, rng: np.random.Generator) -> Individual:
    """Uniform crossover for building counts."""
    mix1 = p1.mix.model_dump()
    mix2 = p2.mix.model_dump()
    child_mix = {}
    for key in mix1.keys():
        child_mix[key] = mix1[key] if rng.random() < 0.5 else mix2[key]
    return Individual(EnergyMixConfig(**child_mix))


def mutate(
    ind: Individual, rng: np.random.Generator, max_machines: int = 100
) -> Individual:
    """Randomly tweaks building counts."""
    mix_data = ind.mix.model_dump()
    fields_to_mutate = rng.choice(list(mix_data.keys()), size=rng.integers(1, 3))
    for field in fields_to_mutate:
        if field == BatteryName.BATTERY_HEIGHT:
            mix_data[field] = max(
                1.0, round(mix_data[field] + rng.uniform(-2.0, 2.0), 1)
            )
        else:
            delta = rng.integers(-5, 6)
            mix_data[field] = max(0, min(max_machines, int(mix_data[field] + delta)))
    return Individual(EnergyMixConfig(**mix_data))


def get_random_individual(
    rng: np.random.Generator, max_machines: int = 50, max_height: int = 20
) -> Individual:
    data = {
        BatteryName.BATTERY: int(rng.integers(0, max_machines)),
        BatteryName.BATTERY_HEIGHT: float(rng.integers(1, max_height)),
        **{name: int(rng.integers(0, max_machines)) for name in PRODUCER_DATABASE},
    }
    return Individual(EnergyMixConfig(**data))


def run_optimization(
    opt_config: OptimizationConfig,
) -> Tuple[Optional[EnergyMixConfig], float]:
    """Main NSGA-II Loop with parallel evaluation."""
    rng = np.random.default_rng(opt_config.seed)
    pop_size = 40
    generations = opt_config.iteration

    sim_config_base = opt_config.model_dump()
    sim_config_base.pop("iteration")
    sim_config_base.pop("seed")

    # Respect user threads config for the ProcessPool
    max_workers = helpers.calculate_optimal_threads(opt_config.threads, pop_size)

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # 1. Initialize Population
        population = [get_random_individual(rng) for _ in range(pop_size)]
        evaluate_population(population, sim_config_base, rng, executor)

        for gen in range(generations):
            # 2. Create Offspring
            offspring = []
            while len(offspring) < pop_size:
                p1 = tournament_selection(population, rng)
                p2 = tournament_selection(population, rng)
                child = crossover(p1, p2, rng)
                child = mutate(child, rng)
                offspring.append(child)

            # 3. Evaluate Offspring in Parallel
            evaluate_population(offspring, sim_config_base, rng, executor)

            # 4. Combine and Sort
            combined = population + offspring
            fronts = fast_non_dominated_sort(combined)

            # 5. Survival Selection
            new_population = []
            for front in fronts:
                calculate_crowding_distance(front)
                if len(new_population) + len(front) <= pop_size:
                    new_population.extend(front)
                else:
                    front.sort(key=lambda x: x.crowding_distance, reverse=True)
                    new_population.extend(front[: pop_size - len(new_population)])
                    break

            population = new_population

            if gen % 5 == 0:
                best_front = [ind for ind in population if ind.rank == 1]
                logger.info(f"Gen {gen}: Pareto Front Size {len(best_front)}")

    # 6. Final Selection
    pareto_front = [ind for ind in population if ind.rank == 1]
    if not pareto_front:
        return None, 0.0

    target = 0.05
    best_ind = min(pareto_front, key=lambda ind: abs(ind.hours_empty_pct - target))

    logger.info(
        f"Optimization complete. Selected solution with {best_ind.hours_empty_pct:.2%} unreliability."
    )
    return best_ind.mix, best_ind.cost
