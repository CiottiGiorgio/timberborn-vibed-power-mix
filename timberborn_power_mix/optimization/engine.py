import logging
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from concurrent.futures import ProcessPoolExecutor

from timberborn_power_mix.simulation.models import SimulationConfig, EnergyMixConfig
from timberborn_power_mix.optimization.models import OptimizationConfig, Individual
from timberborn_power_mix.simulation.engine import run_simulation_singlethread
from timberborn_power_mix.machines import PRODUCER_DATABASE, BatteryName
from timberborn_power_mix.simulation import consts as sim_consts
from timberborn_power_mix import helpers
import timberborn_power_mix.optimization.helpers as opt_helpers

logger = logging.getLogger(__name__)


# TODO:
# - check that all plurals exposed in the CLI make sense, either keep is consistently singular or consistently plural
# - check that we could migrate the optimization engine to pymoo
# - check that battery height is used propertly throughout the codebase (total capacity where needed, discrete heights where needed)
# - check that we can better type the return types passed around in the optimization engine
# - check that we can make more tests on the simulation engine on a more modular level (unit tests, etc.)
# - check that we can make tests for the optimization engine
# - find a good strategy to run tests automatically
# - write ci/cd for tests and linting (not packaging)
# - modularize the profiling scripts and output files


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

    # Use 95th percentile for both Stress and Hours Empty for consistency
    battery_stress = np.percentile(result.aggregated_samples.stress_results, 95)

    total_hours = sim_config_base["days"] * sim_consts.HOURS_PER_DAY
    worst_case_hours_empty = np.percentile(
        result.aggregated_samples.hours_empty_results, 95
    )
    hours_empty_pct = worst_case_hours_empty / total_hours

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
    """Uniform crossover for building counts and battery heights."""
    mix1 = p1.mix.model_dump()
    mix2 = p2.mix.model_dump()
    child_mix = {}

    # Crossover producers
    for key in PRODUCER_DATABASE.keys():
        child_mix[key] = mix1[key] if rng.random() < 0.5 else mix2[key]

    # Crossover battery heights (list)
    h1, h2 = mix1[BatteryName.BATTERY_HEIGHT], mix2[BatteryName.BATTERY_HEIGHT]
    max_len = max(len(h1), len(h2))
    child_heights = []
    for i in range(max_len):
        if i < len(h1) and i < len(h2):
            child_heights.append(h1[i] if rng.random() < 0.5 else h2[i])
        elif i < len(h1):
            if rng.random() < 0.5:
                child_heights.append(h1[i])
        elif i < len(h2):
            if rng.random() < 0.5:
                child_heights.append(h2[i])

    child_mix[BatteryName.BATTERY_HEIGHT] = child_heights
    return Individual(EnergyMixConfig(**child_mix))


def mutate(
    ind: Individual, rng: np.random.Generator, max_machines: int = 100
) -> Individual:
    """Randomly tweaks building counts and battery heights."""
    mix_data = ind.mix.model_dump()
    producers = list(PRODUCER_DATABASE.keys())

    # Decide whether to mutate producers or batteries
    if rng.random() < 0.7:
        # Mutate producers
        fields_to_mutate = rng.choice(producers, size=rng.integers(1, 3))
        for field in fields_to_mutate:
            delta = rng.integers(-5, 6)
            mix_data[field] = max(0, min(max_machines, int(mix_data[field] + delta)))
    else:
        # Mutate batteries
        heights = mix_data[BatteryName.BATTERY_HEIGHT]
        mutation_type = rng.random()
        if mutation_type < 0.2 and len(heights) < max_machines:
            # Add a battery
            heights.append(rng.integers(1, 21))
        elif mutation_type < 0.4 and len(heights) > 0:
            # Remove a battery
            heights.pop(rng.integers(0, len(heights)))
        elif len(heights) > 0:
            # Tweak an existing battery's height
            idx = rng.integers(0, len(heights))
            heights[idx] = max(1, min(20, int(heights[idx] + rng.integers(-3, 4))))

        mix_data[BatteryName.BATTERY_HEIGHT] = heights

    return Individual(EnergyMixConfig(**mix_data))


def get_random_individual(
    rng: np.random.Generator, max_machines: int = 50, max_height: int = 20
) -> Individual:
    num_batteries = rng.integers(0, max_machines)
    data = {
        BatteryName.BATTERY_HEIGHT: [
            int(rng.integers(1, max_height + 1)) for _ in range(num_batteries)
        ],
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
