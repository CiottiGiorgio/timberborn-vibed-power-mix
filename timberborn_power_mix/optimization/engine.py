import logging
import random
import math
import numpy as np
from typing import Tuple
from timberborn_power_mix.simulation.models import (
    SimulationConfig,
    EnergyMixConfig,
)
from timberborn_power_mix.optimization.models import OptimizationConfig
from timberborn_power_mix.simulation.engine import run_simulation
from timberborn_power_mix.machines import (
    PRODUCER_DATABASE,
    BatteryName,
)
from timberborn_power_mix import consts
from timberborn_power_mix.models import ConfigName
import timberborn_power_mix.simulation.helpers as sim_helpers

logger = logging.getLogger(__name__)


def evaluate_fitness(config: SimulationConfig) -> Tuple[bool, float, float]:
    """
    Returns (is_feasible, cost, reliability_score).
    reliability_score is the 95th percentile of hours empty.
    """
    res = run_simulation(config)
    total_hours = config.days * consts.HOURS_PER_DAY

    # 5% of time empty threshold
    threshold_hours = 0.05 * total_hours

    # 95th percentile of hours empty
    percentile_95 = np.percentile(res.aggregated_samples.hours_empty_results, 95)

    is_feasible = percentile_95 <= threshold_hours
    cost = sim_helpers.calculate_total_wood_cost(config.energy_mix)

    return is_feasible, cost, percentile_95


def get_random_mix(
    max_machines: int = 100, max_battery_height: int = 20
) -> EnergyMixConfig:
    mix_data = {
        BatteryName.BATTERY: random.randint(0, max_machines),
        BatteryName.BATTERY_HEIGHT: float(random.randint(1, max_battery_height)),
    }
    for name in PRODUCER_DATABASE.keys():
        mix_data[name] = random.randint(0, max_machines)
    return EnergyMixConfig(**mix_data)


def mutate_mix(
    mix: EnergyMixConfig,
    is_feasible: bool,
    reliability_score: float,
    threshold_hours: float,
    temp_factor: float,
) -> EnergyMixConfig:
    """
    Aggressive guided mutation.
    - Mutates multiple fields at once.
    - Uses larger step sizes scaled by temperature.
    - Strongly biased by feasibility.
    """
    mix_data = mix.model_dump()
    fields = list(mix_data.keys())

    # Mutate 1 to 3 fields simultaneously
    num_to_mutate = random.randint(1, 3)
    fields_to_mutate = random.sample(fields, num_to_mutate)

    for field in fields_to_mutate:
        # Determine direction based on feasibility
        if not is_feasible:
            # We need more power or storage - 90% bias to increase
            direction = 1 if random.random() < 0.9 else -1
        else:
            # We are reliable, try to reduce cost
            # Bias towards reduction scales with safety margin and temperature
            safety_margin = (threshold_hours - reliability_score) / threshold_hours
            reduction_bias = 0.3 + safety_margin * 0.6
            direction = -1 if random.random() < reduction_bias else 1

        # Aggressive step sizes scaled by temperature factor (0.0 to 1.0)
        # At high temp, we can jump up to 20 units. At low temp, we fine-tune.
        max_step = max(1, int(20 * temp_factor))
        step = random.randint(1, max_step)

        if field == BatteryName.BATTERY_HEIGHT:
            # Height changes are more sensitive
            height_step = max(1.0, 5.0 * temp_factor)
            delta = direction * random.uniform(1.0, height_step)
            mix_data[field] = max(1.0, round(mix_data[field] + delta, 1))
        else:
            mix_data[field] = max(0, mix_data[field] + direction * step)

    return EnergyMixConfig(**mix_data)


def calculate_energy(
    is_feasible: bool, cost: float, reliability_score: float, threshold_hours: float
) -> float:
    """
    Energy function for simulated annealing.
    """
    if not is_feasible:
        # Heavy penalty for infeasibility to force the search into the feasible region
        return 10_000_000 + (reliability_score - threshold_hours) * 5000
    return cost


def run_optimization(opt_config: OptimizationConfig):
    iterations = getattr(opt_config, ConfigName.ITERATION)
    logger.info(
        f"Starting AGGRESSIVE guided simulated annealing for {iterations} iterations..."
    )

    common_data = opt_config.model_dump()
    common_data.pop(ConfigName.ITERATION)

    total_hours = opt_config.days * consts.HOURS_PER_DAY
    threshold_hours = 0.05 * total_hours

    # Initial state
    current_mix = get_random_mix()
    config = SimulationConfig(**common_data, energy_mix=current_mix)
    is_feasible, cost, reliability = evaluate_fitness(config)
    current_energy = calculate_energy(is_feasible, cost, reliability, threshold_hours)

    best_feasible_mix = current_mix if is_feasible else None
    best_feasible_cost = cost if is_feasible else float("inf")

    # Annealing parameters
    initial_temp = 5000.0  # Higher initial temp for more exploration
    final_temp = 1.0

    for i in range(iterations):
        # Progress factor (1.0 at start, 0.0 at end)
        progress_factor = 1.0 - (i / iterations)
        temp = initial_temp * (progress_factor**2) + final_temp  # Quadratic cooling

        # Propose aggressive mutation
        next_mix = mutate_mix(
            current_mix, is_feasible, reliability, threshold_hours, progress_factor
        )
        next_config = SimulationConfig(**common_data, energy_mix=next_mix)
        next_is_feasible, next_cost, next_reliability = evaluate_fitness(next_config)
        next_energy = calculate_energy(
            next_is_feasible, next_cost, next_reliability, threshold_hours
        )

        # Acceptance probability
        if next_energy < current_energy:
            accept = True
        else:
            delta_e = next_energy - current_energy
            # Metropolis criterion
            accept = random.random() < math.exp(-delta_e / temp)

        if accept:
            current_mix = next_mix
            current_energy = next_energy
            is_feasible = next_is_feasible
            reliability = next_reliability

            if next_is_feasible:
                if next_cost < best_feasible_cost:
                    best_feasible_cost = next_cost
                    best_feasible_mix = next_mix
                    logger.info(
                        f"Iteration {i}: New best feasible mix found! Cost: {next_cost} (Temp: {temp:.2f})"
                    )

        if i % 10 == 0 and i > 0:
            logger.info(
                f"Progress: {i}/{iterations} iterations... Current Energy: {current_energy:.2f}, Temp: {temp:.2f}"
            )

    if best_feasible_mix:
        logger.info("Optimization finished!")
        logger.info(f"Best Energy Mix (Cost: {best_feasible_cost}):")
        for field, value in best_feasible_mix.model_dump().items():
            if value > 0:
                logger.info(f"  {field}: {value}")
    else:
        logger.warning(
            "Could not find a feasible solution within the given iterations."
        )
