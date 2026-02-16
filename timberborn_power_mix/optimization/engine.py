import logging
import numpy as np
from typing import Tuple, Optional
from timberborn_power_mix.simulation.models import (
    SimulationConfig,
    EnergyMixConfig,
)
from timberborn_power_mix.optimization.models import OptimizationConfig, FitnessResult
from timberborn_power_mix.simulation.engine import run_simulation
from timberborn_power_mix.machines import (
    PRODUCER_DATABASE,
    BatteryName,
)
from timberborn_power_mix import consts
from timberborn_power_mix.models import ConfigName
import timberborn_power_mix.optimization.helpers as opt_helpers

logger = logging.getLogger(__name__)


def evaluate_fitness(config: SimulationConfig) -> FitnessResult:
    """
    Returns FitnessResult(cost, reliability_score, avg_production, avg_consumption).
    reliability_score is the 95th percentile of hours empty.
    """
    res = run_simulation(config)
    # 95th percentile of hours empty
    percentile_95 = np.percentile(res.aggregated_samples.hours_empty_results, 95)
    cost = opt_helpers.calculate_total_wood_cost(config.energy_mix)

    # Calculate averages from simulation results
    avg_consumption = float(np.mean(res.aggregated_samples.power_consumption))
    avg_production = float(np.mean(res.worst_sample.power_production))

    return FitnessResult(
        cost=cost,
        reliability_score=percentile_95,
        avg_production=avg_production,
        avg_consumption=avg_consumption,
    )


def get_random_mix(
    rng: np.random.Generator, max_machines: int = 100, max_battery_height: int = 20
) -> EnergyMixConfig:
    mix_data = {
        BatteryName.BATTERY: rng.integers(0, max_machines, endpoint=True),
        BatteryName.BATTERY_HEIGHT: float(
            rng.integers(1, max_battery_height, endpoint=True)
        ),
    }
    for name in PRODUCER_DATABASE.keys():
        mix_data[name] = int(rng.integers(0, max_machines, endpoint=True))
    return EnergyMixConfig(**mix_data)


def mutate_mix(
    rng: np.random.Generator,
    mix: EnergyMixConfig,
    res: FitnessResult,
    threshold_hours: float,
) -> EnergyMixConfig:
    """
    Guided mutation for hill climbing.
    - Mutates 1 to 2 fields.
    - Biases field selection and direction based on production/consumption ratio.
    """
    is_feasible = res.reliability_score <= threshold_hours
    mix_data = mix.model_dump()

    producer_fields = list(PRODUCER_DATABASE.keys())
    battery_fields = [BatteryName.BATTERY, BatteryName.BATTERY_HEIGHT]
    all_fields = producer_fields + battery_fields

    # 1. Determine field selection weights
    weights = np.ones(len(all_fields))
    if not is_feasible:
        if res.avg_production < res.avg_consumption:
            # Production is the bottleneck: strongly bias towards producers
            for i, field in enumerate(all_fields):
                if field in producer_fields:
                    weights[i] = 10.0
        else:
            # Production is okay: bias towards batteries to handle peaks/droughts
            for i, field in enumerate(all_fields):
                if field in battery_fields:
                    weights[i] = 5.0

    weights /= np.sum(weights)

    # 2. Select fields to mutate
    num_to_mutate = rng.integers(1, 2, endpoint=True)
    fields_to_mutate = rng.choice(
        all_fields, size=num_to_mutate, replace=False, p=weights
    )

    # 3. Apply mutations
    for field in fields_to_mutate:
        if not is_feasible:
            # Infeasible: bias towards increasing the selected field
            if res.avg_production < res.avg_consumption and field in producer_fields:
                direction = 1 if rng.random() < 0.95 else -1
            else:
                direction = 1 if rng.random() < 0.85 else -1
        else:
            # Feasible: bias towards decreasing to reduce cost
            direction = -1 if rng.random() < 0.85 else 1

        step = rng.integers(1, 5, endpoint=True)

        if field == BatteryName.BATTERY_HEIGHT:
            delta = direction * rng.uniform(0.5, 2.0)
            mix_data[field] = max(1.0, round(mix_data[field] + delta, 1))
        else:
            mix_data[field] = max(0, int(mix_data[field] + direction * step))

    return EnergyMixConfig(**mix_data)


def run_optimization(
    opt_config: OptimizationConfig,
) -> Tuple[Optional[EnergyMixConfig], float]:
    iterations = getattr(opt_config, ConfigName.ITERATION)
    seed = getattr(opt_config, ConfigName.SEED)

    rng = np.random.default_rng(seed)

    common_data = opt_config.model_dump()
    common_data.pop(ConfigName.ITERATION)
    common_data.pop(ConfigName.SEED)

    total_hours = opt_config.days * consts.HOURS_PER_DAY
    threshold_hours = 0.05 * total_hours

    # Initial state
    current_mix = get_random_mix(rng)
    config = SimulationConfig(
        **common_data, energy_mix=current_mix, seed=int(rng.integers(0, 2**32 - 1))
    )
    current_res = evaluate_fitness(config)

    best_feasible_mix = (
        current_mix
        if current_res.reliability_score <= threshold_hours
        else None
    )
    best_feasible_cost = (
        current_res.cost
        if current_res.reliability_score <= threshold_hours
        else float("inf")
    )

    for i in range(iterations):
        # Propose mutation
        next_mix = mutate_mix(
            rng,
            current_mix,
            current_res,
            threshold_hours,
        )
        next_config = SimulationConfig(
            **common_data, energy_mix=next_mix, seed=int(rng.integers(0, 2**32 - 1))
        )
        next_res = evaluate_fitness(next_config)

        # Acceptance logic (Hill Climbing)
        accept = False
        curr_feasible = current_res.reliability_score <= threshold_hours
        next_feasible = next_res.reliability_score <= threshold_hours

        if not curr_feasible:
            if next_feasible:
                # Found first feasible solution
                accept = True
            elif next_res.reliability_score < current_res.reliability_score:
                # Both infeasible, but next is better
                accept = True
        else:
            if next_feasible and next_res.cost < current_res.cost:
                # Both feasible, but next is cheaper
                accept = True

        if accept:
            current_mix = next_mix
            current_res = next_res

            if next_feasible:
                if next_res.cost < best_feasible_cost:
                    best_feasible_cost = next_res.cost
                    best_feasible_mix = next_mix
                    logger.info(
                        f"Iteration {i}: New best feasible mix found! Cost: {next_res.cost}"
                    )

        if i % 10 == 0 and i > 0:
            logger.info(
                f"Progress: {i}/{iterations} iterations... Current Reliability: {current_res.reliability_score:.2f}, Cost: {current_res.cost}"
            )

    return best_feasible_mix, best_feasible_cost
