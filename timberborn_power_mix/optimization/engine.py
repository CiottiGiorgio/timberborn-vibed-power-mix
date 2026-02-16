import logging
import numpy as np
from typing import Tuple, Optional
from timberborn_power_mix.simulation.models import SimulationConfig, EnergyMixConfig
from timberborn_power_mix.optimization.models import OptimizationConfig, FitnessResult
from timberborn_power_mix.simulation.engine import run_simulation
from timberborn_power_mix.machines import PRODUCER_DATABASE, BatteryName
from timberborn_power_mix import consts
from timberborn_power_mix.models import ConfigName
import timberborn_power_mix.optimization.helpers as opt_helpers

logger = logging.getLogger(__name__)

def evaluate_fitness(config: SimulationConfig) -> FitnessResult:
    result = run_simulation(config)
    return FitnessResult(
        cost=opt_helpers.calculate_total_wood_cost(config.energy_mix),
        reliability_score=np.percentile(result.aggregated_samples.hours_empty_results, 95),
        avg_production=float(np.mean(result.worst_sample.power_production)),
        avg_consumption=float(np.mean(result.aggregated_samples.power_consumption)),
    )

def get_random_mix(rng: np.random.Generator, max_machines: int = 100, max_height: int = 20) -> EnergyMixConfig:
    return EnergyMixConfig(
        **{BatteryName.BATTERY: rng.integers(0, max_machines, endpoint=True),
           BatteryName.BATTERY_HEIGHT: float(rng.integers(1, max_height, endpoint=True)),
           **{name: int(rng.integers(0, max_machines, endpoint=True)) for name in PRODUCER_DATABASE}}
    )

def mutate_mix(rng: np.random.Generator, mix: EnergyMixConfig, res: FitnessResult, threshold: float) -> EnergyMixConfig:
    is_feasible = res.reliability_score <= threshold
    mix_data = mix.model_dump()
    fields = list(PRODUCER_DATABASE.keys()) + [BatteryName.BATTERY, BatteryName.BATTERY_HEIGHT]

    weights = np.ones(len(fields))
    if not is_feasible:
        prod_low = res.avg_production < res.avg_consumption
        weights[np.isin(fields, list(PRODUCER_DATABASE.keys()))] = 10.0 if prod_low else 1.0
        weights[np.isin(fields, [BatteryName.BATTERY, BatteryName.BATTERY_HEIGHT])] = 1.0 if prod_low else 5.0

    for field in rng.choice(fields, size=rng.integers(1, 2, endpoint=True), p=weights/weights.sum()):
        inc_chance = (0.85 + 0.1 * (res.avg_production < res.avg_consumption and field in PRODUCER_DATABASE)) if not is_feasible else 0.15
        direction = 1 if rng.random() < inc_chance else -1

        if field == BatteryName.BATTERY_HEIGHT:
            mix_data[field] = max(1.0, round(mix_data[field] + direction * rng.uniform(0.5, 2.0), 1))
        else:
            mix_data[field] = max(0, int(mix_data[field] + direction * rng.integers(1, 5, endpoint=True)))

    return EnergyMixConfig(**mix_data)

def run_optimization(opt_config: OptimizationConfig) -> Tuple[Optional[EnergyMixConfig], float]:
    rng = np.random.default_rng(getattr(opt_config, ConfigName.SEED))
    common_data = opt_config.model_dump()
    iterations, _ = common_data.pop(ConfigName.ITERATION), common_data.pop(ConfigName.SEED)
    threshold = 0.05 * opt_config.days * consts.HOURS_PER_DAY

    current_mix = get_random_mix(rng)
    current_res = evaluate_fitness(SimulationConfig(**common_data, energy_mix=current_mix, seed=int(rng.integers(0, 2**32 - 1))))
    best_mix, best_cost = (current_mix, current_res.cost) if current_res.reliability_score <= threshold else (None, float("inf"))

    for i in range(iterations):
        next_mix = mutate_mix(rng, current_mix, current_res, threshold)
        next_res = evaluate_fitness(SimulationConfig(**common_data, energy_mix=next_mix, seed=int(rng.integers(0, 2**32 - 1))))

        curr_f, nxt_f = current_res.reliability_score <= threshold, next_res.reliability_score <= threshold
        if (not curr_f and (nxt_f or next_res.reliability_score < current_res.reliability_score)) or (curr_f and nxt_f and next_res.cost < current_res.cost):
            current_mix, current_res = next_mix, next_res
            if nxt_f and next_res.cost < best_cost:
                best_mix, best_cost = next_mix, next_res.cost
                logger.info(f"Iteration {i}: New best cost: {best_cost}")

        if i % 10 == 0 and i > 0:
            logger.info(f"Progress {i}/{iterations}: Reliability {current_res.reliability_score:.2f}, Cost {current_res.cost}")

    return best_mix, best_cost
