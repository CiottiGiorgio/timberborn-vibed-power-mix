import logging
from typing import List, Dict, Tuple, Optional

import numpy as np
from numpy.random import Generator

from timberborn_power_mix import consts
from timberborn_power_mix.simulation.models import SimulationConfig, EnergyMixConfig
from timberborn_power_mix.rng import RNGService
from timberborn_power_mix.simulation.core import simulate_scenario
from timberborn_power_mix.simulation.helpers import calculate_total_cost
from timberborn_power_mix.machines import ProducerName, BatteryName
from timberborn_power_mix.models import ConfigName

logger = logging.getLogger(__name__)


class OptimizationResult:
    def __init__(
        self,
        config: EnergyMixConfig,
        cost: float,
        p95_hours_empty: float,
        total_hours: int,
        energy_surplus: float,
    ):
        self.config = config
        self.cost = cost
        self.p95_hours_empty = p95_hours_empty
        self.total_hours = total_hours
        self.energy_surplus = energy_surplus

    @property
    def p95_empty_percent(self) -> float:
        return (
            (self.p95_hours_empty / self.total_hours) * 100.0
            if self.total_hours > 0
            else 0.0
        )

    @property
    def is_valid(self) -> bool:
        return self.p95_empty_percent <= 5.0 and self.energy_surplus >= 0

    def calculate_score(self) -> float:
        if self.energy_surplus < 0:
            return 1e12 + abs(self.energy_surplus)
        if self.p95_empty_percent > 5.0:
            return 1e9 + self.p95_empty_percent
        return self.cost

    def __repr__(self):
        valid_str = "VALID" if self.is_valid else "INVALID"
        return (
            f"[{valid_str}] Cost: {self.cost:.1f}, "
            f"P95 Empty: {self.p95_empty_percent:.1f}%, "
            f"Surplus: {self.energy_surplus:.1f}, "
            f"Config: {self.config}"
        )


def get_random_config(
    bounds: Dict[str, Tuple[int, int]], _rng: Generator
) -> EnergyMixConfig:
    def get_val(name):
        low, high = bounds.get(name, (0, 0))
        return _rng.integers(low, high + 1)

    return EnergyMixConfig(
        **{
            ProducerName.POWER_WHEEL: get_val(ProducerName.POWER_WHEEL),
            ProducerName.WATER_WHEEL: get_val(ProducerName.WATER_WHEEL),
            ProducerName.LARGE_WINDMILL: get_val(ProducerName.LARGE_WINDMILL),
            ProducerName.WINDMILL: get_val(ProducerName.WINDMILL),
            BatteryName.BATTERY: get_val(BatteryName.BATTERY),
            BatteryName.BATTERY_HEIGHT: get_val(BatteryName.BATTERY_HEIGHT),
        }
    )


def mutate_config(
    result: OptimizationResult, bounds: Dict[str, Tuple[int, int]], _rng: Generator
) -> EnergyMixConfig:
    config = result.config
    new_config = config.model_copy(deep=True)

    if result.energy_surplus < 0:
        bias = 1
        prob_bias = 0.9
        fields = [
            ProducerName.POWER_WHEEL,
            ProducerName.WATER_WHEEL,
            ProducerName.LARGE_WINDMILL,
            ProducerName.WINDMILL,
        ]
    elif result.p95_empty_percent > 5.0:
        bias = 1
        prob_bias = 0.8
        fields = [
            ProducerName.POWER_WHEEL,
            ProducerName.WATER_WHEEL,
            ProducerName.LARGE_WINDMILL,
            ProducerName.WINDMILL,
            BatteryName.BATTERY,
            BatteryName.BATTERY_HEIGHT,
        ]
    else:
        bias = -1
        prob_bias = 0.8
        fields = [
            ProducerName.POWER_WHEEL,
            ProducerName.WATER_WHEEL,
            ProducerName.LARGE_WINDMILL,
            ProducerName.WINDMILL,
            BatteryName.BATTERY,
            BatteryName.BATTERY_HEIGHT,
        ]

    field = _rng.choice(fields)
    change = bias if _rng.random() < prob_bias else -bias

    current_val = getattr(new_config, field)
    min_val, max_val = bounds.get(field, (0, 100))
    new_val = max(min_val, min(current_val + change, max_val))
    setattr(new_config, field, new_val)

    return new_config


def evaluate_config(
    base_config: SimulationConfig,
    mix_config: EnergyMixConfig,
    samples_per_config: int,
    total_hours: int,
    rng_service: RNGService,
) -> OptimizationResult:
    current_config = base_config.model_copy(deep=True)
    current_config.energy_mix = mix_config
    current_config.samples = samples_per_config

    hours_empty_list, worst_data, avg_surplus = simulate_scenario(current_config)

    p95_empty = np.percentile(hours_empty_list, 95)
    cost = calculate_total_cost(mix_config)

    return OptimizationResult(mix_config, cost, p95_empty, total_hours, avg_surplus)


def optimize(
    base_config: SimulationConfig,
    iterations: int = consts.DEFAULT_OPTIMIZATION_ITERATIONS,
    simulations_per_config: int = consts.DEFAULT_SAMPLES,
    bounds: Optional[Dict[str, Tuple[int, int]]] = None,
) -> List[OptimizationResult]:
    if bounds is None:
        bounds = {
            ProducerName.POWER_WHEEL: (0, 20),
            ProducerName.WATER_WHEEL: (0, 20),
            ProducerName.LARGE_WINDMILL: (0, 30),
            ProducerName.WINDMILL: (0, 30),
            BatteryName.BATTERY: (0, 20),
            BatteryName.BATTERY_HEIGHT: (1, 20),
        }

    total_hours = getattr(base_config, ConfigName.DAYS) * consts.HOURS_PER_DAY
    rng_service = RNGService()
    main_rng = rng_service.get_generator()

    history = []
    best_result = None
    best_valid_result = None

    logger.info("Starting optimization (parallelized via Numba)...")

    num_walkers = 8
    current_results = [
        evaluate_config(
            base_config,
            get_random_config(bounds, main_rng),
            simulations_per_config,
            total_hours,
            rng_service,
        )
        for _ in range(num_walkers)
    ]

    for i in range(iterations):
        for w in range(num_walkers):
            candidate_config = mutate_config(current_results[w], bounds, main_rng)
            candidate_result = evaluate_config(
                base_config,
                candidate_config,
                simulations_per_config,
                total_hours,
                rng_service,
            )

            if (
                candidate_result.calculate_score()
                < current_results[w].calculate_score()
            ):
                current_results[w] = candidate_result

        round_best = min(current_results, key=lambda r: r.calculate_score())
        if (
            best_result is None
            or round_best.calculate_score() < best_result.calculate_score()
        ):
            best_result = round_best
            history.append(best_result)

            if best_result.is_valid:
                if (
                    best_valid_result is None
                    or best_result.cost < best_valid_result.cost
                ):
                    best_valid_result = best_result

        if (i + 1) % 10 == 0:
            logger.info(
                f"Iteration {i + 1}/{iterations}. "
                f"Best score: {best_result.calculate_score():.0f}. "
                f"Best valid cost: {best_valid_result.cost if best_valid_result else 'None'}"
            )

    return history


def find_optimal_solutions(
    results: List[OptimizationResult], max_empty_percent: float = 5.0
) -> List[OptimizationResult]:
    valid_solutions = [r for r in results if r.is_valid]
    valid_solutions.sort(key=lambda x: x.cost)

    unique_solutions = []
    seen_config = set()
    for sol in valid_solutions:
        p = sol.config
        key = (
            getattr(p, ProducerName.POWER_WHEEL),
            getattr(p, ProducerName.WATER_WHEEL),
            getattr(p, ProducerName.LARGE_WINDMILL),
            getattr(p, ProducerName.WINDMILL),
            getattr(p, BatteryName.BATTERY),
            getattr(p, BatteryName.BATTERY_HEIGHT),
        )
        if key not in seen_config:
            seen_config.add(key)
            unique_solutions.append(sol)

    return unique_solutions
