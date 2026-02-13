import logging
from typing import List, Dict, Tuple, Optional

import numpy as np
from numpy.random import Generator

from timberborn_power_mix import consts
from timberborn_power_mix.models import SimulationOptions, EnergyMixParams
from timberborn_power_mix.rng import RNGService
from timberborn_power_mix.simulation import simulate_scenario

logger = logging.getLogger(__name__)


class OptimizationResult:
    def __init__(
        self,
        params: EnergyMixParams,
        cost: float,
        p95_hours_empty: float,
        total_hours: int,
        energy_surplus: float,
    ):
        self.params = params
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
            f"Params: {self.params}"
        )


def get_random_params(
    bounds: Dict[str, Tuple[int, int]], _rng: Generator
) -> EnergyMixParams:
    def get_val(name):
        low, high = bounds.get(name, (0, 0))
        return _rng.integers(low, high + 1)

    return EnergyMixParams(
        power_wheel=get_val("power_wheel"),
        water_wheel=get_val("water_wheel"),
        large_windmill=get_val("large_windmills"),
        windmill=get_val("windmills"),
        battery=get_val("batteries"),
        battery_height=get_val("battery_height"),
    )


def mutate_params(
    result: OptimizationResult, bounds: Dict[str, Tuple[int, int]], _rng: Generator
) -> EnergyMixParams:
    params = result.params
    new_params = params.model_copy(deep=True)

    if result.energy_surplus < 0:
        bias = 1
        prob_bias = 0.9
        fields = ["power_wheel", "water_wheel", "large_windmills", "windmills"]
    elif result.p95_empty_percent > 5.0:
        bias = 1
        prob_bias = 0.8
        fields = [
            "power_wheel",
            "water_wheel",
            "large_windmills",
            "windmills",
            "batteries",
            "battery_height",
        ]
    else:
        bias = -1
        prob_bias = 0.8
        fields = [
            "power_wheel",
            "water_wheel",
            "large_windmills",
            "windmills",
            "batteries",
            "battery_height",
        ]

    field = _rng.choice(fields)
    change = bias if _rng.random() < prob_bias else -bias

    current_val = getattr(new_params, field)
    min_val, max_val = bounds.get(field, (0, 100))
    new_val = max(min_val, min(current_val + change, max_val))
    setattr(new_params, field, new_val)

    return new_params


def evaluate_config(
    base_params: SimulationOptions,
    mix_params: EnergyMixParams,
    samples_per_config: int,
    total_hours: int,
    rng_service: RNGService,
) -> OptimizationResult:
    current_params = base_params.model_copy(deep=True)
    current_params.energy_mix = mix_params
    current_params.samples = samples_per_config

    hours_empty_list, worst_data, avg_surplus = simulate_scenario(
        current_params, rng_service=rng_service
    )

    p95_empty = np.percentile(hours_empty_list, 95)
    cost = worst_data.total_cost

    return OptimizationResult(mix_params, cost, p95_empty, total_hours, avg_surplus)


def optimize(
    base_params: SimulationOptions,
    iterations: int = consts.DEFAULT_OPTIMIZATION_ITERATIONS,
    simulations_per_config: int = consts.DEFAULT_SAMPLES,
    bounds: Optional[Dict[str, Tuple[int, int]]] = None,
) -> List[OptimizationResult]:
    if bounds is None:
        bounds = {
            "power_wheel": (0, 20),
            "water_wheel": (0, 20),
            "large_windmills": (0, 30),
            "windmills": (0, 30),
            "batteries": (0, 20),
            "battery_height": (1, 20),
        }

    total_hours = base_params.days * consts.HOURS_PER_DAY
    rng_service = RNGService()
    main_rng = rng_service.get_generator()

    history = []
    best_result = None
    best_valid_result = None

    logger.info("Starting optimization (parallelized via Numba)...")

    num_walkers = 8
    current_results = [
        evaluate_config(
            base_params,
            get_random_params(bounds, main_rng),
            simulations_per_config,
            total_hours,
            rng_service,
        )
        for _ in range(num_walkers)
    ]

    for i in range(iterations):
        for w in range(num_walkers):
            candidate_params = mutate_params(current_results[w], bounds, main_rng)
            candidate_result = evaluate_config(
                base_params,
                candidate_params,
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
    seen_params = set()
    for sol in valid_solutions:
        p = sol.params
        key = (
            p.power_wheel,
            p.water_wheel,
            p.large_windmill,
            p.windmill,
            p.battery,
            p.battery_height,
        )
        if key not in seen_params:
            seen_params.add(key)
            unique_solutions.append(sol)

    return unique_solutions
