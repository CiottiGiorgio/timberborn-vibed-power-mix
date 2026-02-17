import os
from multiprocessing.pool import ThreadPool

import numpy as np
from numba import njit
from timberborn_power_mix.simulation.models import (
    SimulationConfig,
    JitSimulationConfig,
    JitSimulationCachedConsts,
)
from timberborn_power_mix import consts
from timberborn_power_mix.simulation.models import (
    SimulationSample,
    AggregatedSamples,
    SimulationResult,
    ProducerGroup,
)
import timberborn_power_mix.simulation.helpers as sim_helpers


def run_simulation(config: SimulationConfig) -> SimulationResult:
    """Bridges pure Python and Numba by converting Pydantic configurations into JIT-compatible structures."""
    rng = np.random.default_rng(config.seed)

    # cpu_count = os.process_cpu_count() or 1
    cpu_count = 1
    threads = max(1, min(cpu_count, config.samples))

    jit_config = config.to_jit_config()
    cached_consts = sim_helpers.calculate_jit_cached_consts(config)

    jit_config = jit_config._replace(samples=config.samples // threads)

    with ThreadPool(processes=threads) as executor:
        results = executor.starmap(
            run_jit_simulation,
            [
                (
                    jit_config,
                    cached_consts,
                    np.random.default_rng(rng.integers(0, 2**31 - 1, size=1)),
                )
                for _ in range(threads)
            ],
        )

    worst_sample = max(results, key=lambda sample: np.max(sample.aggregated_samples.hours_empty_results), ).worst_sample
    aggregated_samples = AggregatedSamples(
        power_consumption=results[0].aggregated_samples.power_consumption,
        hours_empty_results=np.concatenate(
            [r.aggregated_samples.hours_empty_results for r in results]
        ),
    )

    return SimulationResult(
        worst_sample=worst_sample,
        aggregated_samples=aggregated_samples,
    )


@njit(nogil=True, cache=True)
def run_jit_simulation(
    config: JitSimulationConfig,
    sim_consts: JitSimulationCachedConsts,
    rng: np.random.Generator,
) -> SimulationResult:
    """
    Executes the Monte Carlo simulation and aggregates results across all samples.
    Optimizes performance by pre-calculating shared read-only arrays (like base production and consumption)
    to be reused across all stochastic runs.
    """
    total_hours = config.days * consts.HOURS_PER_DAY

    # Pre-calculate static profiles
    time_hours = np.arange(total_hours)
    hour_of_day = time_hours % consts.HOURS_PER_DAY
    is_working_hour = hour_of_day < config.working_hours

    base_power_production = sim_helpers.calculate_base_power_production(
        time_hours,
        is_working_hour,
        config.wet_days,
        config.dry_days,
        config.badtide_days,
        sim_consts.power_wheels,
        sim_consts.water_wheels,
    )

    power_consumption = np.where(
        is_working_hour, sim_consts.total_consumption_rate, 0.0
    )

    hours_empty_results = np.zeros(config.samples)
    worst_sample = SimulationSample(
        power_production=np.zeros(total_hours),
        battery_charge=np.zeros(total_hours),
    )
    max_hours_empty = -1.0

    for s in range(config.samples):
        res = jit_stochastic_simulation(
            base_power_production,
            power_consumption,
            sim_consts.large_windmills,
            sim_consts.windmills,
            sim_consts.total_battery_capacity,
            total_hours,
            rng,
        )
        hours_empty = np.sum(res.battery_charge <= 0)
        hours_empty_results[s] = hours_empty

        if hours_empty > max_hours_empty:
            max_hours_empty = hours_empty
            worst_sample = res

    aggregated_samples = AggregatedSamples(
        power_consumption=power_consumption,
        hours_empty_results=hours_empty_results,
    )

    return SimulationResult(
        worst_sample=worst_sample, aggregated_samples=aggregated_samples
    )


@njit
def jit_stochastic_simulation(
    base_power_production: np.ndarray,
    power_consumption: np.ndarray,
    large_windmills: ProducerGroup,
    windmills: ProducerGroup,
    total_battery_capacity: float,
    total_hours: int,
    rng: np.random.Generator,
) -> SimulationSample:
    """Performs a single Monte Carlo simulation run, handling stochastic input generation and internal state transitions."""
    # Generate wind data inside the core for better cache locality
    max_segments = (total_hours // consts.WIND_DURATION_MIN_HOURS) + 1
    wind_durations = rng.integers(
        consts.WIND_DURATION_MIN_HOURS,
        consts.WIND_DURATION_MAX_HOURS,
        size=max_segments,
    )
    wind_strengths = rng.random(size=max_segments)

    # Optimized Wind production using expansion
    wind_strength_profile = np.repeat(wind_strengths, wind_durations)[:total_hours]

    large_wind_unit_prod = np.where(
        wind_strength_profile > consts.LARGE_WINDMILL_THRESHOLD,
        wind_strength_profile * large_windmills.power,
        0.0,
    )
    small_wind_unit_prod = np.where(
        wind_strength_profile > consts.WINDMILL_THRESHOLD,
        wind_strength_profile * windmills.power,
        0.0,
    )
    wind_production = (large_windmills.count * large_wind_unit_prod) + (
        windmills.count * small_wind_unit_prod
    )

    power_production = base_power_production + wind_production
    power_surplus = power_production - power_consumption

    battery_charge = np.zeros(total_hours)
    current_charge = total_battery_capacity / 2.0

    for i in range(total_hours):
        potential_charge = current_charge + power_surplus[i]
        current_charge = max(0.0, min(potential_charge, total_battery_capacity))
        battery_charge[i] = current_charge

    return SimulationSample(
        power_production=power_production,
        battery_charge=battery_charge,
    )
