from multiprocessing.pool import ThreadPool
from typing import List

import numpy as np
from numba import njit
from timberborn_power_mix.simulation.models import (
    SimulationConfig,
)
from timberborn_power_mix import helpers
from timberborn_power_mix.simulation import consts
from timberborn_power_mix.structures import (
    JitSimulationConfig,
    JitSimulationCachedConsts,
    SimulationSample,
    AggregatedSamples,
    SimulationResult,
    ProducerGroup,
)
import timberborn_power_mix.simulation.helpers as sim_helpers


def run_simulation_singlethread(
    config: SimulationConfig,
) -> SimulationResult:
    """Executes the simulation in a single thread."""
    jit_config = config.to_jit_config()
    cached_consts = sim_helpers.calculate_jit_cached_consts(config)

    # Generate seeds for all samples
    ss = np.random.SeedSequence(config.seed)
    seeds = ss.generate_state(config.samples, dtype=np.uint64)

    aggregated = run_jit_simulation(jit_config, cached_consts, seeds)

    # Find p95 sample
    p95_sample = _reconstruct_p95_sample(config, cached_consts, aggregated, seeds)

    return SimulationResult(
        p95_sample=p95_sample,
        aggregated_samples=aggregated,
    )


def run_simulation_multithread(
    config: SimulationConfig,
) -> SimulationResult:
    """Executes the simulation across multiple threads using a ThreadPool."""
    threads = helpers.calculate_optimal_threads(config.threads, config.samples)

    jit_config = config.to_jit_config()
    cached_consts = sim_helpers.calculate_jit_cached_consts(config)

    # Generate all seeds upfront to ensure we can reconstruct any sample
    ss = np.random.SeedSequence(config.seed)
    all_seeds = ss.generate_state(config.samples, dtype=np.uint64)

    # Distribute seeds across threads
    seed_chunks = np.array_split(all_seeds, threads)

    thread_args = [(jit_config, cached_consts, chunk) for chunk in seed_chunks]

    with ThreadPool(processes=threads) as executor:
        results: List[AggregatedSamples] = executor.starmap(
            run_jit_simulation, thread_args
        )

    # Aggregate results
    all_hours_empty = np.concatenate([r.hours_empty_results for r in results])

    aggregated = AggregatedSamples(
        power_consumption=results[0].power_consumption,
        hours_empty_results=all_hours_empty,
    )

    # Find p95 sample (Second Pass)
    p95_sample = _reconstruct_p95_sample(config, cached_consts, aggregated, all_seeds)

    return SimulationResult(
        p95_sample=p95_sample,
        aggregated_samples=aggregated,
    )


def _reconstruct_p95_sample(
    config: SimulationConfig,
    sim_consts: JitSimulationCachedConsts,
    aggregated: AggregatedSamples,
    seeds: np.ndarray,
) -> SimulationSample:
    """Finds the p95 seed and re-runs that specific simulation to get full data."""
    total_hours = config.days * consts.HOURS_PER_DAY

    # Get indices that would sort the results
    sorted_indices = np.argsort(aggregated.hours_empty_results)

    # Pick the index at the 95th percentile position
    # For N samples, this is the element at floor(0.95 * (N - 1))
    p95_pos = int(np.floor(0.95 * (len(sorted_indices) - 1)))
    idx = sorted_indices[p95_pos]
    target_seed = seeds[idx]

    # Re-calculate base profiles (same as in run_jit_simulation)
    time_hours = np.arange(total_hours)
    is_working_hour = (time_hours % consts.HOURS_PER_DAY) < config.working_hours

    base_power_production = sim_helpers.calculate_base_power_production(
        time_hours,
        is_working_hour,
        config.wet_days,
        config.dry_days,
        config.badtide_days,
        sim_consts.power_wheels,
        sim_consts.water_wheels,
    )
    power_consumption = np.where(is_working_hour, sim_consts.total_consumption_rate, 0)

    return jit_stochastic_simulation(
        base_power_production,
        power_consumption,
        sim_consts.large_windmills,
        sim_consts.windmills,
        sim_consts.total_battery_capacity,
        total_hours,
        target_seed,
    )


@njit(nogil=True, cache=True)
def run_jit_simulation(
    config: JitSimulationConfig,
    sim_consts: JitSimulationCachedConsts,
    seeds: np.ndarray,
) -> AggregatedSamples:
    """
    Executes the Monte Carlo simulation and aggregates metrics.
    Does NOT store full time-series for every sample to save memory.
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

    power_consumption = np.where(is_working_hour, sim_consts.total_consumption_rate, 0)

    n_samples = len(seeds)
    hours_empty_results = np.zeros(n_samples)

    for s in range(n_samples):
        res = jit_stochastic_simulation(
            base_power_production,
            power_consumption,
            sim_consts.large_windmills,
            sim_consts.windmills,
            sim_consts.total_battery_capacity,
            total_hours,
            seeds[s],
        )
        hours_empty_results[s] = np.sum(res.battery_charge <= 0)

    return AggregatedSamples(
        power_consumption=power_consumption,
        hours_empty_results=hours_empty_results,
    )


@njit
def jit_stochastic_simulation(
    base_power_production: np.ndarray,
    power_consumption: np.ndarray,
    large_windmills: ProducerGroup,
    windmills: ProducerGroup,
    total_battery_capacity: float,
    total_hours: int,
    seed: int,
) -> SimulationSample:
    """Performs a single Monte Carlo simulation run with a specific seed."""
    np.random.seed(seed)

    # Generate wind data
    max_segments = (total_hours // consts.WIND_DURATION_MIN_HOURS) + 1
    wind_durations = np.random.randint(
        consts.WIND_DURATION_MIN_HOURS,
        consts.WIND_DURATION_MAX_HOURS,
        size=max_segments,
    )
    wind_strengths = np.random.random(size=max_segments)

    # Optimized Wind production using expansion
    wind_strength_profile = np.repeat(wind_strengths, wind_durations)[:total_hours]

    large_wind_unit_prod = np.where(
        wind_strength_profile > consts.LARGE_WINDMILL_THRESHOLD,
        wind_strength_profile * large_windmills.power,
        0,
    )
    small_wind_unit_prod = np.where(
        wind_strength_profile > consts.WINDMILL_THRESHOLD,
        wind_strength_profile * windmills.power,
        0,
    )
    wind_production = (large_windmills.quantity * large_wind_unit_prod) + (
        windmills.quantity * small_wind_unit_prod
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
