from multiprocessing.pool import ThreadPool

import numpy as np
from numpy.typing import NDArray
from numba import njit, objmode
from timberborn_power_mix.simulation import consts
from timberborn_power_mix.structures import (
    JitSimulationConfig,
    JitSimulationCachedConsts,
    SimulationSample,
    ProducerGroup,
    AggregatedSamples,
    SimulationResult,
)
import timberborn_power_mix.simulation.helpers as sim_helpers


@njit(cache=True)
def run_simulation_singlethread(
    config: JitSimulationConfig,
    sim_consts: JitSimulationCachedConsts,
    all_seeds: NDArray[np.uint64],
) -> SimulationResult:
    total_hours = config.days * consts.HOURS_PER_DAY

    # Pre-calculate static profiles
    time_hours = np.arange(total_hours, dtype=np.int32)
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
        is_working_hour, sim_consts.total_consumption_rate, 0
    ).astype(np.int32)

    all_hours_empty = jit_batched_simulation(
        base_power_production,
        power_consumption,
        total_hours,
        sim_consts,
        all_seeds,
    )

    aggregated = AggregatedSamples(
        power_consumption=power_consumption,
        hours_empty_results=all_hours_empty,
    )

    # Find p95 sample (Second Pass)
    p95_hours_empty = np.percentile(all_hours_empty, 95)
    p95_idx = np.where(all_hours_empty >= p95_hours_empty)[0][0]
    p95_seed = all_seeds[p95_idx]

    p95_sample = jit_stochastic_simulation(
        base_power_production,
        power_consumption,
        sim_consts.large_windmills,
        sim_consts.windmills,
        sim_consts.total_battery_capacity,
        total_hours,
        p95_seed,
    )

    return SimulationResult(
        p95_sample=p95_sample,
        aggregated_samples=aggregated,
    )


@njit(cache=True)
def run_simulation_multithread(
    config: JitSimulationConfig,
    sim_consts: JitSimulationCachedConsts,
    all_seeds: NDArray[np.uint64],
) -> SimulationResult:
    total_hours = config.days * consts.HOURS_PER_DAY

    # Pre-calculate static profiles
    time_hours = np.arange(total_hours, dtype=np.int32)
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
        is_working_hour, sim_consts.total_consumption_rate, 0
    ).astype(np.int32)

    with objmode(all_hours_empty="uint32[:]"):
        pool = ThreadPool(processes=config.threads)
        seed_chunks = np.array_split(all_seeds, config.threads)

        try:
            results = pool.starmap(
                jit_batched_simulation,
                [
                    (
                        base_power_production,
                        power_consumption,
                        total_hours,
                        sim_consts,
                        seeds,
                    )
                    for seeds in seed_chunks
                ],
            )
            all_hours_empty = np.concatenate(results)
        finally:
            pool.close()
            pool.join()

    aggregated = AggregatedSamples(
        power_consumption=power_consumption,
        hours_empty_results=all_hours_empty,
    )

    # Find p95 sample (Second Pass)
    p95_hours_empty = np.percentile(all_hours_empty, 95)
    p95_idx = np.where(all_hours_empty >= p95_hours_empty)[0][0]
    p95_seed = all_seeds[p95_idx]

    p95_sample = jit_stochastic_simulation(
        base_power_production,
        power_consumption,
        sim_consts.large_windmills,
        sim_consts.windmills,
        sim_consts.total_battery_capacity,
        total_hours,
        p95_seed,
    )

    return SimulationResult(
        p95_sample=p95_sample,
        aggregated_samples=aggregated,
    )


@njit(nogil=True)
def jit_batched_simulation(
    base_power_production: NDArray[np.int32],
    power_consumption: NDArray[np.int32],
    total_hours: int,
    sim_consts: JitSimulationCachedConsts,
    seeds: NDArray[np.uint64],
) -> NDArray[np.uint32]:
    """
    Executes the Monte Carlo simulation and aggregates metrics.
    Does NOT store full time-series for every sample to save memory.
    """
    n_samples = len(seeds)
    hours_empty_results = np.zeros(n_samples, dtype=np.uint32)

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

    return hours_empty_results


@njit
def jit_stochastic_simulation(
    base_power_production: NDArray[np.int32],
    power_consumption: NDArray[np.int32],
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
    ).astype(np.int32)
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

    battery_charge = np.zeros(total_hours, dtype=np.float64)
    current_charge = total_battery_capacity / 2.0

    for i in range(total_hours):
        potential_charge = current_charge + power_surplus[i]
        current_charge = max(0.0, min(potential_charge, total_battery_capacity))
        battery_charge[i] = current_charge

    return SimulationSample(
        power_production=power_production,
        battery_charge=battery_charge,
    )
