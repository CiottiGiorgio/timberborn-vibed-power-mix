from multiprocessing.pool import ThreadPool

import numpy as np
from numpy.typing import NDArray
from numba import njit, objmode

from timberborn_power_mix.simulation.core import jit_stochastic_simulation
from timberborn_power_mix.simulation.models import SimulationConfig
from timberborn_power_mix.structures import (
    JitSimulationConfig,
    JitSimulationCachedConsts,
    SimulationResult,
)
import timberborn_power_mix.simulation.helpers as sim_helpers
import timberborn_power_mix.helpers as helpers


def run_simulation(config: SimulationConfig) -> SimulationResult:
    jit_config = config.to_jit_config()
    cached_consts = sim_helpers.calculate_jit_cached_consts(config)

    # Generate seeds for all samples
    ss = np.random.SeedSequence(config.seed)
    all_seeds = ss.generate_state(config.samples, dtype=np.uint32)

    res: SimulationResult
    if config.threads is None or config.threads > 1:
        threads = helpers.calculate_optimal_threads(config.threads, config.samples)

        res = run_simulation_multithread(
            jit_config,
            threads,
            cached_consts,
            all_seeds,
        )
    else:
        res = run_simulation_singlethread(jit_config, cached_consts, all_seeds)

    return res


@njit(cache=True)
def run_simulation_singlethread(
    config: JitSimulationConfig,
    sim_consts: JitSimulationCachedConsts,
    all_seeds: NDArray[np.uint32],
) -> SimulationResult:
    base_power_production, power_consumption, total_hours = (
        sim_helpers.jit_simulation_prelude(config, sim_consts)
    )

    all_hours_empty = jit_batched_simulation(
        all_seeds, total_hours, base_power_production, power_consumption, sim_consts
    )

    return sim_helpers.jit_simulation_conclusion(
        all_hours_empty,
        all_seeds,
        base_power_production,
        power_consumption,
        sim_consts,
        total_hours,
    )


@njit(cache=True)
def run_simulation_multithread(
    config: JitSimulationConfig,
    threads: int,
    sim_consts: JitSimulationCachedConsts,
    all_seeds: NDArray[np.uint32],
) -> SimulationResult:
    base_power_production, power_consumption, total_hours = (
        sim_helpers.jit_simulation_prelude(config, sim_consts)
    )

    with objmode(all_hours_empty="uint32[:]"):
        pool = ThreadPool(processes=threads)
        seed_chunks = np.array_split(all_seeds, threads)

        try:
            results = pool.starmap(
                jit_batched_simulation,
                [
                    (
                        seeds,
                        total_hours,
                        base_power_production,
                        power_consumption,
                        sim_consts,
                    )
                    for seeds in seed_chunks
                ],
            )
            all_hours_empty = np.concatenate(results)
        finally:
            pool.close()
            pool.join()

    return sim_helpers.jit_simulation_conclusion(
        all_hours_empty,
        all_seeds,
        base_power_production,
        power_consumption,
        sim_consts,
        total_hours,
    )


@njit(nogil=True)
def jit_batched_simulation(
    seeds: NDArray[np.uint32],
    total_hours: int,
    base_power_production: NDArray[np.uint32],
    power_consumption: NDArray[np.uint32],
    sim_consts: JitSimulationCachedConsts,
) -> NDArray[np.uint32]:
    """
    Executes the Monte Carlo simulation and aggregates metrics.
    Does NOT store full time-series for every sample to save memory.
    """
    n_samples = len(seeds)
    hours_empty_results = np.zeros(n_samples, dtype=np.uint32)

    for s in range(n_samples):
        res = jit_stochastic_simulation(
            seeds[s],
            total_hours,
            base_power_production,
            power_consumption,
            sim_consts.total_battery_capacity,
            sim_consts.large_windmills,
            sim_consts.windmills,
        )
        hours_empty_results[s] = np.sum(res.battery_charge <= 0)

    return hours_empty_results
