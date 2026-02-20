from multiprocessing.pool import ThreadPool

import numpy as np
from numpy.typing import NDArray
from numba import njit, objmode

from timberborn_power_mix.simulation.core import jit_stochastic_simulation
from timberborn_power_mix.structures import (
    JitSimulationConfig,
    JitSimulationCachedConsts,
    SimulationResult,
)
import timberborn_power_mix.simulation.helpers as sim_helpers


@njit(cache=True)
def run_simulation_singlethread(
    config: JitSimulationConfig,
    sim_consts: JitSimulationCachedConsts,
    all_seeds: NDArray[np.uint64],
) -> SimulationResult:
    base_power_production, power_consumption, total_hours = (
        sim_helpers.jit_simulation_prelude(config, sim_consts)
    )

    all_hours_empty = jit_batched_simulation(
        base_power_production,
        power_consumption,
        total_hours,
        sim_consts,
        all_seeds,
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
    sim_consts: JitSimulationCachedConsts,
    all_seeds: NDArray[np.uint64],
) -> SimulationResult:
    base_power_production, power_consumption, total_hours = (
        sim_helpers.jit_simulation_prelude(config, sim_consts)
    )

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
    base_power_production: NDArray[np.uint32],
    power_consumption: NDArray[np.uint32],
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
