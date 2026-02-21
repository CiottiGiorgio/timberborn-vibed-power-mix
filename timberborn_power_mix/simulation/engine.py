from concurrent.futures.thread import ThreadPoolExecutor
from itertools import repeat

import numpy as np
from numpy.typing import NDArray
from numba import njit, objmode

from timberborn_power_mix.simulation.core import jit_stochastic_simulation_no_sample
from timberborn_power_mix.structures import (
    JitSimulationConfig,
    JitSimulationCachedConsts,
    SimulationResult,
)
import timberborn_power_mix.simulation.helpers as sim_helpers


@njit(cache=True)
def jit_singlethread_simulation(
    config: JitSimulationConfig,
    sim_consts: JitSimulationCachedConsts,
) -> SimulationResult:
    with objmode(all_seeds="uint32[:]"):
        ss = np.random.SeedSequence(config.seed)
        all_seeds = ss.generate_state(config.samples, dtype=np.uint32)

    base_surplus, base_power_production, power_consumption, total_hours = (
        sim_helpers.jit_simulation_prelude(config, sim_consts)
    )

    all_hours_empty = jit_batched_simulation(
        all_seeds, total_hours, base_surplus, sim_consts
    )

    return sim_helpers.jit_simulation_epilogue(
        all_hours_empty,
        all_seeds,
        base_surplus,
        base_power_production,
        power_consumption,
        sim_consts,
        total_hours,
    )


@njit(nogil=True)
def jit_singlethread_simulation_no_plots(
    config: JitSimulationConfig, sim_consts: JitSimulationCachedConsts
) -> np.uint32:
    with objmode(all_seeds="uint32[:]"):
        ss = np.random.SeedSequence(config.seed)
        all_seeds = ss.generate_state(config.samples, dtype=np.uint32)

    base_surplus, _, _, total_hours = sim_helpers.jit_simulation_prelude(
        config, sim_consts
    )

    all_hours_empty = jit_batched_simulation(
        all_seeds, total_hours, base_surplus, sim_consts
    )

    return np.percentile(all_hours_empty, 95)


@njit(cache=True)
def jit_multithread_simulation(
    config: JitSimulationConfig,
    threads: int,
    sim_consts: JitSimulationCachedConsts,
) -> SimulationResult:
    base_surplus, base_power_production, power_consumption, total_hours = (
        sim_helpers.jit_simulation_prelude(config, sim_consts)
    )

    with objmode(all_seeds="uint32[:]", all_hours_empty="uint32[:]"):
        executor = ThreadPoolExecutor(max_workers=threads)

        # Generate seeds for all samples
        ss = np.random.SeedSequence(config.seed)
        all_seeds = ss.generate_state(config.samples, dtype=np.uint32)
        seed_chunks = np.array_split(all_seeds, threads)

        try:
            results = executor.map(
                jit_batched_simulation,
                seed_chunks,
                repeat(total_hours),
                repeat(base_surplus),
                repeat(sim_consts),
            )
            all_hours_empty = np.concatenate([r for r in results])
        finally:
            executor.shutdown()

    return sim_helpers.jit_simulation_epilogue(
        all_hours_empty,
        all_seeds,
        base_surplus,
        base_power_production,
        power_consumption,
        sim_consts,
        total_hours,
    )


@njit(nogil=True)
def jit_batched_simulation(
    seeds: NDArray[np.uint32],
    total_hours: int,
    base_surplus: NDArray[np.int64],
    sim_consts: JitSimulationCachedConsts,
) -> NDArray[np.uint32]:
    """
    Executes the Monte Carlo simulation and aggregates metrics.
    Does NOT store full time-series for every sample to save memory.
    """
    n_samples = len(seeds)
    hours_empty_results = np.zeros(n_samples, dtype=np.uint32)

    for s in range(n_samples):
        hours_empty_results[s] = jit_stochastic_simulation_no_sample(
            seeds[s],
            total_hours,
            base_surplus,
            sim_consts.total_battery_capacity,
            sim_consts.large_windmills,
            sim_consts.windmills,
        )

    return hours_empty_results
