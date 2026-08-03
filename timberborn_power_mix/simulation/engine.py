import warnings
from concurrent.futures.thread import ThreadPoolExecutor
from itertools import repeat

import numpy as np
from numba import njit, objmode
from numba.core.errors import NumbaWarning
from numpy.typing import NDArray

import timberborn_power_mix.simulation.helpers as sim_helpers
from timberborn_power_mix.simulation.core import jit_stochastic_simulation_no_sample
from timberborn_power_mix.structures import (
    JitSimulationCachedConsts,
    JitSimulationConfig,
    SimulationResult,
)

# Suppress NumbaWarning about objmode usage in nogil functions
warnings.filterwarnings(
    "ignore",
    category=NumbaWarning,
    message=".*Code running in object mode won't allow parallel execution.*",
)


@njit(cache=True)
def jit_singlethread_simulation(
    config: JitSimulationConfig,
    sim_consts: JitSimulationCachedConsts,
) -> SimulationResult:
    """
    Runs a full batched simulation on a single thread.

    This function handles the entire simulation pipeline: prelude (static profiles),
    batched stochastic runs, and epilogue (aggregation and P95 selection).
    """
    with objmode(all_seeds="uint32[:]"):
        ss = np.random.SeedSequence(config.seed)
        all_seeds = ss.generate_state(config.samples, dtype=np.uint32)

    prelude = sim_helpers.jit_simulation_prelude(config, sim_consts)

    all_lost_hours = jit_batched_simulation(
        all_seeds,
        prelude.total_hours,
        prelude.base_surplus,
        prelude.power_consumption,
        prelude.is_working_hour,
        sim_consts,
    )

    return sim_helpers.jit_simulation_epilogue(
        all_lost_hours,
        all_seeds,
        prelude.base_surplus,
        prelude.base_power_production,
        prelude.power_consumption,
        sim_consts,
        prelude.total_hours,
    )


@njit(nogil=True)
def jit_singlethread_simulation_no_plots(
    config: JitSimulationConfig,
    sim_consts: JitSimulationCachedConsts,
    percentile: int,
) -> np.float64:
    """
    Runs a batched simulation and returns only the unreliability metric for a given percentile.

    Optimized for use in the optimization engine where full results and
    time-series data are not required.
    """
    with objmode(all_seeds="uint32[:]"):
        ss = np.random.SeedSequence(config.seed)
        all_seeds = ss.generate_state(config.samples, dtype=np.uint32)

    prelude = sim_helpers.jit_simulation_prelude(config, sim_consts)

    all_lost_hours = jit_batched_simulation(
        all_seeds,
        prelude.total_hours,
        prelude.base_surplus,
        prelude.power_consumption,
        prelude.is_working_hour,
        sim_consts,
    )

    return np.percentile(all_lost_hours, percentile)


@njit(cache=True)
def jit_multithread_simulation(
    config: JitSimulationConfig,
    threads: int,
    sim_consts: JitSimulationCachedConsts,
) -> SimulationResult:
    """
    Runs a full batched simulation using multiple threads.

    Distributes the stochastic samples across a thread pool. Each thread
    executes a batch of simulations independently.
    """
    prelude = sim_helpers.jit_simulation_prelude(config, sim_consts)

    with objmode(all_seeds="uint32[:]", all_lost_hours="float64[:]"):
        executor = ThreadPoolExecutor(max_workers=threads)

        # Generate seeds for all samples
        ss = np.random.SeedSequence(config.seed)
        all_seeds = ss.generate_state(config.samples, dtype=np.uint32)
        seed_chunks = np.array_split(all_seeds, threads)

        try:
            results = executor.map(
                jit_batched_simulation,
                seed_chunks,
                repeat(prelude.total_hours),
                repeat(prelude.base_surplus),
                repeat(prelude.power_consumption),
                repeat(prelude.is_working_hour),
                repeat(sim_consts),
            )
            all_lost_hours = np.concatenate([r for r in results])
        finally:
            executor.shutdown()

    return sim_helpers.jit_simulation_epilogue(
        all_lost_hours,
        all_seeds,
        prelude.base_surplus,
        prelude.base_power_production,
        prelude.power_consumption,
        sim_consts,
        prelude.total_hours,
    )


@njit(nogil=True)
def jit_batched_simulation(
    seeds: NDArray[np.uint32],
    total_hours: int,
    base_surplus: NDArray[np.int64],
    power_consumption: NDArray[np.uint32],
    is_working_hour: NDArray[np.bool_],
    sim_consts: JitSimulationCachedConsts,
) -> NDArray[np.float64]:
    """
    Executes a batch of Monte Carlo simulations and aggregates metrics.

    Does NOT store full time-series for every sample to save memory.
    Returns an array containing the number of lost working hours for each seed.
    """
    n_samples = len(seeds)
    lost_hours_results = np.zeros(n_samples, dtype=np.float64)

    for s in range(n_samples):
        lost_hours_results[s] = jit_stochastic_simulation_no_sample(
            seeds[s],
            total_hours,
            base_surplus,
            power_consumption,
            is_working_hour,
            sim_consts.total_battery_capacity,
            sim_consts.large_windmills,
            sim_consts.windmills,
        )

    return lost_hours_results
