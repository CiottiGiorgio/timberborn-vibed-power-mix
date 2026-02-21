# Simulation Architecture: Reproducible Parallel Monte Carlo

This document explains the architectural choices made in the `timberborn-power-mix` simulation engine. The engine is designed to run 100,000+ stochastic samples of power production and consumption while maintaining strict reproducibility and high performance.

## Core Constraints

The architecture is driven by four primary technical constraints:
1.  **Reproducibility**: Any specific sample (e.g., the 95th percentile "worst case") must be perfectly replayable to generate detailed time-series data for plotting.
2.  **Memory Efficiency**: Storing full time-series data for every sample is prohibitive. We must only store aggregate results for the bulk of the simulation.
3.  **Numba Compatibility**: The core simulation must run in Numba's `nopython` mode to achieve C-like speeds.
4.  **Thread Safety**: Legacy NumPy randomness (`np.random.seed`) is not thread-safe in a way that guarantees determinism across parallel execution.

---

## Architectural Decisions

### 1. Explicit Seed Generation (`SeedSequence` & `objmode`)
To ensure every sample is reproducible, we don't rely on a global random state. Instead, we generate an array of unique seeds upfront.

We use `np.random.SeedSequence` because it is the modern standard for generating high-quality, independent seeds for parallel processes. Since `SeedSequence` is a complex Python object not supported by Numba, we wrap its usage in an `objmode` block:

```python
# In engine.py
with objmode(all_seeds="uint32[:]"):
    ss = np.random.SeedSequence(config.seed)
    all_seeds = ss.generate_state(config.samples, dtype=np.uint32)
```

This "Lookup Table of Reality" allows us to:
*   Distribute specific seeds to parallel workers.
*   Re-run the exact same simulation for the P95 sample later by just passing its specific seed.

### 2. Python `ThreadPoolExecutor` and the GIL
While Numba offers `parallel=True` with `prange`, we deliberately chose Python's `ThreadPoolExecutor` for **Coarse-Grained Parallelism**.

**Why not `prange`?**
Legacy NumPy randomness in Numba uses a global internal state. If multiple threads call `np.random.random()` inside a `prange` loop, the order in which they "grab" the next state is non-deterministic. This would break reproducibility even if the starting seed was the same.

**The Solution:**
We split the seeds into large "chunks" and dispatch them to a `ThreadPoolExecutor`. To ensure this actually runs in parallel, the worker functions are decorated with `nogil=True`.

```python
@njit(nogil=True)
def jit_batched_simulation(...):
    # ...
```

Since the simulation is entirely compute-heavy with no I/O waits, the Python Global Interpreter Lock (GIL) would normally prevent any true concurrency. By using `nogil=True`, we allow Numba to release the GIL, enabling multiple threads to execute the simulation logic simultaneously on different CPU cores.

### 3. Caching and Entry Points
The "front-facing" functions in `engine.py` are decorated with `cache=True`. This ensures that the compilation overhead is only paid once, and subsequent runs of the simulation start instantly.

```python
@njit(cache=True)
def jit_multithread_simulation(...):
    # ...
```

### 4. Two-Pass Execution (Metrics vs. Full)
To solve the memory constraint, the simulation is split into two variants:

1.  **Pass 1 (The Batch)**: `jit_stochastic_simulation_no_sample`
    *   Calculates only the "Hours Empty" metric.
    *   Allocates no large arrays.
    *   Used for the bulk of the samples in the parallel loop.
2.  **Pass 2 (The Replay)**: `jit_stochastic_simulation`
    *   Calculates full time-series for `power_production` and `battery_charge`.
    *   Allocates arrays for the result.
    *   Used **only once** for the P95 sample identified in Pass 1.

---

## Summary of Data Flow

1.  **Python**: User provides a single master `seed`.
2.  **Numba (objmode)**: `SeedSequence` generates the required number of `uint32` seeds.
3.  **Python**: `ThreadPoolExecutor` splits seeds into chunks and sends them to workers.
4.  **Numba (Batch)**: Workers run the lean simulation with `nogil=True`, returning only the "Hours Empty" count for each seed.
5.  **Numba (Epilogue)**: `np.percentile` identifies the P95 "Hours Empty" value and its corresponding seed.
6.  **Numba (Full)**: The P95 seed is used to run the full simulation once, generating the detailed data for the UI.
