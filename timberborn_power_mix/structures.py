from enum import StrEnum, IntEnum
from typing import NamedTuple
import numpy as np
from numpy.typing import NDArray


class CommonConfigName(StrEnum):
    SEED = "seed"
    THREADS = "threads"
    SAMPLES = "samples"
    DAYS = "days"
    WORKING_HOURS = "working_hours"
    WET_DAYS = "wet_days"
    DRY_DAYS = "dry_days"
    BADTIDE_DAYS = "badtide_days"
    FACTORIES = "factories"


class SimulateConfigName(StrEnum):
    ENERGY_MIX = "energy_mix"


class OptimizeConfigName(StrEnum):
    MAX_TIME = "max_time"
    PERCENTILE = "percentile"
    TARGET_RELIABILITY = "target_reliability"


class Percentile(IntEnum):
    P5 = 5
    P50 = 50
    P95 = 95


class JitSimulationConfig(NamedTuple):
    """Subset of SimulationConfig used for jitted simulation configuration."""

    seed: int
    samples: int
    days: int
    working_hours: int
    wet_days: int
    dry_days: int
    badtide_days: int


class ProducerGroup(NamedTuple):
    """Combines a machine count with its individual power production rate."""

    quantity: int
    power: int


class JitSimulationCachedConsts(NamedTuple):
    """Constants for the jitted simulation that don't change between samples."""

    total_consumption_rate: int
    total_battery_capacity: int
    large_windmills: ProducerGroup
    windmills: ProducerGroup
    power_wheels: ProducerGroup
    water_wheels: ProducerGroup


class SimulationSample(NamedTuple):
    """Represents the time-series data for production and storage state from a single simulation run."""

    power_production: NDArray[np.uint32]
    battery_charge: NDArray[np.uint32]


class AggregatedSamples(NamedTuple):
    """Holds aggregated metrics and consumption profiles collected across all samples in a simulation."""

    power_consumption: NDArray[np.uint32]
    hours_empty_results: NDArray[np.uint32]


class SimulationResult(NamedTuple):
    """Final output of the simulation process, containing aggregated metrics and the p95 scenario."""

    p95_sample: SimulationSample
    aggregated_samples: AggregatedSamples


class JitSimulationPrelude(NamedTuple):
    """Static time-series profiles calculated before the stochastic simulation runs."""

    base_surplus: NDArray[np.int64]
    base_power_production: NDArray[np.uint32]
    power_consumption: NDArray[np.uint32]
    is_working_hour: NDArray[np.bool_]
    total_hours: int
