from enum import StrEnum
from typing import NamedTuple, Optional
import numpy as np
from numpy.typing import NDArray


class ConfigName(StrEnum):
    SEED = "seed"
    THREADS = "threads"
    SAMPLES = "samples"
    DAYS = "days"
    WORKING_HOURS = "working_hours"
    WET_DAYS = "wet_days"
    DRY_DAYS = "dry_days"
    BADTIDE_DAYS = "badtide_days"
    FACTORIES = "factories"
    ENERGY_MIX = "energy_mix"
    ITERATIONS = "iterations"


class JitSimulationConfig(NamedTuple):
    """Subset of SimulationConfig used for jitted simulation configuration."""

    threads: int
    samples: int
    days: int
    working_hours: int
    wet_days: int
    dry_days: int
    badtide_days: int
    seed: Optional[int] = None


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

    power_consumption: NDArray[np.int32]
    hours_empty_results: NDArray[np.uint32]


class SimulationResult(NamedTuple):
    """Final output of the simulation process, containing aggregated metrics and the p95 scenario."""

    p95_sample: SimulationSample
    aggregated_samples: AggregatedSamples
