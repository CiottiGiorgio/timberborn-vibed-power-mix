from typing import NamedTuple
import numpy as np
from pydantic import create_model, BaseModel, ConfigDict

from timberborn_power_mix.machines import (
    FACTORY_DATABASE,
    BatteryName,
    PRODUCER_DATABASE,
)

FactoryParams = create_model(
    "FactoryParams", **{key: int for key in FACTORY_DATABASE.keys()}
)

EnergyMixParams = create_model(
    "EnergyMixParams",
    **{BatteryName.BATTERY: int, BatteryName.BATTERY_HEIGHT: float},
    **{key: int for key in PRODUCER_DATABASE.keys()},
)


class SimulationOptions(BaseModel):
    samples: int
    days: int
    working_hours: int
    wet_season_days: int
    dry_season_days: int
    badtide_season_days: int
    factories: FactoryParams
    energy_mix: EnergyMixParams


class SimulationResult(BaseModel):
    power_production: np.ndarray
    battery_charge: np.ndarray

    model_config = ConfigDict(arbitrary_types_allowed=True)


class ProducerGroup(NamedTuple):
    """Combines a machine count with its individual power production rate."""

    count: int
    power: int


class SimulationSample(NamedTuple):
    """Represents the time-series data for production and storage state from a single simulation run."""

    power_production: np.ndarray
    battery_charge: np.ndarray


class AggregatedSamples(NamedTuple):
    """Holds aggregated metrics and consumption profiles collected across all samples in a simulation batch."""

    hours_empty_results: np.ndarray
    final_surpluses: np.ndarray
    power_consumption: np.ndarray


class BatchedSimulationResult(NamedTuple):
    """Combines the results of a batched simulation, including the worst-performing sample and aggregated statistics."""

    worst_sample: SimulationSample
    aggregated_samples: AggregatedSamples
