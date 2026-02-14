from typing import NamedTuple
import numpy as np
from pydantic import create_model, BaseModel, ConfigDict

from timberborn_power_mix.machines import (
    FACTORY_DATABASE,
    BatteryName,
    PRODUCER_DATABASE,
)
from timberborn_power_mix.models import ConfigName

FactoryConfig = create_model(
    "FactoryConfig", **{key: int for key in FACTORY_DATABASE.keys()}
)

EnergyMixConfig = create_model(
    "EnergyMixConfig",
    **{BatteryName.BATTERY: int, BatteryName.BATTERY_HEIGHT: float},
    **{key: int for key in PRODUCER_DATABASE.keys()},
)


class BatchConfig(NamedTuple):
    """Subset of SimulationConfig used for batch simulation configuration."""

    samples: int
    days: int
    working_hours: int
    wet_days: int
    dry_days: int
    badtide_days: int


SimulationConfig = create_model(
    "SimulationConfig",
    **{ConfigName.SAMPLES: int},
    **{ConfigName.DAYS: int},
    **{ConfigName.WORKING_HOURS: int},
    **{ConfigName.WET_DAYS: int},
    **{ConfigName.DRY_DAYS: int},
    **{ConfigName.BADTIDE_DAYS: int},
    **{ConfigName.FACTORIES: FactoryConfig},
    **{ConfigName.ENERGY_MIX: EnergyMixConfig},
    __base__=BaseModel,
)


def get_batch_config(config: SimulationConfig) -> BatchConfig:
    return BatchConfig(
        samples=getattr(config, ConfigName.SAMPLES),
        days=getattr(config, ConfigName.DAYS),
        working_hours=getattr(config, ConfigName.WORKING_HOURS),
        wet_days=getattr(config, ConfigName.WET_DAYS),
        dry_days=getattr(config, ConfigName.DRY_DAYS),
        badtide_days=getattr(config, ConfigName.BADTIDE_DAYS),
    )


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
