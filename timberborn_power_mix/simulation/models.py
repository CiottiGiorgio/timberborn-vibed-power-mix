from typing import NamedTuple
import numpy as np
from pydantic import create_model, BaseModel

from timberborn_power_mix.machines import (
    FACTORY_DATABASE,
    BatteryName,
    PRODUCER_DATABASE,
)
from timberborn_power_mix.models import ConfigName

"""
This module defines the configuration and result models for the power simulation.
Many models are created dynamically using Pydantic's `create_model` to stay in sync 
with the machine databases defined in `machines.py`.

The dynamic structures effectively look like this:

class FactoryConfig(BaseModel):
    lumber_mill: int = 0
    gear_workshop: int = 0
    ... (all other factories)

class EnergyMixConfig(BaseModel):
    battery: int = 0
    battery_height: float = 0.0
    windmill: int = 0
    water_wheel: int = 0
    ... (all other producers)

class SimulationConfig(BaseModel):
    samples: int
    days: int
    working_hours: int
    wet_days: int
    dry_days: int
    badtide_days: int
    factories: FactoryConfig
    energy_mix: EnergyMixConfig

class OptimizationConfig(BaseModel):
    iterations: int
    samples: int
    days: int
    working_hours: int
    wet_days: int
    dry_days: int
    badtide_days: int
    factories: FactoryConfig
"""

FactoryConfig = create_model(
    "FactoryConfig", **{key: int for key in FACTORY_DATABASE.keys()}
)

EnergyMixConfig = create_model(
    "EnergyMixConfig",
    **{BatteryName.BATTERY: int, BatteryName.BATTERY_HEIGHT: float},
    **{key: int for key in PRODUCER_DATABASE.keys()},
)


class ParallelConfig(NamedTuple):
    """Subset of SimulationConfig used for parallel simulation configuration."""

    samples: int
    days: int
    working_hours: int
    wet_days: int
    dry_days: int
    badtide_days: int


CommonConfig = create_model(
    "CommonConfig",
    **{ConfigName.SAMPLES: int},
    **{ConfigName.DAYS: int},
    **{ConfigName.WORKING_HOURS: int},
    **{ConfigName.WET_DAYS: int},
    **{ConfigName.DRY_DAYS: int},
    **{ConfigName.BADTIDE_DAYS: int},
    **{ConfigName.FACTORIES: FactoryConfig},
)


class SimulationConfigBase(BaseModel):
    @property
    def to_parallel_config(self) -> ParallelConfig:
        return ParallelConfig(
            samples=getattr(self, ConfigName.SAMPLES),
            days=getattr(self, ConfigName.DAYS),
            working_hours=getattr(self, ConfigName.WORKING_HOURS),
            wet_days=getattr(self, ConfigName.WET_DAYS),
            dry_days=getattr(self, ConfigName.DRY_DAYS),
            badtide_days=getattr(self, ConfigName.BADTIDE_DAYS),
        )


SimulationConfig = create_model(
    "SimulationConfig",
    **{ConfigName.ENERGY_MIX: EnergyMixConfig},
    __base__=SimulationConfigBase,
)

# Flatten CommonConfig into SimulationConfig
for name, field in CommonConfig.model_fields.items():
    SimulationConfig.model_fields[name] = field
SimulationConfig.model_rebuild(force=True)

OptimizationConfig = create_model(
    "OptimizationConfig",
    **{ConfigName.ITERATIONS: int},
)

# Flatten CommonConfig into OptimizationConfig
for name, field in CommonConfig.model_fields.items():
    OptimizationConfig.model_fields[name] = field
OptimizationConfig.model_rebuild(force=True)


class ProducerGroup(NamedTuple):
    """Combines a machine count with its individual power production rate."""

    count: int
    power: int


class SimulationSample(NamedTuple):
    """Represents the time-series data for production and storage state from a single simulation run."""

    power_production: np.ndarray
    battery_charge: np.ndarray


class AggregatedSamples(NamedTuple):
    """Holds aggregated metrics and consumption profiles collected across all samples in a simulation parallel."""

    hours_empty_results: np.ndarray
    final_surpluses: np.ndarray
    power_consumption: np.ndarray


class ParallelSimulationResult(NamedTuple):
    """Combines the results of a parallel simulation, including the worst-performing sample and aggregated statistics."""

    worst_sample: SimulationSample
    aggregated_samples: AggregatedSamples
