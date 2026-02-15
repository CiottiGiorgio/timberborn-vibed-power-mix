from typing import NamedTuple, Optional
import numpy as np
from pydantic import create_model, BaseModel

from timberborn_power_mix.machines import (
    BatteryName,
    PRODUCER_DATABASE,
)
from timberborn_power_mix.models import ConfigName, CommonConfig

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
    seed: Optional[int] = None
    samples: int
    days: int
    working_hours: int
    wet_days: int
    dry_days: int
    badtide_days: int
    factories: FactoryConfig
    energy_mix: EnergyMixConfig
"""

EnergyMixConfig = create_model(
    "EnergyMixConfig",
    **{BatteryName.BATTERY: int, BatteryName.BATTERY_HEIGHT: float},
    **{key: int for key in PRODUCER_DATABASE.keys()},
)


class JitSimulationConfig(NamedTuple):
    """Subset of SimulationConfig used for jitted simulation configuration."""

    samples: int
    days: int
    working_hours: int
    wet_days: int
    dry_days: int
    badtide_days: int
    seed: Optional[int] = None


class SimulationConfigBase(BaseModel):
    @property
    def to_jit_config(self) -> JitSimulationConfig:
        return JitSimulationConfig(
            samples=getattr(self, ConfigName.SAMPLES),
            days=getattr(self, ConfigName.DAYS),
            working_hours=getattr(self, ConfigName.WORKING_HOURS),
            wet_days=getattr(self, ConfigName.WET_DAYS),
            dry_days=getattr(self, ConfigName.DRY_DAYS),
            badtide_days=getattr(self, ConfigName.BADTIDE_DAYS),
            seed=getattr(self, ConfigName.SEED),
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


class ProducerGroup(NamedTuple):
    """Combines a machine count with its individual power production rate."""

    count: int
    power: int


class SimulationSample(NamedTuple):
    """Represents the time-series data for production and storage state from a single simulation run."""

    power_production: np.ndarray
    battery_charge: np.ndarray


class AggregatedSamples(NamedTuple):
    """Holds aggregated metrics and consumption profiles collected across all samples in a simulation."""

    power_consumption: np.ndarray
    hours_empty_results: np.ndarray


class SimulationResult(NamedTuple):
    """Final output of the simulation process, containing aggregated metrics and the worst-case scenario."""

    worst_sample: SimulationSample
    aggregated_samples: AggregatedSamples
