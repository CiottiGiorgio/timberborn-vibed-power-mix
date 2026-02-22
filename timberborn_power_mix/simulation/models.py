from typing import List
from pydantic import create_model, BaseModel

from timberborn_power_mix.machines import (
    BatteryName,
    PRODUCER_DATABASE,
)
from timberborn_power_mix.models import CommonConfig
from timberborn_power_mix.structures import (
    CommonConfigName,
    SimulateConfigName,
    JitSimulationConfig,
)

"""
This module defines the configuration and result models for the power simulation.
Many models are created dynamically using Pydantic's `create_model` to stay in sync 
with the machine databases defined in `machines.py`.

The dynamic structures effectively look like this:

class EnergyMixConfig(BaseModel):
    battery_heights: List[int]
    windmills: int
    water_wheels: int
    ... (all other producers)

class SimulationConfig(BaseModel):
    energy_mix: EnergyMixConfig
    seed: Optional[int] = None
    threads: Optional[int] = None
    samples: int
    days: int
    working_hours: int
    wet_days: int
    dry_days: int
    badtide_days: int
    factories: FactoryConfig
"""

EnergyMixConfig = create_model(
    "EnergyMixConfig",
    **{BatteryName.BATTERY_HEIGHTS.value: List[int]},
    **{key.value: int for key in PRODUCER_DATABASE.keys()},
)


class SimulationConfigBase(BaseModel):
    """Base class for SimulationConfig providing conversion to JIT-compatible structures."""

    def to_jit_config(self) -> JitSimulationConfig:
        """Converts the Pydantic model to a JIT-compatible NamedTuple."""
        return JitSimulationConfig(
            seed=getattr(self, CommonConfigName.SEED),
            samples=getattr(self, CommonConfigName.SAMPLES),
            days=getattr(self, CommonConfigName.DAYS),
            working_hours=getattr(self, CommonConfigName.WORKING_HOURS),
            wet_days=getattr(self, CommonConfigName.WET_DAYS),
            dry_days=getattr(self, CommonConfigName.DRY_DAYS),
            badtide_days=getattr(self, CommonConfigName.BADTIDE_DAYS),
        )


SimulationConfig = create_model(
    "SimulationConfig",
    **{SimulateConfigName.ENERGY_MIX: EnergyMixConfig},
    __base__=SimulationConfigBase,
)

# Flatten CommonConfig into SimulationConfig
for name, field in CommonConfig.model_fields.items():
    SimulationConfig.model_fields[name] = field
SimulationConfig.model_rebuild(force=True)
