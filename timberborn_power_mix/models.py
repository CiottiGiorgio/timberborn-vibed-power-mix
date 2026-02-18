from enum import StrEnum
from typing import Optional

from pydantic import create_model
from timberborn_power_mix.machines import FACTORY_DATABASE

"""
This module defines the base configuration models for the power simulation.
Many models are created dynamically using Pydantic's `create_model` to stay in sync 
with the machine databases defined in `machines.py`.

The dynamic structures effectively look like this:

class FactoryConfig(BaseModel):
    lumber_mills: int
    gear_workshops: int
    ... (all other factories)

class CommonConfig(BaseModel):
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


FactoryConfig = create_model(
    "FactoryConfig", **{key.value: int for key in FACTORY_DATABASE.keys()}
)

CommonConfig = create_model(
    "CommonConfig",
    **{ConfigName.SEED.value: (Optional[int], None)},
    **{ConfigName.THREADS.value: (Optional[int], None)},
    **{ConfigName.SAMPLES.value: int},
    **{ConfigName.DAYS.value: int},
    **{ConfigName.WORKING_HOURS.value: int},
    **{ConfigName.WET_DAYS.value: int},
    **{ConfigName.DRY_DAYS.value: int},
    **{ConfigName.BADTIDE_DAYS.value: int},
    **{ConfigName.FACTORIES.value: FactoryConfig},
)
