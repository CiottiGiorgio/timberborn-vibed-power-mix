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
    lumber_mill: int = 0
    gear_workshop: int = 0
    ... (all other factories)

class CommonConfig(BaseModel):
    seed: Optional[int] = None
    samples: int
    days: int
    working_hours: int
    wet_days: int
    dry_days: int
    badtide_days: int
    threads: Optional[int] = None
    factories: FactoryConfig
"""


class ConfigName(StrEnum):
    SEED = "seed"
    SAMPLES = "samples"
    DAYS = "days"
    WORKING_HOURS = "working_hours"
    WET_DAYS = "wet_days"
    DRY_DAYS = "dry_days"
    BADTIDE_DAYS = "badtide_days"
    FACTORIES = "factories"
    ENERGY_MIX = "energy_mix"
    ITERATION = "iteration"
    THREADS = "threads"


FactoryConfig = create_model(
    "FactoryConfig", **{key: int for key in FACTORY_DATABASE.keys()}
)

CommonConfig = create_model(
    "CommonConfig",
    **{ConfigName.SEED: (Optional[int], None)},
    **{ConfigName.SAMPLES: int},
    **{ConfigName.DAYS: int},
    **{ConfigName.WORKING_HOURS: int},
    **{ConfigName.WET_DAYS: int},
    **{ConfigName.DRY_DAYS: int},
    **{ConfigName.BADTIDE_DAYS: int},
    **{ConfigName.THREADS: (Optional[int], None)},
    **{ConfigName.FACTORIES: FactoryConfig},
)
