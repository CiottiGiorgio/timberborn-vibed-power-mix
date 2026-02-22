from typing import Optional

from pydantic import create_model
from timberborn_power_mix.machines import FACTORY_DATABASE
from timberborn_power_mix.structures import CommonConfigName

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


FactoryConfig = create_model(
    "FactoryConfig", **{key.value: int for key in FACTORY_DATABASE.keys()}
)

CommonConfig = create_model(
    "CommonConfig",
    **{CommonConfigName.SEED.value: (Optional[int], None)},
    **{CommonConfigName.THREADS.value: (Optional[int], None)},
    **{CommonConfigName.SAMPLES.value: int},
    **{CommonConfigName.DAYS.value: int},
    **{CommonConfigName.WORKING_HOURS.value: int},
    **{CommonConfigName.WET_DAYS.value: int},
    **{CommonConfigName.DRY_DAYS.value: int},
    **{CommonConfigName.BADTIDE_DAYS.value: int},
    **{CommonConfigName.FACTORIES.value: FactoryConfig},
)
