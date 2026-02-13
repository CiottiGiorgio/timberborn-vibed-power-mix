from pydantic import BaseModel, ConfigDict, create_model
from typing import List, Tuple
import numpy as np
from timberborn_power_mix.machines import (
    FACTORY_DATABASE,
    PRODUCER_DATABASE,
    BatteryName,
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
    power_consumption: np.ndarray
    battery_charge: np.ndarray
    total_battery_capacity: float
    season_boundaries: List[Tuple[int, str]]
    total_cost: float

    model_config = ConfigDict(arbitrary_types_allowed=True)
