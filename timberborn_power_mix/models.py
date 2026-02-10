from pydantic import BaseModel, Field, ConfigDict
from typing import List, Tuple, Union
import numpy as np
from timberborn_power_mix import consts


class FactoryParams(BaseModel):
    lumber_mill: int = 0
    gear_workshop: int = 0
    steel_factory: int = 0
    wood_workshop: int = 0
    paper_mill: int = 0
    printing_press: int = 0
    observatory: int = 0
    bot_part_factory: int = 0
    bot_assembler: int = 0
    explosives_factory: int = 0
    grillmist: int = 0
    centrifuge: int = 0


class EnergyMixParams(BaseModel):
    power_wheels: int = 0
    water_wheels: int = 0
    large_windmills: int = 0
    windmills: int = 0
    batteries: int = 0
    battery_height: Union[int, float] = 0


class SimulationParams(BaseModel):
    days: int = Field(default=consts.DEFAULT_DAYS)
    working_hours: int = Field(default=consts.DEFAULT_WORKING_HOURS)
    wet_season_days: int = Field(default=consts.DEFAULT_WET_SEASON_DAYS)
    dry_season_days: int = Field(default=consts.DEFAULT_DRY_SEASON_DAYS)
    badtide_season_days: int = Field(default=consts.DEFAULT_BADTIDE_SEASON_DAYS)
    factories: FactoryParams = Field(default_factory=FactoryParams)
    energy_mix: EnergyMixParams = Field(default_factory=EnergyMixParams)


class SimulationResult(BaseModel):
    time_days: np.ndarray
    power_production: np.ndarray
    power_consumption: np.ndarray
    power_surplus: np.ndarray
    effective_balance: np.ndarray
    battery_charge: np.ndarray
    energy_production: np.ndarray
    energy_consumption: np.ndarray
    total_battery_capacity: float
    season_boundaries: List[Tuple[int, str]]
    params: SimulationParams
    total_cost: float

    model_config = ConfigDict(arbitrary_types_allowed=True)
