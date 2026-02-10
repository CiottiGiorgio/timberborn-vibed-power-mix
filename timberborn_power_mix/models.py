from pydantic import BaseModel, Field, field_validator
from typing import Dict, List, Tuple, Union
import numpy as np
from . import consts
from .machines import iter_consumers


class FactoryParams(BaseModel):
    counts: Dict[str, int] = Field(default_factory=dict)

    @field_validator("counts")
    @classmethod
    def validate_counts(cls, v):
        valid_machines = {name for name, _ in iter_consumers()}
        for name in v:
            if name not in valid_machines:
                raise ValueError(
                    f"Unknown factory: {name}. Must be one of {valid_machines}"
                )
        return v

    def __getattr__(self, item):
        # Allow access like params.factories.lumber_mills
        if item in self.counts:
            return self.counts[item]
        raise AttributeError(f"'FactoryParams' object has no attribute '{item}'")


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
    season_boundaries: List[Tuple[float, str]]
    params: SimulationParams
    total_cost: float

    class Config:
        arbitrary_types_allowed = True
