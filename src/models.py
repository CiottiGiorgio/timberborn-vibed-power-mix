from pydantic import BaseModel, Field, field_validator
from typing import Dict, Literal, List, Tuple, Union
import numpy as np
import consts


class Machine(BaseModel):
    power: float
    role: Literal["consumer", "producer"]
    cost: int = 0


class BatteryConfig(BaseModel):
    base_capacity: float
    capacity_per_height: float
    base_cost: int = 0
    cost_per_height: int = 0


class SimulationConfig(BaseModel):
    machines: Dict[str, Machine]
    battery: BatteryConfig

    @classmethod
    def from_json_file(cls, filepath: str) -> "SimulationConfig":
        with open(filepath, "r") as f:
            json_content = f.read()
        return cls.model_validate_json(json_content)


class FactoryParams(BaseModel):
    lumber_mills: int = Field(default=consts.DEFAULT_LUMBER_MILLS)
    gear_workshops: int = Field(default=consts.DEFAULT_GEAR_WORKSHOPS)
    steel_factories: int = Field(default=consts.DEFAULT_STEEL_FACTORIES)
    wood_workshops: int = Field(default=consts.DEFAULT_WOOD_WORKSHOPS)
    paper_mills: int = Field(default=consts.DEFAULT_PAPER_MILLS)
    printing_presses: int = Field(default=consts.DEFAULT_PRINTING_PRESSES)
    observatories: int = Field(default=consts.DEFAULT_OBSERVATORIES)
    bot_part_factories: int = Field(default=consts.DEFAULT_BOT_PART_FACTORIES)
    bot_assemblers: int = Field(default=consts.DEFAULT_BOT_ASSEMBLERS)
    explosives_factories: int = Field(default=consts.DEFAULT_EXPLOSIVES_FACTORIES)
    grillmists: int = Field(default=consts.DEFAULT_GRILLMISTS)


class EnergyMixParams(BaseModel):
    water_wheels: int = Field(default=consts.DEFAULT_WATER_WHEELS)
    large_windmills: int = Field(default=consts.DEFAULT_LARGE_WINDMILLS)
    windmills: int = Field(default=consts.DEFAULT_WINDMILLS)
    batteries: int = Field(default=consts.DEFAULT_BATTERIES)
    battery_height: Union[int, List[int]] = Field(default=consts.DEFAULT_BATTERY_HEIGHT)

    @field_validator("battery_height")
    def validate_battery_height(cls, v, info):
        if isinstance(v, list):
            # We need to access the 'batteries' field.
            # In Pydantic v2, we can access other fields via info.data
            batteries = info.data.get("batteries")
            if batteries is not None and len(v) != batteries:
                raise ValueError(
                    f"Length of battery_height list ({len(v)}) must match number of batteries ({batteries})"
                )
        return v


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
    effective_surplus: np.ndarray
    effective_deficit: np.ndarray
    battery_charge: np.ndarray
    energy_production: np.ndarray
    energy_consumption: np.ndarray
    total_battery_capacity: float
    season_boundaries: List[Tuple[float, str]]
    params: SimulationParams
    total_cost: float

    class Config:
        arbitrary_types_allowed = True
