from pydantic import BaseModel, Field
from typing import Dict, Literal
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


class SimulationParams(BaseModel):
    days: int = Field(default=consts.DEFAULT_DAYS)
    working_hours: int = Field(default=consts.DEFAULT_WORKING_HOURS)
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
    water_wheels: int = Field(default=consts.DEFAULT_WATER_WHEELS)
    large_windmills: int = Field(default=consts.DEFAULT_LARGE_WINDMILLS)
    windmills: int = Field(default=consts.DEFAULT_WINDMILLS)
    batteries: int = Field(default=consts.DEFAULT_BATTERIES)
    battery_height: int = Field(default=consts.DEFAULT_BATTERY_HEIGHT)
    wet_season_days: int = Field(default=consts.DEFAULT_WET_SEASON_DAYS)
    dry_season_days: int = Field(default=consts.DEFAULT_DRY_SEASON_DAYS)
    badtide_season_days: int = Field(default=consts.DEFAULT_BADTIDE_SEASON_DAYS)
