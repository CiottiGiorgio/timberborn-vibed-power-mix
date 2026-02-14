from typing import Tuple, List
import numpy as np

from timberborn_power_mix import consts
from timberborn_power_mix.machines import (
    PRODUCER_DATABASE,
    ProducerName,
    BatteryName,
    battery_cost,
    battery_capacity,
    FACTORY_DATABASE,
)
from timberborn_power_mix.simulation.models import EnergyMixConfig, SimulationConfig
from timberborn_power_mix.consts import ConfigKey


def calculate_total_cost(energy_mix: EnergyMixConfig) -> float:
    wheel_spec = PRODUCER_DATABASE[ProducerName.WATER_WHEEL]
    windmill_spec = PRODUCER_DATABASE[ProducerName.WINDMILL]
    large_windmill_spec = PRODUCER_DATABASE[ProducerName.LARGE_WINDMILL]
    power_wheel_spec = PRODUCER_DATABASE[ProducerName.POWER_WHEEL]

    num_batteries = getattr(energy_mix, BatteryName.BATTERY)
    num_water_wheels = getattr(energy_mix, ProducerName.WATER_WHEEL)
    num_windmills = getattr(energy_mix, ProducerName.WINDMILL)
    num_large_windmills = getattr(energy_mix, ProducerName.LARGE_WINDMILL)
    num_power_wheels = getattr(energy_mix, ProducerName.POWER_WHEEL)
    battery_height = getattr(energy_mix, BatteryName.BATTERY_HEIGHT)

    return (
        (num_power_wheels * power_wheel_spec.cost)
        + (num_water_wheels * wheel_spec.cost)
        + (num_large_windmills * large_windmill_spec.cost)
        + (num_windmills * windmill_spec.cost)
        + (num_batteries * battery_cost(battery_height))
    )


def calculate_total_battery_capacity(energy_mix: EnergyMixConfig) -> float:
    num_batteries = getattr(energy_mix, BatteryName.BATTERY)
    battery_height = getattr(energy_mix, BatteryName.BATTERY_HEIGHT)
    return num_batteries * battery_capacity(battery_height)


def calculate_season_boundaries(config: SimulationConfig) -> List[Tuple[int, str]]:
    season_boundaries = []
    curr_day = 0
    days = getattr(config, ConfigKey.DAYS)
    wet_days = getattr(config, ConfigKey.WET_DAYS)
    dry_days = getattr(config, ConfigKey.DRY_DAYS)
    badtide_days = getattr(config, ConfigKey.BADTIDE_DAYS)

    while curr_day < days:
        season_boundaries.append((curr_day * consts.HOURS_PER_DAY, "Wet"))
        curr_day += wet_days
        if curr_day >= days:
            break
        season_boundaries.append((curr_day * consts.HOURS_PER_DAY, "Dry"))
        curr_day += dry_days
        if curr_day >= days:
            break
        season_boundaries.append((curr_day * consts.HOURS_PER_DAY, "Wet"))
        curr_day += wet_days
        if curr_day >= days:
            break
        season_boundaries.append((curr_day * consts.HOURS_PER_DAY, "Badtide"))
        curr_day += badtide_days
    return season_boundaries


def calculate_power_consumption_profile(config: SimulationConfig) -> np.ndarray:
    days = getattr(config, ConfigKey.DAYS)
    working_hours = getattr(config, ConfigKey.WORKING_HOURS)
    factories = getattr(config, ConfigKey.FACTORIES)

    total_hours = days * consts.HOURS_PER_DAY
    time_hours = np.arange(total_hours)
    hour_of_day = time_hours % consts.HOURS_PER_DAY
    is_working_hour = hour_of_day < working_hours

    total_consumption_rate = 0
    for name, spec in FACTORY_DATABASE.items():
        count = getattr(factories, name)
        total_consumption_rate += count * spec.power

    return np.where(is_working_hour, total_consumption_rate, 0.0)
