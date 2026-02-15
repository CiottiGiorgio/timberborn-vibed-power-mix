from typing import Tuple, List
import numpy as np
from numba import njit

from timberborn_power_mix import consts
from timberborn_power_mix.machines import (
    PRODUCER_DATABASE,
    ProducerName,
    BatteryName,
    battery_cost,
    battery_capacity,
)
from timberborn_power_mix.simulation.models import (
    EnergyMixConfig,
    SimulationConfig,
    ProducerGroup,
)
from timberborn_power_mix.models import ConfigName


def calculate_total_wood_cost(energy_mix: EnergyMixConfig) -> float:
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
    days = getattr(config, ConfigName.DAYS)
    wet_days = getattr(config, ConfigName.WET_DAYS)
    dry_days = getattr(config, ConfigName.DRY_DAYS)
    badtide_days = getattr(config, ConfigName.BADTIDE_DAYS)

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


@njit(inline="always")
def calculate_base_power_production(
    time_hours: np.ndarray,
    is_working_hour: np.ndarray,
    wet_days: int,
    dry_days: int,
    badtide_days: int,
    power_wheels: ProducerGroup,
    water_wheels: ProducerGroup,
) -> np.ndarray:
    """Calculates the deterministic base power production profile (water wheels and power wheels)."""
    hours_per_wet = wet_days * consts.HOURS_PER_DAY
    hours_per_dry = dry_days * consts.HOURS_PER_DAY
    hours_per_badtide = badtide_days * consts.HOURS_PER_DAY
    cycle_length_hours = 2 * hours_per_wet + hours_per_dry + hours_per_badtide

    hour_of_cycle = time_hours % cycle_length_hours
    is_dry = (hour_of_cycle >= hours_per_wet) & (
        hour_of_cycle < (hours_per_wet + hours_per_dry)
    )
    is_water_active = ~is_dry

    power_wheel_production_rate = np.where(
        is_working_hour, power_wheels.count * power_wheels.power, 0.0
    )
    water_wheel_production_rate = water_wheels.count * water_wheels.power

    return (
        np.where(is_water_active, water_wheel_production_rate, 0.0)
        + power_wheel_production_rate
    )
