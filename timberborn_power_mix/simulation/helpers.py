from typing import Tuple, List
import numpy as np
from numba import njit

from timberborn_power_mix.simulation import consts
from timberborn_power_mix.machines import (
    PRODUCER_DATABASE,
    FACTORY_DATABASE,
    ProducerName,
    BatteryName,
    battery_capacity,
)
from timberborn_power_mix.simulation.models import (
    EnergyMixConfig,
    SimulationConfig,
)
from timberborn_power_mix.structures import (
    ProducerGroup,
    JitSimulationCachedConsts,
)
from timberborn_power_mix.models import FactoryConfig
from timberborn_power_mix.structures import ConfigName


def calculate_total_battery_capacity(energy_mix: EnergyMixConfig) -> float:
    """Calculates the total energy storage capacity of all batteries in the mix."""
    heights = getattr(energy_mix, BatteryName.BATTERY_HEIGHTS)
    return sum(battery_capacity(h) for h in heights)


def calculate_total_consumption_rate(factories: FactoryConfig) -> int:
    """Calculates the total power consumption rate of all active factories."""
    total_consumption_rate = 0
    for name, spec in FACTORY_DATABASE.items():
        count = getattr(factories, name)
        total_consumption_rate += count * spec.power
    return total_consumption_rate


def calculate_jit_cached_consts(
    config: SimulationConfig,
) -> JitSimulationCachedConsts:
    """Pre-calculates static simulation constants from the full configuration."""
    # Consumption
    total_consumption_rate = calculate_total_consumption_rate(config.factories)

    # Production specs
    wheel_spec = PRODUCER_DATABASE[ProducerName.WATER_WHEELS]
    windmill_spec = PRODUCER_DATABASE[ProducerName.WINDMILLS]
    large_windmill_spec = PRODUCER_DATABASE[ProducerName.LARGE_WINDMILLS]
    power_wheel_spec = PRODUCER_DATABASE[ProducerName.POWER_WHEELS]

    # Counts
    num_water_wheels = getattr(config.energy_mix, ProducerName.WATER_WHEELS)
    num_windmills = getattr(config.energy_mix, ProducerName.WINDMILLS)
    num_large_windmills = getattr(config.energy_mix, ProducerName.LARGE_WINDMILLS)
    num_power_wheels = getattr(config.energy_mix, ProducerName.POWER_WHEELS)

    total_battery_capacity = calculate_total_battery_capacity(config.energy_mix)

    return JitSimulationCachedConsts(
        total_consumption_rate=total_consumption_rate,
        total_battery_capacity=total_battery_capacity,
        large_windmills=ProducerGroup(num_large_windmills, large_windmill_spec.power),
        windmills=ProducerGroup(num_windmills, windmill_spec.power),
        power_wheels=ProducerGroup(num_power_wheels, power_wheel_spec.power),
        water_wheels=ProducerGroup(num_water_wheels, wheel_spec.power),
    )


def calculate_season_boundaries(
    config: SimulationConfig,
) -> List[Tuple[int, str]]:
    """Determines the start hour and name of each season in the simulation timeline."""
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
        is_working_hour, power_wheels.quantity * power_wheels.power, 0.0
    )
    water_wheel_production_rate = water_wheels.quantity * water_wheels.power

    return (
        np.where(is_water_active, water_wheel_production_rate, 0.0)
        + power_wheel_production_rate
    )


@njit(cache=True)
def calculate_battery_stress(
    battery_charge: np.ndarray, total_battery_capacity: float
) -> float:
    """
    Calculates a stress index for the battery state.
    Higher values indicate more time spent near zero charge.
    """
    if total_battery_capacity <= 0:
        return float(battery_charge.size)

    # sum((1 - charge/capacity)^8)
    return np.sum((1.0 - (battery_charge / total_battery_capacity)) ** 8)
