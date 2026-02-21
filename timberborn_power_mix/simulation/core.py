import numpy as np
from numpy.typing import NDArray
from numba import njit

from timberborn_power_mix.simulation import consts
from timberborn_power_mix.structures import (
    ProducerGroup,
    SimulationSample,
)


@njit(inline="always")
def _calculate_wind_power(
    large_windmills: ProducerGroup,
    windmills: ProducerGroup,
) -> int:
    """Calculates total wind power for a given wind strength."""
    strength = np.random.random()
    l_prod = (
        strength * large_windmills.power
        if strength > consts.LARGE_WINDMILL_THRESHOLD
        else 0
    )
    s_prod = strength * windmills.power if strength > consts.WINDMILL_THRESHOLD else 0
    return (large_windmills.quantity * l_prod) + (windmills.quantity * s_prod)


@njit
def jit_stochastic_simulation_no_sample(
    seed: np.uint32,
    total_hours: int,
    base_surplus: NDArray[np.int64],
    total_battery_capacity: int,
    large_windmills: ProducerGroup,
    windmills: ProducerGroup,
) -> np.uint32:
    """Variant that only returns the number of hours the battery was empty."""
    np.random.seed(seed)

    current_hour = 0
    current_charge = np.int64(total_battery_capacity // 2)
    empty_hours = 0

    while current_hour < total_hours:
        duration = np.random.randint(
            consts.WIND_DURATION_MIN_HOURS, consts.WIND_DURATION_MAX_HOURS
        )
        wind_power = _calculate_wind_power(large_windmills, windmills)

        end_hour = min(current_hour + duration, total_hours)
        for h in range(current_hour, end_hour):
            current_charge += base_surplus[h] + np.int64(wind_power)
            if current_charge < 0:
                current_charge = 0
                empty_hours += 1
            elif current_charge > total_battery_capacity:
                current_charge = np.int64(total_battery_capacity)

        current_hour = end_hour

    return np.uint32(empty_hours)


@njit
def jit_stochastic_simulation(
    seed: np.uint32,
    total_hours: int,
    base_surplus: NDArray[np.int64],
    base_power_production: NDArray[np.uint32],
    total_battery_capacity: int,
    large_windmills: ProducerGroup,
    windmills: ProducerGroup,
) -> SimulationSample:
    """Variant that returns the full time-series data."""
    np.random.seed(seed)

    power_production = np.empty(total_hours, dtype=np.uint32)
    battery_charge = np.empty(total_hours, dtype=np.uint32)

    current_hour = 0
    current_charge = np.int64(total_battery_capacity // 2)

    while current_hour < total_hours:
        duration = np.random.randint(
            consts.WIND_DURATION_MIN_HOURS, consts.WIND_DURATION_MAX_HOURS
        )
        wind_power = _calculate_wind_power(large_windmills, windmills)

        end_hour = min(current_hour + duration, total_hours)
        for h in range(current_hour, end_hour):
            wind_power_int = np.uint32(wind_power)
            power_production[h] = base_power_production[h] + wind_power_int

            current_charge += base_surplus[h] + np.int64(wind_power_int)
            if current_charge < 0:
                current_charge = 0
            elif current_charge > total_battery_capacity:
                current_charge = np.int64(total_battery_capacity)
            battery_charge[h] = np.uint32(current_charge)

        current_hour = end_hour

    return SimulationSample(
        power_production=power_production,
        battery_charge=battery_charge,
    )
