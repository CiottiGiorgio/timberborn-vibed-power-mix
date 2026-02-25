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
    """
    Calculates total wind power for a single wind event.

    Wind strength is sampled from a uniform distribution and applied to
    windmill groups if they exceed their respective activation thresholds.
    """
    # Use integer math for wind strength (0-WIND_STRENGTH_MAX) for performance.
    strength_int = np.random.randint(0, consts.WIND_STRENGTH_MAX + 1)

    l_prod = 0
    if strength_int > consts.LARGE_WINDMILL_THRESHOLD:
        l_prod = (strength_int * large_windmills.power) // consts.WIND_STRENGTH_MAX

    s_prod = 0
    if strength_int > consts.WINDMILL_THRESHOLD:
        s_prod = (strength_int * windmills.power) // consts.WIND_STRENGTH_MAX

    return (large_windmills.quantity * l_prod) + (windmills.quantity * s_prod)


@njit
def jit_stochastic_simulation_no_sample(
    seed: np.uint32,
    total_hours: int,
    base_surplus: NDArray[np.int64],
    power_consumption: NDArray[np.uint32],
    is_working_hour: NDArray[np.bool_],
    total_battery_capacity: int,
    large_windmills: ProducerGroup,
    windmills: ProducerGroup,
) -> np.float64:
    """
    Executes a single Monte Carlo simulation run and returns the number of hours
    where the battery charge was zero. This optimized variant avoids allocating
    large arrays for time-series data, making it suitable for high-iteration
    optimization loops.
    """
    np.random.seed(seed)

    current_hour = 0
    current_charge = np.int64(total_battery_capacity // 2)
    lost_working_hours = 0.0

    while current_hour < total_hours:
        duration = np.random.randint(
            consts.WIND_DURATION_MIN_HOURS, consts.WIND_DURATION_MAX_HOURS
        )
        wind_power = _calculate_wind_power(large_windmills, windmills)

        end_hour = min(current_hour + duration, total_hours)
        for h in range(current_hour, end_hour):
            wind_power_int = np.uint32(wind_power)
            current_charge += base_surplus[h] + np.int64(wind_power_int)

            # Check for battery underflow (empty battery)
            if current_charge < 0:
                deficit = float(-current_charge)
                current_charge = np.int64(0)

                # If factories are running, calculate productivity loss
                if is_working_hour[h]:
                    consumption = float(power_consumption[h])
                    if consumption > 0:
                        # Productivity loss is the fraction of demand not met
                        loss = deficit / consumption
                        if loss > 1.0:
                            loss = 1.0
                        lost_working_hours += loss

            # Check for battery overflow (full battery)
            elif current_charge > total_battery_capacity:
                current_charge = np.int64(total_battery_capacity)

        current_hour = end_hour

    return np.float64(lost_working_hours)


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
    """
    Executes a single Monte Carlo simulation run and returns full time-series data
    for hourly power production and battery charge levels. Used primarily for
    visualization and detailed analysis of a specific scenario.
    """
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

            # Check for battery underflow (empty battery)
            if current_charge < 0:
                current_charge = np.int64(0)

            # Check for battery overflow (full battery)
            elif current_charge > total_battery_capacity:
                current_charge = np.int64(total_battery_capacity)

            battery_charge[h] = np.uint32(current_charge)

        current_hour = end_hour

    return SimulationSample(
        power_production=power_production,
        battery_charge=battery_charge,
    )
