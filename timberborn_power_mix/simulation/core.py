import numpy as np
from numpy.typing import NDArray
from numba import njit

from timberborn_power_mix.simulation import consts
from timberborn_power_mix.structures import (
    ProducerGroup,
    SimulationSample,
)


@njit
def jit_stochastic_simulation(
    seed: np.uint32,
    total_hours: int,
    base_power_production: NDArray[np.uint32],
    power_consumption: NDArray[np.uint32],
    total_battery_capacity: int,
    large_windmills: ProducerGroup,
    windmills: ProducerGroup,
) -> SimulationSample:
    """Performs a single Monte Carlo simulation run with a specific seed."""
    np.random.seed(seed)

    # Generate wind data
    max_segments = (total_hours // consts.WIND_DURATION_MIN_HOURS) + 1
    wind_durations = np.random.randint(
        consts.WIND_DURATION_MIN_HOURS,
        consts.WIND_DURATION_MAX_HOURS,
        size=max_segments,
    ).astype(np.int32)
    wind_strengths = np.random.random(size=max_segments)

    # Optimized Wind production using expansion
    wind_strength_profile = np.repeat(wind_strengths, wind_durations)[:total_hours]

    large_wind_unit_prod = np.where(
        wind_strength_profile > consts.LARGE_WINDMILL_THRESHOLD,
        wind_strength_profile * large_windmills.power,
        0,
    )
    small_wind_unit_prod = np.where(
        wind_strength_profile > consts.WINDMILL_THRESHOLD,
        wind_strength_profile * windmills.power,
        0,
    )
    wind_production = (large_windmills.quantity * large_wind_unit_prod) + (
        windmills.quantity * small_wind_unit_prod
    )

    power_production = base_power_production + wind_production
    power_surplus = power_production - power_consumption

    battery_charge = np.zeros(total_hours, dtype=np.uint32)
    current_charge = total_battery_capacity // 2

    for i in range(total_hours):
        potential_charge = current_charge + power_surplus[i]
        current_charge = max(0, min(potential_charge, total_battery_capacity))
        battery_charge[i] = current_charge

    return SimulationSample(
        power_production=power_production,
        battery_charge=battery_charge,
    )
