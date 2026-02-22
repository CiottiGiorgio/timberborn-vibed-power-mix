from typing import Tuple, List

import numpy as np
from numpy.typing import NDArray
from numba import njit

from timberborn_power_mix.simulation import consts
from timberborn_power_mix.machines import (
    PRODUCER_DATABASE,
    FACTORY_DATABASE,
    ProducerName,
    BatteryName,
    battery_capacity,
)
from timberborn_power_mix.simulation.core import jit_stochastic_simulation
from timberborn_power_mix.simulation.models import (
    EnergyMixConfig,
    SimulationConfig,
)
from timberborn_power_mix.structures import (
    ProducerGroup,
    JitSimulationCachedConsts,
    JitSimulationConfig,
    SimulationResult,
    AggregatedSamples,
    JitSimulationPrelude,
)
from timberborn_power_mix.models import FactoryConfig
from timberborn_power_mix.structures import CommonConfigName


def calculate_total_battery_capacity(energy_mix: EnergyMixConfig) -> int:
    """Calculates the total energy storage capacity of all batteries in the mix."""
    heights: list[int] = getattr(energy_mix, BatteryName.BATTERY_HEIGHTS)
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
    """
    Pre-calculates static simulation constants from the full configuration.

    This converts high-level Pydantic models into a JIT-friendly NamedTuple
    containing only the raw numerical data needed by the simulation engine.
    """
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
    """
    Determines the start hour and name of each season in the simulation timeline.

    Used primarily for plotting and UI to show when seasons transition.
    """
    season_boundaries = []
    curr_day = 0
    days = getattr(config, CommonConfigName.DAYS)
    wet_days = getattr(config, CommonConfigName.WET_DAYS)
    dry_days = getattr(config, CommonConfigName.DRY_DAYS)
    badtide_days = getattr(config, CommonConfigName.BADTIDE_DAYS)

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


@njit
def jit_simulation_prelude(
    config: JitSimulationConfig,
    sim_consts: JitSimulationCachedConsts,
) -> JitSimulationPrelude:
    """
    Pre-calculates static time-series profiles that are constant across all samples.

    This includes:
    - Working hour schedules
    - Seasonal water wheel availability
    - Base power production (water wheels + power wheels)
    - Power consumption profile
    - Base surplus (production - consumption)
    - Working hour mask
    """
    total_hours = config.days * consts.HOURS_PER_DAY

    # Pre-calculate static profiles
    time_hours = np.arange(total_hours, dtype=np.uint32)
    hour_of_day = time_hours % consts.HOURS_PER_DAY
    is_working_hour = hour_of_day < config.working_hours

    # Inlined calculate_base_power_production logic
    hours_per_wet = config.wet_days * consts.HOURS_PER_DAY
    hours_per_dry = config.dry_days * consts.HOURS_PER_DAY
    hours_per_badtide = config.badtide_days * consts.HOURS_PER_DAY
    cycle_length_hours = 2 * hours_per_wet + hours_per_dry + hours_per_badtide

    hour_of_cycle = time_hours % cycle_length_hours
    is_dry = (hour_of_cycle >= hours_per_wet) & (
        hour_of_cycle < (hours_per_wet + hours_per_dry)
    )
    is_water_active = ~is_dry

    power_wheel_production_rate = np.where(
        is_working_hour,
        sim_consts.power_wheels.quantity * sim_consts.power_wheels.power,
        0,
    )
    water_wheel_production_rate = (
        sim_consts.water_wheels.quantity * sim_consts.water_wheels.power
    )

    base_power_production = (
        np.where(is_water_active, water_wheel_production_rate, 0)
        + power_wheel_production_rate
    ).astype(np.uint32)

    power_consumption = np.where(
        is_working_hour, sim_consts.total_consumption_rate, 0
    ).astype(np.uint32)

    base_surplus = base_power_production.astype(np.int64) - power_consumption.astype(
        np.int64
    )

    return JitSimulationPrelude(
        base_surplus=base_surplus,
        base_power_production=base_power_production,
        power_consumption=power_consumption,
        is_working_hour=is_working_hour,
        total_hours=total_hours,
    )


@njit
def jit_simulation_epilogue(
    all_hours_empty: NDArray[np.uint32],
    all_seeds: NDArray[np.uint32],
    base_surplus: NDArray[np.int64],
    base_power_production: NDArray[np.uint32],
    power_consumption: NDArray[np.uint32],
    sim_consts: JitSimulationCachedConsts,
    total_hours: int,
) -> SimulationResult:
    """
    Aggregates results from all samples and identifies the P95 worst-case scenario.

    After all stochastic runs are complete, this function:
    1. Packages the aggregated unreliability data.
    2. Identifies the seed that produced the 95th percentile result.
    3. Re-runs that specific seed to generate full time-series data for visualization.
    """
    aggregated = AggregatedSamples(
        power_consumption=power_consumption,
        hours_empty_results=all_hours_empty,
    )

    # Find p95 sample (Second Pass)
    p95_hours_empty = np.percentile(all_hours_empty, 95)
    p95_idx = np.where(all_hours_empty >= p95_hours_empty)[0][0]
    p95_seed = all_seeds[p95_idx]

    p95_sample = jit_stochastic_simulation(
        p95_seed,
        total_hours,
        base_surplus,
        base_power_production,
        sim_consts.total_battery_capacity,
        sim_consts.large_windmills,
        sim_consts.windmills,
    )

    return SimulationResult(
        p95_sample=p95_sample,
        aggregated_samples=aggregated,
    )
