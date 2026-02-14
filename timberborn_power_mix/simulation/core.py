import numpy as np
from numba import njit, prange
from timberborn_power_mix.simulation.models import (
    SimulationConfig,
    ParallelSimulationConfig,
)
from timberborn_power_mix.machines import (
    PRODUCER_DATABASE,
    FACTORY_DATABASE,
    ProducerName,
)
from timberborn_power_mix import consts
from timberborn_power_mix.simulation.models import (
    SimulationSample,
    ParallelSimulationResult,
    AggregatedSamples,
    ProducerGroup,
    SimulationResult,
)
import timberborn_power_mix.simulation.helpers as sim_helpers


def run_simulation(config: SimulationConfig) -> SimulationResult:
    """Bridges pure Python and Numba by reshaping input parameters and aggregating simulation results for external modules."""
    # Consumption
    total_consumption_rate = 0
    for name, spec in FACTORY_DATABASE.items():
        count = getattr(config.factories, name)
        total_consumption_rate += count * spec.power

    # Production specs
    wheel_spec = PRODUCER_DATABASE[ProducerName.WATER_WHEEL]
    windmill_spec = PRODUCER_DATABASE[ProducerName.WINDMILL]
    large_windmill_spec = PRODUCER_DATABASE[ProducerName.LARGE_WINDMILL]
    power_wheel_spec = PRODUCER_DATABASE[ProducerName.POWER_WHEEL]

    # Counts
    num_water_wheels = getattr(config.energy_mix, ProducerName.WATER_WHEEL)
    num_windmills = getattr(config.energy_mix, ProducerName.WINDMILL)
    num_large_windmills = getattr(config.energy_mix, ProducerName.LARGE_WINDMILL)
    num_power_wheels = getattr(config.energy_mix, ProducerName.POWER_WHEEL)

    total_battery_capacity = sim_helpers.calculate_total_battery_capacity(
        config.energy_mix
    )

    parallel_res = jit_parallel_simulation(
        config.to_parallel_config,
        total_battery_capacity,
        total_consumption_rate,
        ProducerGroup(num_large_windmills, large_windmill_spec.power),
        ProducerGroup(num_windmills, windmill_spec.power),
        ProducerGroup(num_power_wheels, power_wheel_spec.power),
        ProducerGroup(num_water_wheels, wheel_spec.power),
    )

    return SimulationResult(
        hours_empty_results=parallel_res.aggregated_samples.hours_empty_results,
        worst_sample=parallel_res.worst_sample,
        average_final_surplus=float(
            np.mean(parallel_res.aggregated_samples.final_surpluses)
        ),
    )


@njit(parallel=True, cache=True)
def jit_parallel_simulation(
    config: ParallelSimulationConfig,
    total_battery_capacity: float,
    total_consumption_rate: int,
    large_windmills: ProducerGroup,
    windmills: ProducerGroup,
    power_wheels: ProducerGroup,
    water_wheels: ProducerGroup,
) -> ParallelSimulationResult:
    """Manages parallel simulation execution, including heavy memory allocation and caching of shared read-only arrays."""
    total_hours = config.days * consts.HOURS_PER_DAY
    time_hours = np.arange(total_hours)

    # Pre-calculate static profiles
    hour_of_day = time_hours % consts.HOURS_PER_DAY
    is_working_hour = hour_of_day < config.working_hours

    hours_per_wet = config.wet_days * consts.HOURS_PER_DAY
    hours_per_dry = config.dry_days * consts.HOURS_PER_DAY
    hours_per_badtide = config.badtide_days * consts.HOURS_PER_DAY
    cycle_length_hours = 2 * hours_per_wet + hours_per_dry + hours_per_badtide

    hour_of_cycle = time_hours % cycle_length_hours
    is_first_wet = hour_of_cycle < hours_per_wet
    is_second_wet = (hour_of_cycle >= (hours_per_wet + hours_per_dry)) & (
        hour_of_cycle < (2 * hours_per_wet + hours_per_dry)
    )
    is_badtide = hour_of_cycle >= (2 * hours_per_wet + hours_per_dry)
    is_water_active = is_first_wet | is_second_wet | is_badtide

    power_consumption = np.where(is_working_hour, total_consumption_rate, 0.0)

    power_wheel_production_rate = np.where(
        is_working_hour, power_wheels.count * power_wheels.power, 0.0
    )
    water_wheel_production_rate = water_wheels.count * water_wheels.power
    base_power_production = (
        np.where(is_water_active, water_wheel_production_rate, 0.0)
        + power_wheel_production_rate
    )

    # Pre-allocate to store all results for filtering
    all_power_prod = np.zeros((config.samples, total_hours))
    all_batt_charge = np.zeros((config.samples, total_hours))

    hours_empty_results = np.zeros(config.samples)
    final_surpluses = np.zeros(config.samples)

    for s in prange(config.samples):
        res = jit_stochastic_simulation(
            total_hours,
            base_power_production,
            power_consumption,
            large_windmills,
            windmills,
            total_battery_capacity,
        )
        all_power_prod[s] = res.power_production
        all_batt_charge[s] = res.battery_charge

        hours_empty_results[s] = np.sum(res.battery_charge <= 0)
        final_surpluses[s] = np.sum(res.power_production) - np.sum(power_consumption)

    # Find worst run index
    worst_idx = np.argmax(hours_empty_results)

    worst_sample = SimulationSample(
        power_production=all_power_prod[worst_idx].copy(),
        battery_charge=all_batt_charge[worst_idx].copy(),
    )

    aggregated_samples = AggregatedSamples(
        hours_empty_results=hours_empty_results,
        final_surpluses=final_surpluses,
        power_consumption=power_consumption,
    )

    return ParallelSimulationResult(
        worst_sample=worst_sample, aggregated_samples=aggregated_samples
    )


@njit
def jit_stochastic_simulation(
    total_hours: int,
    base_power_production: np.ndarray,
    power_consumption: np.ndarray,
    large_windmills: ProducerGroup,
    windmills: ProducerGroup,
    total_battery_capacity: float,
) -> SimulationSample:
    """Performs a single Monte Carlo simulation run, handling stochastic input generation and internal state transitions."""
    # Generate wind data inside the core for better cache locality
    max_segments = (total_hours // consts.WIND_DURATION_MIN_HOURS) + 1
    wind_durations = np.random.randint(
        consts.WIND_DURATION_MIN_HOURS,
        consts.WIND_DURATION_MAX_HOURS,
        size=max_segments,
    )
    wind_strengths = np.random.random(size=max_segments)

    # Optimized Wind production using expansion
    wind_strength_profile = np.repeat(wind_strengths, wind_durations)[:total_hours]

    large_wind_unit_prod = np.where(
        wind_strength_profile > consts.LARGE_WINDMILL_THRESHOLD,
        wind_strength_profile * large_windmills.power,
        0.0,
    )
    small_wind_unit_prod = np.where(
        wind_strength_profile > consts.WINDMILL_THRESHOLD,
        wind_strength_profile * windmills.power,
        0.0,
    )
    wind_production = (large_windmills.count * large_wind_unit_prod) + (
        windmills.count * small_wind_unit_prod
    )

    power_production = base_power_production + wind_production
    power_surplus = power_production - power_consumption

    battery_charge = np.zeros(total_hours)
    current_charge = total_battery_capacity / 2.0

    for i in range(total_hours):
        surplus = power_surplus[i]
        if surplus > 0:
            space_available = total_battery_capacity - current_charge
            energy_to_store = min(surplus, space_available)
            current_charge += energy_to_store
        else:
            deficit = -surplus
            energy_available = current_charge
            energy_from_battery = min(deficit, energy_available)
            current_charge -= energy_from_battery
        battery_charge[i] = current_charge

    return SimulationSample(
        power_production=power_production,
        battery_charge=battery_charge,
    )
