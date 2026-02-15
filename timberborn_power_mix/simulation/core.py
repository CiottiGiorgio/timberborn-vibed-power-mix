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

    # Seeding externally as requested
    np.random.seed(config.seed)

    return jit_parallel_simulation(
        config.to_parallel_config,
        total_consumption_rate,
        ProducerGroup(num_large_windmills, large_windmill_spec.power),
        ProducerGroup(num_windmills, windmill_spec.power),
        ProducerGroup(num_power_wheels, power_wheel_spec.power),
        ProducerGroup(num_water_wheels, wheel_spec.power),
        total_battery_capacity,
    )


@njit(parallel=True, cache=True)
def jit_parallel_simulation(
    config: ParallelSimulationConfig,
    total_consumption_rate: int,
    large_windmills: ProducerGroup,
    windmills: ProducerGroup,
    power_wheels: ProducerGroup,
    water_wheels: ProducerGroup,
    total_battery_capacity: float,
) -> SimulationResult:
    """Manages parallel simulation execution, including heavy memory allocation and caching of shared read-only arrays."""
    total_hours = config.days * consts.HOURS_PER_DAY

    # Generate the array inside the jitted function to avoid memory transfer overhead
    sample_seeds = np.random.randint(0, 2**31 - 1, size=config.samples)

    base_power_production = sim_helpers.calculate_base_power_production(
        total_hours,
        config.working_hours,
        config.wet_days,
        config.dry_days,
        config.badtide_days,
        power_wheels,
        water_wheels,
    )

    # Pre-calculate static profiles
    time_hours = np.arange(total_hours)
    hour_of_day = time_hours % consts.HOURS_PER_DAY
    is_working_hour = hour_of_day < config.working_hours
    power_consumption = np.where(is_working_hour, total_consumption_rate, 0.0)

    hours_empty_results = np.zeros(config.samples)

    for s in prange(config.samples):
        res = jit_stochastic_simulation(
            sample_seeds[s],
            base_power_production,
            power_consumption,
            large_windmills,
            windmills,
            total_battery_capacity,
            total_hours,
        )
        hours_empty_results[s] = np.sum(res.battery_charge <= 0)

    worst_idx = np.argmax(hours_empty_results)

    # Replay the worst run to get the full sample data without storing all samples in memory
    worst_sample = jit_stochastic_simulation(
        sample_seeds[worst_idx],
        base_power_production,
        power_consumption,
        large_windmills,
        windmills,
        total_battery_capacity,
        total_hours,
    )

    aggregated_samples = AggregatedSamples(
        power_consumption=power_consumption,
        hours_empty_results=hours_empty_results,
    )

    return SimulationResult(
        worst_sample=worst_sample, aggregated_samples=aggregated_samples
    )


@njit
def jit_stochastic_simulation(
    seed: int,
    base_power_production: np.ndarray,
    power_consumption: np.ndarray,
    large_windmills: ProducerGroup,
    windmills: ProducerGroup,
    total_battery_capacity: float,
    total_hours: int,
) -> SimulationSample:
    """Performs a single Monte Carlo simulation run, handling stochastic input generation and internal state transitions."""
    np.random.seed(seed)

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
