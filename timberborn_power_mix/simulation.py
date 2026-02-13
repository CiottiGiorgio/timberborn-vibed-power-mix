import numpy as np
from numpy.random import Generator
from timberborn_power_mix.models import SimulationParams, SimulationResult
from timberborn_power_mix.machines import (
    PRODUCER_DATABASE,
    battery_capacity,
    battery_cost,
    FACTORY_DATABASE, ProducerName, BatteryName,
)
from timberborn_power_mix import consts


def simulate_scenario(params: SimulationParams, _rng: Generator) -> SimulationResult:
    # Consumption
    total_consumption_rate = 0

    # Iterate over all consumers in MachineDatabase
    for name, spec in FACTORY_DATABASE.items():
        count = getattr(params.factories, name)
        total_consumption_rate += count * spec.power

    # Production
    wheel_spec = PRODUCER_DATABASE[ProducerName.WATER_WHEEL]
    windmill_spec = PRODUCER_DATABASE[ProducerName.WINDMILL]
    large_windmill_spec = PRODUCER_DATABASE[ProducerName.LARGE_WINDMILL]
    power_wheel_spec = PRODUCER_DATABASE[ProducerName.POWER_WHEEL]

    wheel_production = wheel_spec.power
    windmill_production = windmill_spec.power
    large_windmill_production = large_windmill_spec.power
    power_wheel_production = power_wheel_spec.power

    wheel_cost = wheel_spec.cost
    windmill_cost = windmill_spec.cost
    large_windmill_cost = large_windmill_spec.cost
    power_wheel_cost = power_wheel_spec.cost

    # Get counts from params.energy_mix
    num_batteries = getattr(params.energy_mix, BatteryName.BATTERY)
    num_water_wheels = getattr(params.energy_mix, ProducerName.WATER_WHEEL)
    num_windmills = getattr(params.energy_mix, ProducerName.WINDMILL)
    num_large_windmills = getattr(params.energy_mix, ProducerName.LARGE_WINDMILL)
    num_power_wheels = getattr(params.energy_mix, ProducerName.POWER_WHEEL)

    # Handle battery height (int or float)
    battery_height = getattr(params.energy_mix, BatteryName.BATTER_HEIGHT)

    # Calculate total capacity and cost
    # We assume all batteries have the average height
    capacity_per_battery = battery_capacity(battery_height)
    cost_per_battery = battery_cost(battery_height)

    total_battery_capacity = num_batteries * capacity_per_battery
    total_battery_cost = num_batteries * cost_per_battery

    # Total Cost Calculation
    total_cost = (
        (num_power_wheels * power_wheel_cost)
        + (num_water_wheels * wheel_cost)
        + (num_large_windmills * large_windmill_cost)
        + (num_windmills * windmill_cost)
        + total_battery_cost
    )

    total_hours = params.days * consts.HOURS_PER_DAY

    # Time axis (in days)
    time_hours = np.arange(total_hours)
    time_days = time_hours / consts.HOURS_PER_DAY

    # Production components
    water_wheel_production_rate = num_water_wheels * wheel_production

    # Simulate wind variability based on Timberborn Wiki mechanics
    # Strength: 0-100%, Duration: 5-12 hours
    wind_strength = np.zeros(total_hours)
    curr_h = 0

    while curr_h < total_hours:
        duration = _rng.integers(
            consts.WIND_DURATION_MIN_HOURS, consts.WIND_DURATION_MAX_HOURS
        )
        strength = _rng.random()  # 0.0 to 1.0 (0% to 100%)
        wind_strength[curr_h : curr_h + duration] = strength
        curr_h += duration
    wind_strength = wind_strength[:total_hours]  # Trim to exact simulation length

    # Calculate Wind Production with Thresholds:
    # Large Windmills: 0 if strength <= 20%, else strength * 300
    # Small Windmills: 0 if strength <= 30%, else strength * 120
    large_wind_unit_prod = np.where(
        wind_strength > consts.LARGE_WINDMILL_THRESHOLD,
        wind_strength * large_windmill_production,
        0,
    )
    small_wind_unit_prod = np.where(
        wind_strength > consts.SMALL_WINDMILL_THRESHOLD,
        wind_strength * windmill_production,
        0,
    )
    wind_production = (num_large_windmills * large_wind_unit_prod) + (
        num_windmills * small_wind_unit_prod
    )

    # Calculate hour of the day for each time step
    hour_of_day = time_hours % consts.HOURS_PER_DAY

    # Season Logic
    # Cycle: Wet Season -> Dry Season -> Wet Season -> Badtide Season ...
    cycle_length_days = (
        params.wet_season_days
        + params.dry_season_days
        + params.wet_season_days
        + params.badtide_season_days
    )
    day_of_cycle = time_days % cycle_length_days

    # Determine current season
    is_first_wet = day_of_cycle < params.wet_season_days
    is_second_wet = (
        day_of_cycle >= (params.wet_season_days + params.dry_season_days)
    ) & (day_of_cycle < (2 * params.wet_season_days + params.dry_season_days))
    is_badtide = day_of_cycle >= (2 * params.wet_season_days + params.dry_season_days)

    # Water wheels run in Wet and Badtide seasons (for now)
    is_water_active = is_first_wet | is_second_wet | is_badtide

    # Power wheels run only during working hours
    is_working_hour = hour_of_day < params.working_hours
    power_wheel_production_rate = np.where(
        is_working_hour, num_power_wheels * power_wheel_production, 0
    )

    # Production depends on season (Water Wheel runs only in wet/badtide season)
    # Windmills run all the time with variability
    power_production = (
        np.where(is_water_active, water_wheel_production_rate, 0)
        + wind_production
        + power_wheel_production_rate
    )

    # Consumption depends on working hours
    # Factories are active only during the first working_hours hours of the day
    power_consumption = np.where(is_working_hour, total_consumption_rate, 0)

    # Calculate Surplus Power (Production - Consumption)
    power_surplus = power_production - power_consumption

    # Calculate Battery State of Charge and Effective Surplus/Deficit
    battery_charge = np.zeros(total_hours)
    effective_balance = np.zeros(
        total_hours
    )  # Positive for surplus, negative for deficit

    current_charge = total_battery_capacity / 2  # Start at 50% capacity

    for i in range(total_hours):
        surplus = power_surplus[i]

        if surplus > 0:
            # Charging
            space_available = total_battery_capacity - current_charge
            energy_to_store = min(surplus, space_available)

            current_charge += energy_to_store
            # Remaining surplus that couldn't be stored
            effective_balance[i] = surplus - energy_to_store

        else:
            # Discharging (surplus is negative, so it's a deficit)
            deficit = -surplus
            energy_available = current_charge
            energy_from_battery = min(deficit, energy_available)

            current_charge -= energy_from_battery
            # Remaining deficit that couldn't be covered (negative value)
            effective_balance[i] = -(deficit - energy_from_battery)

        battery_charge[i] = current_charge

    # Calculate Energy (hph) - Cumulative sum of power over time (dt = 1 hour)
    energy_consumption = np.cumsum(power_consumption)
    energy_production = np.cumsum(power_production)

    # Calculate season transition points for vertical lines and labels
    season_boundaries = []
    curr = 0
    while curr < params.days:
        season_boundaries.append((curr, "Wet"))
        curr += params.wet_season_days
        if curr >= params.days:
            break
        season_boundaries.append((curr, "Dry"))
        curr += params.dry_season_days
        if curr >= params.days:
            break
        season_boundaries.append((curr, "Wet"))
        curr += params.wet_season_days
        if curr >= params.days:
            break
        season_boundaries.append((curr, "Badtide"))
        curr += params.badtide_season_days

    return SimulationResult(
        time_days=time_days,
        power_production=power_production,
        power_consumption=power_consumption,
        power_surplus=power_surplus,
        effective_balance=effective_balance,
        battery_charge=battery_charge,
        energy_production=energy_production,
        energy_consumption=energy_consumption,
        total_battery_capacity=total_battery_capacity,
        season_boundaries=season_boundaries,
        params=params,
        total_cost=total_cost,
    )


def run_simulation_task(params: SimulationParams, _rng: Generator):
    """
    Runs a single simulation and returns the results.
    Returns a tuple: (hours_empty, simulation_data)
    """
    data = simulate_scenario(params, _rng)

    # Count hours where battery is 0 or less
    hours_empty = np.sum(data.battery_charge <= 0)

    return hours_empty, data


def run_simulation_batch(params: SimulationParams, samples: int, rng_service):
    """
    Runs a batch of simulations and returns a list of results.
    """
    # Get a new generator for this batch
    _rng = rng_service.get_generator()

    results = []
    for _ in range(samples):
        results.append(run_simulation_task(params, _rng))
    return results
