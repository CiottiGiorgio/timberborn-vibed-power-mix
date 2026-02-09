import numpy as np
from models import SimulationParams, SimulationResult
from machines import MachineDatabase
import consts


def simulate_scenario(params: SimulationParams) -> SimulationResult:

    # Helper to safely get power
    def get_power(name):
        if hasattr(MachineDatabase, name):
            return getattr(MachineDatabase, name).power
        return 0

    def get_cost(name):
        if hasattr(MachineDatabase, name):
            return getattr(MachineDatabase, name).cost
        return 0

    # Consumption
    total_consumption_rate = 0

    # Iterate over all consumers in MachineDatabase
    for name, spec in MachineDatabase.iter_consumers():
        count = 0
        # Check if count is provided in params.factories
        if hasattr(params.factories, name):
            count = getattr(params.factories, name)
        elif name in params.factories.counts:
            count = params.factories.counts[name]
        else:
            # Fallback to 0 if not specified
            count = 0

        total_consumption_rate += count * spec.power

    # Production
    wheel_production = MachineDatabase.water_wheel.power
    large_windmill_production = MachineDatabase.large_windmill.power
    windmill_production = MachineDatabase.windmill.power

    wheel_cost = MachineDatabase.water_wheel.cost
    large_windmill_cost = MachineDatabase.large_windmill.cost
    windmill_cost = MachineDatabase.windmill.cost

    # Get counts from params.energy_mix
    num_water_wheels = params.energy_mix.water_wheels
    num_large_windmills = params.energy_mix.large_windmills
    num_windmills = params.energy_mix.windmills
    num_batteries = params.energy_mix.batteries

    # Battery Constants
    battery_info = MachineDatabase.battery

    # Handle battery height (int or list of ints)
    battery_heights = params.energy_mix.battery_height
    if isinstance(battery_heights, int):
        # If it's a single int, treat it as a list of identical heights
        battery_heights = [battery_heights] * num_batteries

    # Calculate total capacity and cost by summing over individual batteries
    total_battery_capacity = 0
    total_battery_cost = 0

    for h in battery_heights:
        capacity = battery_info.calculate_capacity(h)
        cost = battery_info.calculate_cost(h)
        total_battery_capacity += capacity
        total_battery_cost += cost

    # Total Cost Calculation
    total_cost = (
        (num_water_wheels * wheel_cost)
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
        duration = np.random.randint(
            consts.WIND_DURATION_MIN_HOURS, consts.WIND_DURATION_MAX_HOURS
        )
        strength = np.random.random()  # 0.0 to 1.0 (0% to 100%)
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

    # Production depends on season (Water Wheel runs only in wet/badtide season)
    # Windmills run all the time with variability
    power_production = (
        np.where(is_water_active, water_wheel_production_rate, 0) + wind_production
    )

    # Consumption depends on working hours
    # Factories are active only during the first working_hours hours of the day
    is_working_hour = hour_of_day < params.working_hours
    power_consumption = np.where(is_working_hour, total_consumption_rate, 0)

    # Calculate Surplus Power (Production - Consumption)
    power_surplus = power_production - power_consumption

    # Calculate Battery State of Charge and Effective Surplus/Deficit
    battery_charge = np.zeros(total_hours)
    effective_surplus = np.zeros(total_hours)  # Surplus after charging battery
    effective_deficit = np.zeros(total_hours)  # Deficit after discharging battery

    current_charge = total_battery_capacity / 2  # Start at 50% capacity

    for i in range(total_hours):
        surplus = power_surplus[i]

        if surplus > 0:
            # Charging
            space_available = total_battery_capacity - current_charge
            energy_to_store = min(surplus, space_available)

            current_charge += energy_to_store
            # Remaining surplus that couldn't be stored
            effective_surplus[i] = surplus - energy_to_store
            effective_deficit[i] = 0

        else:
            # Discharging (surplus is negative, so it's a deficit)
            deficit = -surplus
            energy_available = current_charge
            energy_from_battery = min(deficit, energy_available)

            current_charge -= energy_from_battery
            # Remaining deficit that couldn't be covered
            effective_deficit[i] = -(deficit - energy_from_battery)
            effective_surplus[i] = 0

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
        effective_surplus=effective_surplus,
        effective_deficit=effective_deficit,
        battery_charge=battery_charge,
        energy_production=energy_production,
        energy_consumption=energy_consumption,
        total_battery_capacity=total_battery_capacity,
        season_boundaries=season_boundaries,
        params=params,
        total_cost=total_cost,
    )


def run_simulation_task(params: SimulationParams):
    """
    Runs a single simulation and returns the results.
    Returns a tuple: (hours_empty, simulation_data)
    """
    data = simulate_scenario(params)

    # Count hours where battery is 0 or less
    hours_empty = np.sum(data.battery_charge <= 0)

    return hours_empty, data


def run_simulation_batch(params: SimulationParams, runs: int):
    """
    Runs a batch of simulations and returns a list of results.
    """
    results = []
    for _ in range(runs):
        results.append(run_simulation_task(params))
    return results
