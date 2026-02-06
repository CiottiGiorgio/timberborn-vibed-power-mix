import numpy as np
from models import SimulationConfig, SimulationParams
import consts

def simulate_scenario(config: SimulationConfig, params: SimulationParams):
    
    machines = config.machines
    battery_info = config.battery

    # Helper to safely get power, defaulting to 0 if machine not in config (though validation ensures structure)
    # We assume the keys in JSON match the variable names we expect, or we map them.
    # The previous code assumed keys like 'lumber_mill'.
    
    def get_power(name):
        return machines[name].power

    # Consumption
    lumber_mill_consumption = get_power('lumber_mill')
    gear_workshop_consumption = get_power('gear_workshop')
    steel_consumption = get_power('steel_factory')
    wood_workshop_consumption = get_power('wood_workshop')
    paper_mill_consumption = get_power('paper_mill')
    printing_press_consumption = get_power('printing_press')
    observatory_consumption = get_power('observatory')
    bot_part_factory_consumption = get_power('bot_part_factory')
    bot_assembler_consumption = get_power('bot_assembler')
    explosives_factory_consumption = get_power('explosives_factory')
    grillmist_consumption = get_power('grillmist')
    
    # Production
    wheel_production = get_power('water_wheel')
    large_windmill_production = get_power('large_windmill')
    windmill_production = get_power('windmill')
    
    # Battery Constants
    base_capacity = battery_info.base_capacity
    capacity_per_height = battery_info.capacity_per_height
    battery_capacity = base_capacity + (params.battery_height * capacity_per_height)
    total_battery_capacity = params.batteries * battery_capacity

    total_hours = params.days * consts.HOURS_PER_DAY
    
    # Time axis (in days)
    time_hours = np.arange(total_hours)
    time_days = time_hours / consts.HOURS_PER_DAY

    # Power values (hp)
    total_consumption_rate = (
        (params.lumber_mills * lumber_mill_consumption) + 
        (params.gear_workshops * gear_workshop_consumption) + 
        (params.steel_factories * steel_consumption) +
        (params.wood_workshops * wood_workshop_consumption) +
        (params.paper_mills * paper_mill_consumption) +
        (params.printing_presses * printing_press_consumption) +
        (params.observatories * observatory_consumption) +
        (params.bot_part_factories * bot_part_factory_consumption) +
        (params.bot_assemblers * bot_assembler_consumption) +
        (params.explosives_factories * explosives_factory_consumption) +
        (params.grillmists * grillmist_consumption)
    )
    
    # Production components
    water_wheel_production_rate = params.water_wheels * wheel_production

    # Simulate wind variability based on Timberborn Wiki mechanics
    # Strength: 0-100%, Duration: 5-12 hours
    wind_strength = np.zeros(total_hours)
    curr_h = 0
    while curr_h < total_hours:
        duration = np.random.randint(consts.WIND_DURATION_MIN_HOURS, consts.WIND_DURATION_MAX_HOURS) 
        strength = np.random.random()       # 0.0 to 1.0 (0% to 100%)
        wind_strength[curr_h : curr_h + duration] = strength
        curr_h += duration
    wind_strength = wind_strength[:total_hours] # Trim to exact simulation length

    # Calculate Wind Production with Thresholds:
    # Large Windmills: 0 if strength <= 20%, else strength * 300
    # Small Windmills: 0 if strength <= 30%, else strength * 120
    large_wind_unit_prod = np.where(wind_strength > consts.LARGE_WINDMILL_THRESHOLD, wind_strength * large_windmill_production, 0)
    small_wind_unit_prod = np.where(wind_strength > consts.SMALL_WINDMILL_THRESHOLD, wind_strength * windmill_production, 0)
    wind_production = (params.large_windmills * large_wind_unit_prod) + (params.windmills * small_wind_unit_prod)

    # Calculate hour of the day for each time step
    hour_of_day = time_hours % consts.HOURS_PER_DAY

    # Season Logic
    # Cycle: Wet Season -> Dry Season -> Wet Season -> Badtide Season ...
    cycle_length_days = params.wet_season_days + params.dry_season_days + params.wet_season_days + params.badtide_season_days
    day_of_cycle = time_days % cycle_length_days
    
    # Determine current season
    is_first_wet = day_of_cycle < params.wet_season_days
    is_dry = (day_of_cycle >= params.wet_season_days) & (day_of_cycle < (params.wet_season_days + params.dry_season_days))
    is_second_wet = (day_of_cycle >= (params.wet_season_days + params.dry_season_days)) & (day_of_cycle < (2 * params.wet_season_days + params.dry_season_days))
    is_badtide = day_of_cycle >= (2 * params.wet_season_days + params.dry_season_days)
    
    # Water wheels run in Wet and Badtide seasons (for now)
    is_water_active = is_first_wet | is_second_wet | is_badtide
    
    # Production depends on season (Water Wheel runs only in wet/badtide season)
    # Windmills run all the time with variability
    power_production = np.where(is_water_active, water_wheel_production_rate, 0) + wind_production

    # Consumption depends on working hours
    # Factories are active only during the first working_hours hours of the day
    is_working_hour = hour_of_day < params.working_hours
    power_consumption = np.where(is_working_hour, total_consumption_rate, 0)

    # Calculate Surplus Power (Production - Consumption)
    power_surplus = power_production - power_consumption

    # Calculate Battery State of Charge and Effective Surplus/Deficit
    battery_charge = np.zeros(total_hours)
    effective_surplus = np.zeros(total_hours) # Surplus after charging battery
    effective_deficit = np.zeros(total_hours) # Deficit after discharging battery
    
    current_charge = total_battery_capacity / 2 # Start at 50% capacity

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
        if curr >= params.days: break
        season_boundaries.append((curr, "Dry"))
        curr += params.dry_season_days
        if curr >= params.days: break
        season_boundaries.append((curr, "Wet"))
        curr += params.wet_season_days
        if curr >= params.days: break
        season_boundaries.append((curr, "Badtide"))
        curr += params.badtide_season_days

    return {
        'time_days': time_days,
        'power_production': power_production,
        'power_consumption': power_consumption,
        'power_surplus': power_surplus,
        'effective_surplus': effective_surplus,
        'effective_deficit': effective_deficit,
        'battery_charge': battery_charge,
        'energy_production': energy_production,
        'energy_consumption': energy_consumption,
        'total_battery_capacity': total_battery_capacity,
        'season_boundaries': season_boundaries,
        'params': params
    }

def run_simulation_batch(config: SimulationConfig, params: SimulationParams, num_runs: int):
    """
    Runs a batch of simulations and returns the results.
    Returns a list of tuples: (hours_empty, simulation_data)
    """
    results = []
    for _ in range(num_runs):
        data = simulate_scenario(config, params)
        
        # Check if battery reached 0 after day 1 (24 hours)
        battery_after_day1 = data['battery_charge'][24:]
        
        # Count hours where battery is 0 or less
        hours_empty = np.sum(battery_after_day1 <= 0)
        
        results.append((hours_empty, data))
        
    return results
