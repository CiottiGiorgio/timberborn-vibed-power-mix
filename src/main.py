import argparse
import os
import matplotlib.pyplot as plt
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from models import SimulationConfig, SimulationParams
from simulation import simulate_scenario, run_simulation_batch
from plots.power_plot import plot_power
from plots.energy_plot import plot_energy
from plots.surplus_plot import plot_surplus
from plots.battery_plot import plot_battery
from plots.empty_hours_plot import plot_empty_hours

def plot_simulation(data, run_empty_hours, total_runs):
    # Unpack data
    time_days = data['time_days']
    power_production = data['power_production']
    power_consumption = data['power_consumption']
    power_surplus = data['power_surplus']
    effective_surplus = data['effective_surplus']
    effective_deficit = data['effective_deficit']
    battery_charge = data['battery_charge']
    energy_production = data['energy_production']
    energy_consumption = data['energy_consumption']
    total_battery_capacity = data['total_battery_capacity']
    season_boundaries = data['season_boundaries']
    params = data['params']
    
    days = params.days
    water_wheels = params.water_wheels
    large_windmills = params.large_windmills
    windmills = params.windmills
    batteries = params.batteries
    battery_height = params.battery_height

    # Visualization
    # Always create 5 plots
    num_plots = 5
    fig, axes = plt.subplots(num_plots, 1, figsize=(12, 5 * num_plots), sharex=False)
    
    (ax1, ax2, ax3, ax4, ax5) = axes

    # Link x-axes for the first 4 plots
    ax2.sharex(ax1)
    ax3.sharex(ax1)
    ax4.sharex(ax1)
    
    # Hide tick labels for 1, 2, 3
    plt.setp(ax1.get_xticklabels(), visible=False)
    plt.setp(ax2.get_xticklabels(), visible=False)
    plt.setp(ax3.get_xticklabels(), visible=False)

    # Add vertical lines for season boundaries to all time-series plots
    for ax in [ax1, ax2, ax3, ax4]:
        for boundary_day, _ in season_boundaries:
            ax.axvline(x=boundary_day, color='#444444', linestyle='-', alpha=0.4, linewidth=1.5)

    # Plot 1: Power
    plot_power(ax1, time_days, power_production, power_consumption, season_boundaries, days, water_wheels, large_windmills, windmills)

    # Plot 2: Energy
    plot_energy(ax2, time_days, energy_production, energy_consumption, days)

    # Plot 3: Surplus Power (Effective)
    plot_surplus(ax3, time_days, power_surplus, effective_surplus, effective_deficit)

    # Plot 4: Battery Charge
    plot_battery(ax4, time_days, battery_charge, total_battery_capacity, batteries, battery_height)

    # Plot 5: Empty Battery Duration Distribution
    plot_empty_hours(ax5, run_empty_hours, total_runs)

    plt.tight_layout()
    plt.show()

def main():
    parser = argparse.ArgumentParser(description='Visualize power and energy profiles for an industrial complex.')
    parser.add_argument('--days', type=int, help='Number of days for the simulation')
    parser.add_argument('--working-hours', type=int, help='Number of working hours per day')
    parser.add_argument('--lumber-mills', type=int, help='Number of lumber mills')
    parser.add_argument('--gear-workshops', type=int, help='Number of gear workshops')
    parser.add_argument('--steel-factories', type=int, help='Number of steel factories')
    parser.add_argument('--wood-workshops', type=int, help='Number of wood workshops')
    parser.add_argument('--paper-mills', type=int, help='Number of paper mills')
    parser.add_argument('--printing-presses', type=int, help='Number of printing presses')
    parser.add_argument('--observatories', type=int, help='Number of observatories')
    parser.add_argument('--bot-part-factories', type=int, help='Number of bot part factories')
    parser.add_argument('--bot-assemblers', type=int, help='Number of bot assemblers')
    parser.add_argument('--explosives-factories', type=int, help='Number of explosives factories')
    parser.add_argument('--grillmists', type=int, help='Number of grillmists')
    parser.add_argument('--water-wheels', type=int, help='Number of water wheels')
    parser.add_argument('--large-windmills', type=int, help='Number of large windmills')
    parser.add_argument('--windmills', type=int, help='Number of windmills')
    parser.add_argument('--batteries', type=int, help='Number of gravity batteries')
    parser.add_argument('--battery-height', type=int, help='Height of gravity batteries in meters')
    parser.add_argument('--wet-season-days', type=int, help='Duration of wet season in days')
    parser.add_argument('--dry-season-days', type=int, help='Duration of dry season in days')
    parser.add_argument('--badtide-season-days', type=int, help='Duration of badtide season in days')
    parser.add_argument('--runs', type=int, default=1, help='Number of simulation runs')
    parser.add_argument('--machine-data', type=str, default='machines.json', help='Path to machine data JSON file')

    args = parser.parse_args()

    # Load machine data
    try:
        # Resolve path relative to script location if not absolute
        script_dir = os.path.dirname(os.path.abspath(__file__))
        machine_data_path = args.machine_data
        if not os.path.isabs(machine_data_path):
            machine_data_path = os.path.join(script_dir, machine_data_path)
            
        config = SimulationConfig.from_json_file(machine_data_path)
    except Exception as e:
        print(f"Error loading machine data: {e}")
        return

    # Create SimulationParams from args, filtering out None values to use defaults from the model
    # We only include keys that are present in SimulationParams fields
    param_fields = SimulationParams.model_fields.keys()
    params_dict = {k: v for k, v in vars(args).items() if v is not None and k in param_fields}
    
    params = SimulationParams(**params_dict)

    if args.runs > 1:
        run_empty_hours = []
        worst_run_data = None
        max_hours_empty = -1
        
        # Determine number of workers (cores)
        num_workers = os.cpu_count() or 1
        
        # Calculate runs per worker
        runs_per_worker = args.runs // num_workers
        remainder = args.runs % num_workers
        
        batch_sizes = [runs_per_worker + (1 if i < remainder else 0) for i in range(num_workers)]
        # Filter out 0s if runs < workers
        batch_sizes = [b for b in batch_sizes if b > 0]
        
        print(f"Running {args.runs} simulations across {len(batch_sizes)} processes...")

        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(run_simulation_batch, config, params, size) for size in batch_sizes]
            
            for future in futures:
                batch_results = future.result()
                for hours_empty, data in batch_results:
                    # Collect data for ALL runs
                    run_empty_hours.append(hours_empty)
                    
                    if hours_empty > max_hours_empty:
                        max_hours_empty = hours_empty
                        worst_run_data = data
                    
                    # If worst_run_data is still None, keep the last one just in case
                    if worst_run_data is None:
                        worst_run_data = data
        
        plot_simulation(worst_run_data, run_empty_hours, args.runs)

    else:
        data = simulate_scenario(config, params)
        # Even for a single run, we can pass the data to show the histogram (it will just be one bar)
        # But usually single run visualization focuses on the time series.
        # However, to be consistent with "Always show the fifth graph", let's pass it.
        # We need to calculate hours_empty for this single run to pass it.
        battery_after_day1 = data['battery_charge'][24:]
        hours_empty = np.sum(battery_after_day1 <= 0)
        plot_simulation(data, [hours_empty], 1)

if __name__ == "__main__":
    main()
