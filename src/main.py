import argparse
import os
import matplotlib.pyplot as plt
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from models import SimulationConfig, SimulationParams
from simulation import simulate_scenario, run_simulation_batch

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
    production_label = f'Production ({water_wheels} Wheels, {large_windmills} L.Wind, {windmills} Wind)'
    ax1.plot(time_days, power_production, label=production_label, color='#1f77b4', linewidth=1, alpha=0.8)
    consumption_label = 'Total Factory Consumption'
    ax1.plot(time_days, power_consumption, label=consumption_label, color='#ff7f0e', linewidth=2)
    
    ax1.set_ylabel('Power (hp)')
    ax1.set_title(f'Power Profile ({days} Days)', pad=35)
    ax1.legend(loc='upper right')
    ax1.grid(True, linestyle='--', alpha=0.7)
    
    # Add season labels at the top of the first plot
    for i, (start_day, label) in enumerate(season_boundaries):
        end_day = season_boundaries[i+1][0] if i+1 < len(season_boundaries) else days
        mid_point = (start_day + end_day) / 2
        if mid_point < days:
            ax1.text(mid_point, 1.01, label, transform=ax1.get_xaxis_transform(), 
                     ha='center', va='bottom', fontsize=10, fontweight='bold', alpha=0.6)

    # Plot 2: Energy
    ax2.plot(time_days, energy_production, label='Cumulative Energy Produced', color='#1f77b4', linewidth=2)
    ax2.plot(time_days, energy_consumption, label='Cumulative Energy Consumed', color='#ff7f0e', linewidth=2)

    ax2.set_ylabel('Energy (hph)')
    ax2.set_title(f'Energy Profile ({days} Days)')
    ax2.legend(loc='upper right')
    ax2.grid(True, linestyle='--', alpha=0.7)

    # Plot 3: Surplus Power (Effective)
    ax3.plot(time_days, power_surplus, color='#D3D3D3', linewidth=1, alpha=0.4, label='Original Surplus/Deficit')
    ax3.fill_between(time_days, power_surplus, 0, color='#D3D3D3', alpha=0.15)
    ax3.fill_between(time_days, effective_surplus, 0, where=(effective_surplus > 0), facecolor='green', alpha=0.6, label='Unstored Surplus')
    ax3.fill_between(time_days, effective_deficit, 0, where=(effective_deficit < 0), facecolor='red', alpha=0.6, label='Uncovered Deficit')
    
    ax3.axhline(0, color='black', linewidth=1, linestyle='-')
    ax3.set_ylabel('Effective Surplus (hp)')
    ax3.set_title(f'Power Surplus Profile (After Battery Buffering)')
    ax3.legend(loc='upper right')
    ax3.grid(True, linestyle='--', alpha=0.7)

    # Plot 4: Battery Charge
    ax4.plot(time_days, battery_charge, label=f'Battery Charge (Max: {total_battery_capacity} hph)', color='#9467bd', linewidth=2)
    ax4.axhline(total_battery_capacity, color='black', linestyle='--', label='Max Capacity')
    ax4.fill_between(time_days, battery_charge, 0, color='#9467bd', alpha=0.3)
    
    ax4.set_ylabel('Stored Energy (hph)')
    ax4.set_xlabel('Time (days)')
    ax4.set_title(f'Battery Status ({batteries} Batteries @ {battery_height}m)')
    ax4.legend(loc='upper right')
    ax4.grid(True, linestyle='--', alpha=0.7)

    # Plot 5: Empty Battery Duration Distribution
    
    # Create bins of 2 hours width, including the first bin (0-2)
    if run_empty_hours:
        max_val = max(run_empty_hours)
        # Ensure we have at least one bin if max_val is 0
        if max_val == 0:
            bins = np.arange(0, 4, 2) # 0-2, 2-4
        else:
            bins = np.arange(0, max_val + 4, 2)
            
        # Plot the histogram
        n, bins, patches = ax5.hist(run_empty_hours, bins=bins, color='red', alpha=0.7, edgecolor='black')
        
        # Style the first bar (0-2 hours) to be grey with diagonal hatching
        if len(patches) > 0:
            patches[0].set_facecolor('lightgrey')
            patches[0].set_hatch('//')
            patches[0].set_edgecolor('black')
        
        # Scale the y-axis ignoring the first bin (0-2 hours)
        # The first bin corresponds to n[0]
        if len(n) > 1:
            max_freq_excluding_first = max(n[1:])
            # Add some headroom (e.g., 10%)
            ax5.set_ylim(0, max_freq_excluding_first * 1.1)
            
        # Calculate percentiles and mean for ALL runs
        if run_empty_hours:
            p5 = np.percentile(run_empty_hours, 5)
            p50 = np.percentile(run_empty_hours, 50)
            p95 = np.percentile(run_empty_hours, 95)
            mean_val = np.mean(run_empty_hours)
            
            # Add vertical lines for percentiles
            ax5.axvline(p5, color='blue', linestyle='--', linewidth=1.5, label=f'5th % ({p5:.1f}h)')
            ax5.axvline(p50, color='green', linestyle='--', linewidth=1.5, label=f'Median ({p50:.1f}h)')
            ax5.axvline(mean_val, color='orange', linestyle='-', linewidth=1.5, label=f'Mean ({mean_val:.1f}h)')
            ax5.axvline(p95, color='purple', linestyle='--', linewidth=1.5, label=f'95th % ({p95:.1f}h)')
            ax5.legend(loc='upper right')
    
    ax5.set_title(f'Distribution of Time Spent with Empty Battery ({total_runs} Runs)')
    ax5.set_xlabel('Hours with Empty Battery')
    ax5.set_ylabel('Number of Runs')
    ax5.grid(True, linestyle='--', alpha=0.5)

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
