import matplotlib.pyplot as plt
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from models import SimulationResult
from simulation import simulate_scenario, run_simulation_task
from plots.power_plot import plot_power
from plots.energy_plot import plot_energy
from plots.surplus_plot import plot_surplus
from plots.battery_plot import plot_battery
from plots.empty_hours_plot import plot_empty_hours
from cli import create_cli, parse_params


def plot_simulation(data: SimulationResult, run_empty_hours, total_runs):
    # Unpack data
    time_days = data.time_days
    power_production = data.power_production
    power_consumption = data.power_consumption
    power_surplus = data.power_surplus
    effective_surplus = data.effective_surplus
    effective_deficit = data.effective_deficit
    battery_charge = data.battery_charge
    energy_production = data.energy_production
    energy_consumption = data.energy_consumption
    total_battery_capacity = data.total_battery_capacity
    season_boundaries = data.season_boundaries
    params = data.params
    total_cost = data.total_cost

    days = params.days
    water_wheels = params.energy_mix.water_wheels
    large_windmills = params.energy_mix.large_windmills
    windmills = params.energy_mix.windmills
    batteries = params.energy_mix.batteries
    battery_height = params.energy_mix.battery_height

    # Visualization
    # Always create 5 plots
    num_plots = 5
    fig, axes = plt.subplots(num_plots, 1, figsize=(12, 5 * num_plots), sharex=False)

    # Add title with total cost
    fig.suptitle(
        f"Simulation Results (Total Cost: {total_cost} logs)", fontsize=16, y=0.99
    )

    ax1, ax2, ax3, ax4, ax5 = axes

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
            ax.axvline(
                x=boundary_day, color="#444444", linestyle="-", alpha=0.4, linewidth=1.5
            )

    # Plot 1: Power
    plot_power(
        ax1,
        time_days,
        power_production,
        power_consumption,
        season_boundaries,
        days,
        water_wheels,
        large_windmills,
        windmills,
    )

    # Plot 2: Energy
    plot_energy(ax2, time_days, energy_production, energy_consumption, days)

    # Plot 3: Surplus Power (Effective)
    plot_surplus(ax3, time_days, power_surplus, effective_surplus, effective_deficit)

    # Plot 4: Battery Charge
    plot_battery(
        ax4,
        time_days,
        battery_charge,
        total_battery_capacity,
        batteries,
        battery_height,
    )

    # Plot 5: Empty Battery Duration Distribution
    plot_empty_hours(ax5, run_empty_hours, total_runs)

    plt.tight_layout(rect=(0, 0.03, 1, 0.97))
    plt.show()


def main(**kwargs):
    """Visualize power and energy profiles for an industrial complex."""

    params = parse_params(**kwargs)
    runs = kwargs.get("runs", 1)

    if runs > 1:
        run_empty_hours = []
        worst_run_data = None
        max_hours_empty = -1

        print(f"Running {runs} simulations...")

        # Let ProcessPoolExecutor handle worker count automatically (defaults to os.cpu_count())
        with ProcessPoolExecutor() as executor:
            # Submit individual tasks instead of batches
            futures = [
                executor.submit(run_simulation_task, params) for _ in range(runs)
            ]

            for future in futures:
                hours_empty, data = future.result()

                # Collect data for ALL runs
                run_empty_hours.append(hours_empty)

                if hours_empty > max_hours_empty:
                    max_hours_empty = hours_empty
                    worst_run_data = data

                # If worst_run_data is still None, keep the last one just in case
                if worst_run_data is None:
                    worst_run_data = data

        plot_simulation(worst_run_data, run_empty_hours, runs)

    else:
        data = simulate_scenario(params)
        # Even for a single run, we can pass the data to show the histogram (it will just be one bar)
        # But usually single run visualization focuses on the time series.
        # However, to be consistent with "Always show the fifth graph", let's pass it.
        # We need to calculate hours_empty for this single run to pass it.
        battery_after_day1 = data.battery_charge[24:]
        hours_empty = np.sum(battery_after_day1 <= 0)
        plot_simulation(data, [hours_empty], 1)


if __name__ == "__main__":
    cli = create_cli(main)
    cli()
