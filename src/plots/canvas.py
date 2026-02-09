import matplotlib.pyplot as plt
from models import SimulationResult
from plots.power_plot import plot_power
from plots.energy_plot import plot_energy
from plots.surplus_plot import plot_surplus
from plots.battery_plot import plot_battery
from plots.empty_hours_plot import plot_empty_hours_percentage
import consts


def plot_simulation(data: SimulationResult, run_empty_hours, total_runs):
    # Unpack data
    time_days = data.time_days
    power_production = data.power_production
    power_consumption = data.power_consumption
    power_surplus = data.power_surplus
    effective_balance = data.effective_balance
    battery_charge = data.battery_charge
    energy_production = data.energy_production
    energy_consumption = data.energy_consumption
    total_battery_capacity = data.total_battery_capacity
    season_boundaries = data.season_boundaries
    params = data.params
    total_cost = data.total_cost

    days = params.days
    power_wheels = params.energy_mix.power_wheels
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

    # Handle battery height display
    if isinstance(battery_height, list):
        avg_height = sum(battery_height) / len(battery_height) if battery_height else 0
        height_str = f"Avg: {avg_height:.1f}"
    else:
        height_str = str(battery_height)

    # Add Energy Mix Info Box
    mix_info = (
        f"Energy Mix:\n"
        f"  Power Wheels: {power_wheels}\n"
        f"  Water Wheels: {water_wheels}\n"
        f"  Large Windmills: {large_windmills}\n"
        f"  Windmills: {windmills}\n"
        f"  Batteries: {batteries} (Height: {height_str})"
    )

    # Place text box in top left corner
    fig.text(
        0.02,
        0.98,
        mix_info,
        fontsize=10,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
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
    plot_surplus(ax3, time_days, power_surplus, effective_balance)

    # Plot 4: Battery Charge
    plot_battery(
        ax4,
        time_days,
        battery_charge,
        total_battery_capacity,
        batteries,
        battery_height,
    )

    # Plot 5: Empty Battery Duration Distribution (Percentage)
    total_simulation_hours = days * consts.HOURS_PER_DAY
    plot_empty_hours_percentage(
        ax5, run_empty_hours, total_runs, total_simulation_hours
    )

    plt.tight_layout(
        rect=(0, 0.03, 1, 0.95)
    )  # Adjusted top margin to make room for text box
    plt.show()
