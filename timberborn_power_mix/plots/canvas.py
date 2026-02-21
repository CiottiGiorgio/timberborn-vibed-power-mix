import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from timberborn_power_mix.simulation.models import SimulationConfig
from timberborn_power_mix.structures import SimulationResult
from timberborn_power_mix.plots.power_plot import plot_power
from timberborn_power_mix.plots.energy_plot import plot_energy
from timberborn_power_mix.plots.surplus_plot import plot_surplus
from timberborn_power_mix.plots.battery_plot import plot_battery
from timberborn_power_mix.plots.empty_hours_plot import plot_empty_hours_percentage
from timberborn_power_mix.simulation import consts as sim_consts
from timberborn_power_mix.optimization.helpers import calculate_total_wood_cost
from timberborn_power_mix.simulation.helpers import (
    calculate_total_battery_capacity,
    calculate_season_boundaries,
)
from timberborn_power_mix.machines import ProducerName, BatteryName
from timberborn_power_mix.structures import ConfigName


def create_simulation_figure(config: SimulationConfig, res: SimulationResult) -> Figure:
    # Unpack data
    data = res.p95_sample
    run_empty_hours = res.aggregated_samples.hours_empty_results
    power_consumption = res.aggregated_samples.power_consumption

    days = getattr(config, ConfigName.DAYS)
    total_hours = days * sim_consts.HOURS_PER_DAY

    time_hours = np.arange(total_hours)
    time_days = time_hours / sim_consts.HOURS_PER_DAY

    power_production = data.power_production
    battery_charge = data.battery_charge

    # Recompute derived values
    # Cast to int64 to avoid uint32 overflow during subtraction
    power_surplus = power_production.astype(np.int64) - power_consumption.astype(
        np.int64
    )

    total_battery_capacity = calculate_total_battery_capacity(config.energy_mix)

    # Effective balance is the surplus that couldn't be absorbed by the battery
    # or the deficit that couldn't be covered by the battery.
    battery_charge_shifted = np.zeros_like(battery_charge, dtype=np.int64)
    battery_charge_shifted[0] = total_battery_capacity // 2  # Initial charge
    battery_charge_shifted[1:] = battery_charge[:-1]
    delta_charge = battery_charge.astype(np.int64) - battery_charge_shifted
    effective_balance = power_surplus - delta_charge

    # Recompute cumulative energy
    energy_production = np.cumsum(power_production)
    energy_consumption = np.cumsum(power_consumption)

    season_boundaries = calculate_season_boundaries(config)
    total_cost = calculate_total_wood_cost(config.energy_mix)

    power_wheels = getattr(config.energy_mix, ProducerName.POWER_WHEELS)
    water_wheels = getattr(config.energy_mix, ProducerName.WATER_WHEELS)
    large_windmills = getattr(config.energy_mix, ProducerName.LARGE_WINDMILLS)
    windmills = getattr(config.energy_mix, ProducerName.WINDMILLS)
    battery_heights = getattr(config.energy_mix, BatteryName.BATTERY_HEIGHTS)

    # Visualization
    # Always create 5 plots
    num_plots = 5
    fig, axes = plt.subplots(num_plots, 1, figsize=(12, 5 * num_plots), sharex=False)

    # Add title with total cost
    fig.suptitle(
        f"Simulation Results (Total Cost: {total_cost} logs)", fontsize=16, y=0.99
    )

    # Handle battery height display
    num_batteries = len(battery_heights)
    if num_batteries > 0:
        avg_height = sum(battery_heights) / num_batteries
        height_str = f"Avg: {avg_height:.1f}"
    else:
        height_str = "0"

    # Add Energy Mix Info Box
    mix_info = (
        f"Energy Mix:\n"
        f"  Power Wheels: {power_wheels}\n"
        f"  Water Wheels: {water_wheels}\n"
        f"  Large Windmills: {large_windmills}\n"
        f"  Windmills: {windmills}\n"
        f"  Batteries: {num_batteries} (Height: {height_str})"
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

    # Add Simulation Info Box
    sim_info = (
        f"Simulation Info:\n"
        f"  Days: {days}\n"
        f"  Working Hours: {getattr(config, ConfigName.WORKING_HOURS)}\n"
        f"  Wet Season: {getattr(config, ConfigName.WET_DAYS)} days\n"
        f"  Dry Season: {getattr(config, ConfigName.DRY_DAYS)} days\n"
        f"  Badtide Season: {getattr(config, ConfigName.BADTIDE_DAYS)} days\n"
        f"  Samples: {getattr(config, ConfigName.SAMPLES)}"
    )

    # Place text box in top right corner
    fig.text(
        0.98,
        0.98,
        sim_info,
        fontsize=10,
        verticalalignment="top",
        horizontalalignment="right",
        bbox=dict(boxstyle="round", facecolor="lightblue", alpha=0.5),
    )

    # Add Disclaimer about p95 Case
    disclaimer_text = (
        "Note: The time-series plots (1-4) show the 'p95' sample\n"
        "(the one representing the 95th percentile of working time spent with an empty battery)."
    )
    fig.text(
        0.5,
        0.95,
        disclaimer_text,
        fontsize=10,
        ha="center",
        va="top",
        style="italic",
        color="#555555",
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
        for i, (start_hour, label) in enumerate(season_boundaries):
            start_day = start_hour / sim_consts.HOURS_PER_DAY
            # Vertical line
            ax.axvline(
                x=start_day, color="#444444", linestyle="-", alpha=0.4, linewidth=1.5
            )

            # Label
            end_hour = (
                season_boundaries[i + 1][0]
                if i + 1 < len(season_boundaries)
                else days * sim_consts.HOURS_PER_DAY
            )
            end_day = end_hour / sim_consts.HOURS_PER_DAY
            mid_point = (start_day + end_day) / 2
            if mid_point < days:
                ax.text(
                    mid_point,
                    1.01,
                    label,
                    transform=ax.get_xaxis_transform(),
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    fontweight="bold",
                    alpha=0.6,
                )

    # Plot 1: Power
    plot_power(
        ax1,
        time_days,
        power_production,
        power_consumption,
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
        battery_heights,
    )

    # Plot 5: Empty Battery Duration Distribution (Percentage)
    working_hours_per_day = getattr(config, ConfigName.WORKING_HOURS)
    total_working_hours = days * working_hours_per_day
    plot_empty_hours_percentage(
        ax5,
        run_empty_hours,
        total_working_hours,
    )

    plt.tight_layout(rect=(0, 0.03, 1, 0.95))

    return fig
