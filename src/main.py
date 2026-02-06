import os
import matplotlib.pyplot as plt
import numpy as np
import click
from concurrent.futures import ProcessPoolExecutor
from models import SimulationConfig, SimulationParams, SimulationResult, FactoryParams, EnergyMixParams
from simulation import simulate_scenario, run_simulation_task
from plots.power_plot import plot_power
from plots.energy_plot import plot_energy
from plots.surplus_plot import plot_surplus
from plots.battery_plot import plot_battery
from plots.empty_hours_plot import plot_empty_hours
import consts


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


class IntOrIntList(click.ParamType):
    name = "int_or_int_list"

    def convert(self, value, param, ctx):
        if isinstance(value, int):
            return value
        if isinstance(value, list):
            return value
        
        # Try to parse as a single int
        try:
            return int(value)
        except ValueError:
            pass
            
        # Try to parse as a comma-separated list of ints
        try:
            return [int(x.strip()) for x in value.split(",")]
        except ValueError:
            self.fail(f"{value!r} is not a valid integer or comma-separated list of integers", param, ctx)


@click.command()
@click.option(
    "--days",
    type=int,
    default=consts.DEFAULT_DAYS,
    help="Number of days for the simulation",
)
@click.option(
    "--working-hours",
    type=int,
    default=consts.DEFAULT_WORKING_HOURS,
    help="Number of working hours per day",
)
@click.option(
    "--lumber-mills",
    type=int,
    default=consts.DEFAULT_LUMBER_MILLS,
    help="Number of lumber mills",
)
@click.option(
    "--gear-workshops",
    type=int,
    default=consts.DEFAULT_GEAR_WORKSHOPS,
    help="Number of gear workshops",
)
@click.option(
    "--steel-factories",
    type=int,
    default=consts.DEFAULT_STEEL_FACTORIES,
    help="Number of steel factories",
)
@click.option(
    "--wood-workshops",
    type=int,
    default=consts.DEFAULT_WOOD_WORKSHOPS,
    help="Number of wood workshops",
)
@click.option(
    "--paper-mills",
    type=int,
    default=consts.DEFAULT_PAPER_MILLS,
    help="Number of paper mills",
)
@click.option(
    "--printing-presses",
    type=int,
    default=consts.DEFAULT_PRINTING_PRESSES,
    help="Number of printing presses",
)
@click.option(
    "--observatories",
    type=int,
    default=consts.DEFAULT_OBSERVATORIES,
    help="Number of observatories",
)
@click.option(
    "--bot-part-factories",
    type=int,
    default=consts.DEFAULT_BOT_PART_FACTORIES,
    help="Number of bot part factories",
)
@click.option(
    "--bot-assemblers",
    type=int,
    default=consts.DEFAULT_BOT_ASSEMBLERS,
    help="Number of bot assemblers",
)
@click.option(
    "--explosives-factories",
    type=int,
    default=consts.DEFAULT_EXPLOSIVES_FACTORIES,
    help="Number of explosives factories",
)
@click.option(
    "--grillmists",
    type=int,
    default=consts.DEFAULT_GRILLMISTS,
    help="Number of grillmists",
)
@click.option(
    "--water-wheels",
    type=int,
    default=consts.DEFAULT_WATER_WHEELS,
    help="Number of water wheels",
)
@click.option(
    "--large-windmills",
    type=int,
    default=consts.DEFAULT_LARGE_WINDMILLS,
    help="Number of large windmills",
)
@click.option(
    "--windmills",
    type=int,
    default=consts.DEFAULT_WINDMILLS,
    help="Number of windmills",
)
@click.option(
    "--batteries",
    type=int,
    default=consts.DEFAULT_BATTERIES,
    help="Number of gravity batteries",
)
@click.option(
    "--battery-height",
    type=IntOrIntList(),
    default=consts.DEFAULT_BATTERY_HEIGHT,
    help="Height of gravity batteries in meters. Can be a single int or a comma-separated list of ints.",
)
@click.option(
    "--wet-season-days",
    type=int,
    default=consts.DEFAULT_WET_SEASON_DAYS,
    help="Duration of wet season in days",
)
@click.option(
    "--dry-season-days",
    type=int,
    default=consts.DEFAULT_DRY_SEASON_DAYS,
    help="Duration of dry season in days",
)
@click.option(
    "--badtide-season-days",
    type=int,
    default=consts.DEFAULT_BADTIDE_SEASON_DAYS,
    help="Duration of badtide season in days",
)
@click.option("--runs", type=int, default=1, help="Number of simulation runs")
def main(
    days,
    working_hours,
    lumber_mills,
    gear_workshops,
    steel_factories,
    wood_workshops,
    paper_mills,
    printing_presses,
    observatories,
    bot_part_factories,
    bot_assemblers,
    explosives_factories,
    grillmists,
    water_wheels,
    large_windmills,
    windmills,
    batteries,
    battery_height,
    wet_season_days,
    dry_season_days,
    badtide_season_days,
    runs,
):
    """Visualize power and energy profiles for an industrial complex."""

    # Load machine data
    try:
        # Resolve path relative to script location if not absolute
        script_dir = os.path.dirname(os.path.abspath(__file__))
        machine_data_path = "machines.json"
        if not os.path.isabs(machine_data_path):
            machine_data_path = os.path.join(script_dir, machine_data_path)

        config = SimulationConfig.from_json_file(machine_data_path)
    except Exception as e:
        print(f"Error loading machine data: {e}")
        return

    # Create SimulationParams from args
    factories = FactoryParams(
        lumber_mills=lumber_mills,
        gear_workshops=gear_workshops,
        steel_factories=steel_factories,
        wood_workshops=wood_workshops,
        paper_mills=paper_mills,
        printing_presses=printing_presses,
        observatories=observatories,
        bot_part_factories=bot_part_factories,
        bot_assemblers=bot_assemblers,
        explosives_factories=explosives_factories,
        grillmists=grillmists,
    )

    energy_mix = EnergyMixParams(
        water_wheels=water_wheels,
        large_windmills=large_windmills,
        windmills=windmills,
        batteries=batteries,
        battery_height=battery_height,
    )

    params = SimulationParams(
        days=days,
        working_hours=working_hours,
        wet_season_days=wet_season_days,
        dry_season_days=dry_season_days,
        badtide_season_days=badtide_season_days,
        factories=factories,
        energy_mix=energy_mix,
    )

    if runs > 1:
        run_empty_hours = []
        worst_run_data = None
        max_hours_empty = -1

        print(f"Running {runs} simulations...")

        # Let ProcessPoolExecutor handle worker count automatically (defaults to os.cpu_count())
        with ProcessPoolExecutor() as executor:
            # Submit individual tasks instead of batches
            futures = [
                executor.submit(run_simulation_task, config, params)
                for _ in range(runs)
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
        data = simulate_scenario(config, params)
        # Even for a single run, we can pass the data to show the histogram (it will just be one bar)
        # But usually single run visualization focuses on the time series.
        # However, to be consistent with "Always show the fifth graph", let's pass it.
        # We need to calculate hours_empty for this single run to pass it.
        battery_after_day1 = data.battery_charge[24:]
        hours_empty = np.sum(battery_after_day1 <= 0)
        plot_simulation(data, [hours_empty], 1)


if __name__ == "__main__":
    main()
