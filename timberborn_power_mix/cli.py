import click
from . import consts
from .machines import iter_consumers, ALL_MACHINES
from .models import FactoryParams, EnergyMixParams, SimulationParams


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
            self.fail(
                f"{value!r} is not a valid integer or comma-separated list of integers",
                param,
                ctx,
            )


def add_common_params(func):
    """Decorator to add common simulation parameters to a click command."""

    # Consumers
    for name, spec in iter_consumers():
        func = click.option(
            f"--{name.replace('_', '-')}",
            type=int,
            default=0,
            help=f"Number of {name.replace('_', ' ')}s",
        )(func)

    # Standard options
    func = click.option(
        "--badtide-season-days",
        type=int,
        default=consts.DEFAULT_BADTIDE_SEASON_DAYS,
        help="Duration of badtide season in days",
    )(func)

    func = click.option(
        "--dry-season-days",
        type=int,
        default=consts.DEFAULT_DRY_SEASON_DAYS,
        help="Duration of dry season in days",
    )(func)

    func = click.option(
        "--wet-season-days",
        type=int,
        default=consts.DEFAULT_WET_SEASON_DAYS,
        help="Duration of wet season in days",
    )(func)

    func = click.option(
        "--working-hours",
        type=int,
        default=consts.DEFAULT_WORKING_HOURS,
        help="Number of working hours per day",
    )(func)

    func = click.option(
        "--days",
        type=int,
        default=consts.DEFAULT_DAYS,
        help="Number of days for the simulation",
    )(func)

    # Common simulation parameter
    func = click.option(
        "--samples-per-sim",
        type=int,
        default=consts.DEFAULT_SAMPLES_PER_SIM,
        help="Number of simulation samples per configuration",
    )(func)

    return func


def add_energy_mix_params(func):
    """Decorator to add energy mix parameters (for run command)."""

    # Battery Height
    func = click.option(
        "--battery-height",
        type=IntOrIntList(),
        default=0,
        help="Height of gravity batteries in meters. Can be a single int or a comma-separated list of ints.",
    )(func)

    # Batteries
    func = click.option(
        "--battery",
        type=int,
        default=0,
        help="Number of gravity batteries",
    )(func)

    # Producers
    if "windmill" in ALL_MACHINES:
        func = click.option(
            "--windmill",
            type=int,
            default=0,
            help="Number of windmills",
        )(func)

    if "large_windmill" in ALL_MACHINES:
        func = click.option(
            "--large-windmill",
            type=int,
            default=0,
            help="Number of large windmills",
        )(func)

    if "water_wheel" in ALL_MACHINES:
        func = click.option(
            "--water-wheel",
            type=int,
            default=0,
            help="Number of water wheels",
        )(func)

    if "power_wheel" in ALL_MACHINES:
        func = click.option(
            "--power-wheel",
            type=int,
            default=0,
            help="Number of power wheels",
        )(func)

    return func


def create_cli(run_callback, optimize_callback):
    @click.group()
    def cli():
        """Timberborn Power Mix Simulation and Optimization Tool."""
        pass

    @cli.command(name="run")
    @add_energy_mix_params
    @add_common_params
    def run_cmd(**kwargs):
        """Run a simulation with the specified parameters."""
        run_callback(**kwargs)

    @cli.command(name="optimize")
    @click.option(
        "--iterations",
        type=int,
        default=consts.DEFAULT_OPTIMIZATION_ITERATIONS,
        help="Number of optimization iterations",
    )
    @add_common_params
    def optimize_cmd(**kwargs):
        """Optimize the energy mix for the specified parameters."""
        optimize_callback(**kwargs)

    return cli


def parse_params(**kwargs) -> SimulationParams:
    # Extract standard args
    days = kwargs.get("days", consts.DEFAULT_DAYS)
    working_hours = kwargs.get("working_hours", consts.DEFAULT_WORKING_HOURS)
    wet_season_days = kwargs.get("wet_season_days", consts.DEFAULT_WET_SEASON_DAYS)
    dry_season_days = kwargs.get("dry_season_days", consts.DEFAULT_DRY_SEASON_DAYS)
    badtide_season_days = kwargs.get(
        "badtide_season_days", consts.DEFAULT_BADTIDE_SEASON_DAYS
    )

    # Create FactoryParams dynamically
    # We iterate over kwargs and see if they match any consumer machine name
    factory_counts = {}
    for name, spec in iter_consumers():
        # CLI args use underscores (click converts dashes to underscores)
        arg_name = name
        if arg_name in kwargs:
            factory_counts[name] = kwargs[arg_name]
        else:
            factory_counts[name] = 0

    factories = FactoryParams(counts=factory_counts)

    # Create EnergyMixParams
    # We look for specific keys
    power_wheels = kwargs.get("power_wheel", 0)
    water_wheels = kwargs.get("water_wheel", 0)
    large_windmills = kwargs.get("large_windmill", 0)
    windmills = kwargs.get("windmill", 0)
    batteries = kwargs.get("battery", 0)
    battery_height = kwargs.get("battery_height", 0)
    if isinstance(battery_height, list):
        if not battery_height:
            battery_height = 0
        else:
            battery_height = sum(battery_height) / len(battery_height)

    energy_mix = EnergyMixParams(
        power_wheels=power_wheels,
        water_wheels=water_wheels,
        large_windmills=large_windmills,
        windmills=windmills,
        batteries=batteries,
        battery_height=battery_height,
    )

    return SimulationParams(
        days=days,
        working_hours=working_hours,
        wet_season_days=wet_season_days,
        dry_season_days=dry_season_days,
        badtide_season_days=badtide_season_days,
        factories=factories,
        energy_mix=energy_mix,
    )
