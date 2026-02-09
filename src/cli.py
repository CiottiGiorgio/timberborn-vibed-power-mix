import click
import consts
from machines import MachineDatabase
from models import FactoryParams, EnergyMixParams, SimulationParams


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


def create_cli(main_func):
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
    def cli_wrapper(**kwargs):
        main_func(**kwargs)

    # Add dynamic options based on MachineDatabase
    # Consumers
    for name, spec in MachineDatabase.iter_consumers():
        option = click.Option(
            [f"--{name.replace('_', '-')}"],
            type=int,
            default=0,
            help=f"Number of {name.replace('_', ' ')}s",
        )
        cli_wrapper.params.append(option)

    # Producers (Energy Mix)
    # Power Wheels
    if hasattr(MachineDatabase, "power_wheel"):
        option = click.Option(
            ["--power-wheels"],
            type=int,
            default=0,
            help="Number of power wheels",
        )
        cli_wrapper.params.append(option)

    # Water Wheels
    if hasattr(MachineDatabase, "water_wheel"):
        option = click.Option(
            ["--water-wheels"],
            type=int,
            default=0,
            help="Number of water wheels",
        )
        cli_wrapper.params.append(option)

    # Large Windmills
    if hasattr(MachineDatabase, "large_windmill"):
        option = click.Option(
            ["--large-windmills"],
            type=int,
            default=0,
            help="Number of large windmills",
        )
        cli_wrapper.params.append(option)

    # Windmills
    if hasattr(MachineDatabase, "windmill"):
        option = click.Option(
            ["--windmills"],
            type=int,
            default=0,
            help="Number of windmills",
        )
        cli_wrapper.params.append(option)

    # Batteries
    option = click.Option(
        ["--batteries"],
        type=int,
        default=0,
        help="Number of gravity batteries",
    )
    cli_wrapper.params.append(option)

    # Battery Height
    option = click.Option(
        ["--battery-height"],
        type=IntOrIntList(),
        default=0,
        help="Height of gravity batteries in meters. Can be a single int or a comma-separated list of ints.",
    )
    cli_wrapper.params.append(option)

    return cli_wrapper


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
    for name, spec in MachineDatabase.iter_consumers():
        # CLI args use underscores (click converts dashes to underscores)
        arg_name = name
        if arg_name in kwargs:
            factory_counts[name] = kwargs[arg_name]
        else:
            factory_counts[name] = 0

    factories = FactoryParams(counts=factory_counts)

    # Create EnergyMixParams
    # We look for specific keys
    power_wheels = kwargs.get("power_wheels", 0)
    water_wheels = kwargs.get("water_wheels", 0)
    large_windmills = kwargs.get("large_windmills", 0)
    windmills = kwargs.get("windmills", 0)
    batteries = kwargs.get("batteries", 0)
    battery_height = kwargs.get("battery_height", 0)

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
