import click
import inflect
from timberborn_power_mix import consts
from timberborn_power_mix.machines import FactoryName, ProducerName, BatteryName
from timberborn_power_mix.models import FactoryParams, EnergyMixParams, SimulationParams

p = inflect.engine()


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

    for name in reversed(FactoryName):
        display_name = name.replace("_", " ")
        func = click.option(
            f"--{name.replace('_', '-')}",
            type=int,
            default=0,
            help=f"Number of {p.plural(display_name)}",
        )(func)

    func = click.option(
        "--badtide-days",
        type=int,
        default=consts.DEFAULT_BADTIDE_SEASON_DAYS,
        show_default=True,
        help="Duration of badtide season in days",
    )(func)
    func = click.option(
        "--dry-days",
        type=int,
        default=consts.DEFAULT_DRY_SEASON_DAYS,
        show_default=True,
        help="Duration of dry season in days",
    )(func)
    func = click.option(
        "--wet-days",
        type=int,
        default=consts.DEFAULT_WET_SEASON_DAYS,
        show_default=True,
        help="Duration of wet season in days",
    )(func)

    # 1. Core simulation parameters (Top of the group)
    func = click.option(
        "--working-hours",
        type=int,
        default=consts.DEFAULT_WORKING_HOURS,
        show_default=True,
        help="Number of working hours per day",
    )(func)
    func = click.option(
        "--days",
        type=int,
        default=consts.DEFAULT_DAYS,
        show_default=True,
        help="Number of days for the simulation",
    )(func)
    func = click.option(
        "--samples",
        type=int,
        default=consts.DEFAULT_SAMPLES,
        show_default=True,
        help="Number of samples per simulation",
    )(func)

    return func


def add_energy_mix_params(func):
    """Decorator to add energy mix parameters (for run command)."""

    # Producers
    for name in reversed(ProducerName):
        display_name = name.replace("_", " ")
        func = click.option(
            f"--{name.replace('_', '-')}",
            type=int,
            default=0,
            help=f"Number of {p.plural(display_name)}",
        )(func)

    func = click.option(
        f"--{BatteryName.BATTERY_HEIGHT.replace('_', '-')}",
        type=IntOrIntList(),
        default="0",
        help="Height of the gravity batteries (accepts single int or list)",
    )(func)

    # 2. Battery Count
    func = click.option(
        f"--{BatteryName.BATTERY.replace('_', '-')}",
        type=int,
        default=0,
        help="Number of batteries",
    )(func)

    return func


def create_cli(run_callback, optimize_callback):
    @click.group()
    def cli():
        """Timberborn Power Mix Simulation and Optimization Tool."""
        pass

    @cli.command(name="run")
    @add_common_params
    @add_energy_mix_params
    def run_cmd(**kwargs):
        """Run a simulation with the specified parameters."""
        run_callback(**kwargs)

    @cli.command(name="optimize")
    @add_common_params
    @click.option(
        "--iterations",
        type=int,
        default=consts.DEFAULT_OPTIMIZATION_ITERATIONS,
        help="Number of optimization iterations",
    )
    def optimize_cmd(**kwargs):
        """Optimize the energy mix for the specified parameters."""
        optimize_callback(**kwargs)

    return cli


def parse_params(**kwargs) -> SimulationParams:
    # Create a FactoryParams by removing all fields from kwargs that are not contained in FactoryParams.
    # Since FactoryParams requires all fields, if kwargs is missing one of those fields, this fails.
    factories = FactoryParams(
        **{
            key: value
            for key, value in kwargs.items()
            if key in FactoryParams.model_fields
        }
    )

    battery_height = kwargs["battery_height"]
    if isinstance(battery_height, list):
        if not battery_height:
            battery_height = 0.0
        else:
            battery_height = sum(battery_height) / len(battery_height)
    energy_mix = EnergyMixParams(
        battery_height=battery_height,
        **{
            key: value
            for key, value in kwargs.items()
            if key in EnergyMixParams.model_fields and key != "battery_height"
        },
    )

    return SimulationParams(
        samples=kwargs["samples"],
        days=kwargs["days"],
        working_hours=kwargs["working_hours"],
        wet_season_days=kwargs["wet_days"],
        dry_season_days=kwargs["dry_days"],
        badtide_season_days=kwargs["badtide_days"],
        factories=factories,
        energy_mix=energy_mix,
    )
