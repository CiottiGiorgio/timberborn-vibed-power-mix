from itertools import chain

import click
import inflect
from timberborn_power_mix import consts
from timberborn_power_mix.machines import FACTORY_DATABASE, PRODUCER_DATABASE, FactoryName, ProducerName, BatteryName
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

    # Common simulation parameter
    func = click.option(
        "--samples",
        type=int,
        default=consts.DEFAULT_SAMPLES,
        help="Number of samples per simulation",
    )(func)

    func = click.option(
        "--days",
        type=int,
        default=consts.DEFAULT_DAYS,
        help="Number of days for the simulation",
    )(func)

    func = click.option(
        "--working-hours",
        type=int,
        default=consts.DEFAULT_WORKING_HOURS,
        help="Number of working hours per day",
    )(func)

    func = click.option(
        "--wet-season-days",
        type=int,
        default=consts.DEFAULT_WET_SEASON_DAYS,
        help="Duration of wet season in days",
    )(func)

    func = click.option(
        "--dry-season-days",
        type=int,
        default=consts.DEFAULT_DRY_SEASON_DAYS,
        help="Duration of dry season in days",
    )(func)

    func = click.option(
        "--badtide-season-days",
        type=int,
        default=consts.DEFAULT_BADTIDE_SEASON_DAYS,
        help="Duration of badtide season in days",
    )(func)

    # Consumers
    for name in FactoryName:
        display_name = name.replace("_", " ")
        func = click.option(
            f"--{name.replace('_', '-')}",
            type=int,
            default=0,
            help=f"Number of {p.plural(display_name)}",
        )(func)

    return func


def add_energy_mix_params(func):
    """Decorator to add energy mix parameters (for run command)."""

    # Producers
    for name in chain(ProducerName, BatteryName):
        display_name = name.replace("_", " ")
        func = click.option(
            f"--{name.replace('_', '-')}",
            type=int,
            default=0,
            help=f"Number of {p.plural(display_name)}",
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
        wet_season_days=kwargs["wet_season_days"],
        dry_season_days=kwargs["dry_season_days"],
        badtide_season_days=kwargs["badtide_season_days"],
        factories=factories,
        energy_mix=energy_mix,
    )
