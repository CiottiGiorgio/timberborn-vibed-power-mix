import click
import inflect
from timberborn_power_mix import consts
from timberborn_power_mix.machines import FactoryName, ProducerName, BatteryName
from timberborn_power_mix.simulation.models import (
    FactoryConfig,
    EnergyMixConfig,
    SimulationConfig,
)
from timberborn_power_mix.consts import ConfigKey

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
        f"--{ConfigKey.BADTIDE_DAYS.replace('_', '-')}",
        type=int,
        default=consts.DEFAULT_BADTIDE_SEASON_DAYS,
        show_default=True,
        help="Duration of badtide season in days",
    )(func)
    func = click.option(
        f"--{ConfigKey.DRY_DAYS.replace('_', '-')}",
        type=int,
        default=consts.DEFAULT_DRY_SEASON_DAYS,
        show_default=True,
        help="Duration of dry season in days",
    )(func)
    func = click.option(
        f"--{ConfigKey.WET_DAYS.replace('_', '-')}",
        type=int,
        default=consts.DEFAULT_WET_SEASON_DAYS,
        show_default=True,
        help="Duration of wet season in days",
    )(func)

    # 1. Core simulation parameters (Top of the group)
    func = click.option(
        f"--{ConfigKey.WORKING_HOURS.replace('_', '-')}",
        type=int,
        default=consts.DEFAULT_WORKING_HOURS,
        show_default=True,
        help="Number of working hours per day",
    )(func)
    func = click.option(
        f"--{ConfigKey.DAYS.replace('_', '-')}",
        type=int,
        default=consts.DEFAULT_DAYS,
        show_default=True,
        help="Number of days for the simulation",
    )(func)
    func = click.option(
        f"--{ConfigKey.SAMPLES.replace('_', '-')}",
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
        f"--{ConfigKey.ITERATIONS}",
        type=int,
        default=consts.DEFAULT_OPTIMIZATION_ITERATIONS,
        help="Number of optimization iterations",
    )
    def optimize_cmd(**kwargs):
        """Optimize the energy mix for the specified parameters."""
        optimize_callback(**kwargs)

    return cli


def parse_config(**kwargs) -> SimulationConfig:
    # Create a FactoryConfig by removing all fields from kwargs that are not contained in FactoryConfig.
    # Since FactoryConfig requires all fields, if kwargs is missing one of those fields, this fails.
    factories = FactoryConfig(
        **{
            key: value
            for key, value in kwargs.items()
            if key in FactoryConfig.model_fields
        }
    )

    battery_height = kwargs[BatteryName.BATTERY_HEIGHT]
    if isinstance(battery_height, list):
        if not battery_height:
            battery_height = 0.0
        else:
            battery_height = sum(battery_height) / len(battery_height)
    energy_mix = EnergyMixConfig(
        battery_height=battery_height,
        **{
            key: value
            for key, value in kwargs.items()
            if key in EnergyMixConfig.model_fields and key != BatteryName.BATTERY_HEIGHT
        },
    )

    return SimulationConfig(
        **{key: kwargs[key] for key in SimulationConfig.model_fields if key not in [ConfigKey.FACTORIES, ConfigKey.ENERGY_MIX]},
        factories=factories,
        energy_mix=energy_mix,
    )
