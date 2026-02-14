import click
import inflect
from timberborn_power_mix import consts
from timberborn_power_mix.machines import FactoryName, ProducerName, BatteryName
from timberborn_power_mix.simulation.models import (
    FactoryConfig,
    EnergyMixConfig,
    SimulationConfig,
    CommonConfig,
    OptimizationConfig,
)
from timberborn_power_mix.models import ConfigName

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
        f"--{ConfigName.BADTIDE_DAYS.replace('_', '-')}",
        type=int,
        default=consts.DEFAULT_BADTIDE_SEASON_DAYS,
        show_default=True,
        help="Duration of badtide season in days",
    )(func)
    func = click.option(
        f"--{ConfigName.DRY_DAYS.replace('_', '-')}",
        type=int,
        default=consts.DEFAULT_DRY_SEASON_DAYS,
        show_default=True,
        help="Duration of dry season in days",
    )(func)
    func = click.option(
        f"--{ConfigName.WET_DAYS.replace('_', '-')}",
        type=int,
        default=consts.DEFAULT_WET_SEASON_DAYS,
        show_default=True,
        help="Duration of wet season in days",
    )(func)

    # 1. Core simulation parameters (Top of the group)
    func = click.option(
        f"--{ConfigName.SEED}",
        type=int,
        default=None,
        help="Seed for the random number generator",
    )(func)
    func = click.option(
        f"--{ConfigName.WORKING_HOURS.replace('_', '-')}",
        type=int,
        default=consts.DEFAULT_WORKING_HOURS,
        show_default=True,
        help="Number of working hours per day",
    )(func)
    func = click.option(
        f"--{ConfigName.DAYS.replace('_', '-')}",
        type=int,
        default=consts.DEFAULT_DAYS,
        show_default=True,
        help="Number of days for the simulation",
    )(func)
    func = click.option(
        f"--{ConfigName.SAMPLES.replace('_', '-')}",
        type=int,
        default=consts.DEFAULT_SAMPLES,
        show_default=True,
        help="Number of samples per simulation",
    )(func)

    return func


def add_energy_mix_params(func):
    """Decorator to add energy mix parameters (for simulate command)."""

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


def create_cli(simulate_callback, optimize_callback):
    @click.group()
    def cli():
        """Timberborn Power Mix Simulation and Optimization Tool."""
        pass

    @cli.command(name="simulate")
    @add_common_params
    @add_energy_mix_params
    def simulate_cmd(**kwargs):
        """Simulate a configuration with the specified parameters."""
        simulate_callback(**kwargs)

    @cli.command(name="optimize")
    @add_common_params
    @click.option(
        f"--{ConfigName.ITERATIONS}",
        type=int,
        default=consts.DEFAULT_OPTIMIZATION_ITERATIONS,
        help="Number of optimization iterations",
    )
    def optimize_cmd(**kwargs):
        """Optimize the energy mix for the specified parameters."""
        optimize_callback(**kwargs)

    return cli


def parse_common_config(**kwargs) -> CommonConfig:
    """Parses common configuration parameters from kwargs."""
    factories = FactoryConfig(
        **{
            key: value
            for key, value in kwargs.items()
            if key in FactoryConfig.model_fields
        }
    )

    return CommonConfig(
        factories=factories,
        **{
            key: value
            for key, value in kwargs.items()
            if key in CommonConfig.model_fields and key != ConfigName.FACTORIES
        },
    )


def parse_simulation_config(**kwargs) -> SimulationConfig:
    """Parses full simulation configuration from kwargs."""
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

    common_config = parse_common_config(**kwargs)

    return SimulationConfig(
        **common_config.model_dump(),
        energy_mix=energy_mix,
    )


def parse_optimization_config(**kwargs) -> OptimizationConfig:
    """Parses full optimization configuration from kwargs."""
    common_config = parse_common_config(**kwargs)

    return OptimizationConfig(
        **common_config.model_dump(),
        **{
            key: value
            for key, value in kwargs.items()
            if key in OptimizationConfig.model_fields
            and key not in CommonConfig.model_fields
        },
    )
