import click
import logging
from typing import Any, Callable, TypeVar
from timberborn_power_mix.simulation import consts as sim_consts
from timberborn_power_mix.optimization import consts as opt_consts
from timberborn_power_mix.machines import FactoryName, ProducerName, BatteryName
from timberborn_power_mix.simulation.models import (
    EnergyMixConfig,
    SimulationConfig,
)
from timberborn_power_mix.optimization.models import OptimizationConfig
from timberborn_power_mix.models import CommonConfig, FactoryConfig
from timberborn_power_mix.structures import ConfigName

F = TypeVar("F", bound=Callable[..., Any])


def add_common_params(func: F) -> F:
    """Decorator to add common simulation parameters to a click command."""

    for name in reversed(FactoryName):
        display_name = name.replace("_", " ")
        func = click.option(
            f"--{name.replace('_', '-')}",
            type=int,
            default=0,
            help=f"Number of {display_name}",
        )(func)

    func = click.option(
        f"--{ConfigName.BADTIDE_DAYS.replace('_', '-')}",
        type=int,
        default=sim_consts.DEFAULT_BADTIDE_SEASON_DAYS,
        help="Duration of badtide season in days",
    )(func)
    func = click.option(
        f"--{ConfigName.DRY_DAYS.replace('_', '-')}",
        type=int,
        default=sim_consts.DEFAULT_DRY_SEASON_DAYS,
        help="Duration of dry season in days",
    )(func)
    func = click.option(
        f"--{ConfigName.WET_DAYS.replace('_', '-')}",
        type=int,
        default=sim_consts.DEFAULT_WET_SEASON_DAYS,
        help="Duration of wet season in days",
    )(func)
    func = click.option(
        f"--{ConfigName.WORKING_HOURS.replace('_', '-')}",
        type=int,
        default=sim_consts.DEFAULT_WORKING_HOURS,
        help="Number of working hours per day",
    )(func)
    func = click.option(
        f"--{ConfigName.DAYS.replace('_', '-')}",
        type=int,
        default=sim_consts.DEFAULT_DAYS,
        help="Number of days for the simulation",
    )(func)
    func = click.option(
        f"--{ConfigName.THREADS}",
        type=int,
        default=None,
        help="Number of threads to use for parallelism",
    )(func)
    func = click.option(
        f"--{ConfigName.SEED}",
        type=int,
        default=None,
        help="Seed for the random number generator",
    )(func)

    return func


def add_energy_mix_params(func: F) -> F:
    """Decorator to add energy mix parameters (for simulate command)."""

    # Producers
    for name in reversed(ProducerName):
        display_name = name.replace("_", " ")
        func = click.option(
            f"--{name.replace('_', '-')}",
            type=int,
            default=0,
            help=f"Number of {display_name}",
        )(func)

    # Battery heights
    func = click.option(
        "--battery",
        BatteryName.BATTERY_HEIGHTS.value,
        type=int,
        multiple=True,
        help="Height of a gravity battery. Can be specified multiple times (e.g., '--battery 10 --battery 15')",
    )(func)

    return func


@click.group()
def cli() -> None:
    """Timberborn Power Mix Simulation Tool."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")


@cli.command(name="simulate")
@add_common_params
@add_energy_mix_params
@click.option(
    f"--{ConfigName.SAMPLES.replace('_', '-')}",
    type=int,
    default=sim_consts.DEFAULT_SIMULATION_SAMPLES,
    help="Number of samples per simulation",
)
def simulate_cmd(**kwargs: Any) -> None:
    """Simulate a configuration with the specified parameters."""
    from timberborn_power_mix.simulation.orchestrator import simulation_orchestrator

    config = parse_simulation_config(**kwargs)
    simulation_orchestrator(config)


@cli.command(name="optimize")
@add_common_params
@click.option(
    f"--{ConfigName.MAX_TIME.replace('_', '-')}",
    type=int,
    default=opt_consts.DEFAULT_MAX_TIME_SECONDS,
    help="Maximum optimization time in seconds",
)
@click.option(
    f"--{ConfigName.SAMPLES.replace('_', '-')}",
    type=int,
    default=opt_consts.DEFAULT_OPTIMIZATION_SAMPLES,
    help="Number of samples per simulation during optimization",
)
def optimize_cmd(**kwargs: Any) -> None:
    """Optimize the energy mix for a given factory configuration."""
    from timberborn_power_mix.optimization.orchestrator import optimization_orchestrator

    config = parse_optimization_config(**kwargs)
    optimization_orchestrator(config)


def parse_common_config(**kwargs: Any) -> CommonConfig:
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


def parse_simulation_config(**kwargs: Any) -> SimulationConfig:
    """Parses full simulation configuration from kwargs."""
    battery_heights = list(kwargs.get(BatteryName.BATTERY_HEIGHTS, ()))

    energy_mix = EnergyMixConfig(
        battery_heights=battery_heights,
        **{
            key: value
            for key, value in kwargs.items()
            if key in EnergyMixConfig.model_fields
            and key != BatteryName.BATTERY_HEIGHTS
        },
    )

    common_config = parse_common_config(**kwargs)

    return SimulationConfig(
        **common_config.model_dump(),
        energy_mix=energy_mix,
    )


def parse_optimization_config(**kwargs: Any) -> OptimizationConfig:
    """Parses optimization configuration from kwargs."""
    common_config = parse_common_config(**kwargs)
    return OptimizationConfig(
        **common_config.model_dump(),
        max_time=kwargs[ConfigName.MAX_TIME],
    )


if __name__ == "__main__":
    cli()
