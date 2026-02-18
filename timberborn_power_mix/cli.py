import click
import inflect
import logging
from timberborn_power_mix.simulation import consts as sim_consts
from timberborn_power_mix.optimization import consts as opt_consts
from timberborn_power_mix.machines import FactoryName, ProducerName, BatteryName
from timberborn_power_mix.simulation.models import (
    EnergyMixConfig,
    SimulationConfig,
)
from timberborn_power_mix.optimization.models import OptimizationConfig
from timberborn_power_mix.models import ConfigName, CommonConfig, FactoryConfig, IntList

p = inflect.engine()


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
        default=sim_consts.DEFAULT_BADTIDE_SEASON_DAYS,
        show_default=True,
        help="Duration of badtide season in days",
    )(func)
    func = click.option(
        f"--{ConfigName.DRY_DAYS.replace('_', '-')}",
        type=int,
        default=sim_consts.DEFAULT_DRY_SEASON_DAYS,
        show_default=True,
        help="Duration of dry season in days",
    )(func)
    func = click.option(
        f"--{ConfigName.WET_DAYS.replace('_', '-')}",
        type=int,
        default=sim_consts.DEFAULT_WET_SEASON_DAYS,
        show_default=True,
        help="Duration of wet season in days",
    )(func)
    func = click.option(
        f"--{ConfigName.WORKING_HOURS.replace('_', '-')}",
        type=int,
        default=sim_consts.DEFAULT_WORKING_HOURS,
        show_default=True,
        help="Number of working hours per day",
    )(func)
    func = click.option(
        f"--{ConfigName.DAYS.replace('_', '-')}",
        type=int,
        default=sim_consts.DEFAULT_DAYS,
        show_default=True,
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

    # Use inflect to pluralize the CLI option name and help text
    display_name = BatteryName.BATTERY_HEIGHT.replace("_", " ")
    func = click.option(
        f"--{p.plural(BatteryName.BATTERY_HEIGHT.replace('_', '-'))}",
        BatteryName.BATTERY_HEIGHT.value,
        type=IntList(),
        default="",
        help=f"Comma-separated list of {p.plural(display_name)} (e.g., '10,15,10')",
    )(func)

    return func


@click.group()
def cli():
    """Timberborn Power Mix Simulation Tool."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")


@cli.command(name="simulate")
@add_common_params
@add_energy_mix_params
@click.option(
    f"--{ConfigName.SAMPLES.replace('_', '-')}",
    type=int,
    default=sim_consts.DEFAULT_SIMULATION_SAMPLES,
    show_default=True,
    help="Number of samples per simulation",
)
def simulate_cmd(**kwargs):
    """Simulate a configuration with the specified parameters."""
    from timberborn_power_mix.simulation.orchestrator import simulation_orchestrator

    config = parse_simulation_config(**kwargs)
    simulation_orchestrator(config)


@cli.command(name="optimize")
@add_common_params
@click.option(
    f"--{ConfigName.ITERATION.replace('_', '-')}",
    type=int,
    default=opt_consts.DEFAULT_ITERATIONS,
    show_default=True,
    help="Number of optimization iterations",
)
@click.option(
    f"--{ConfigName.SAMPLES.replace('_', '-')}",
    type=int,
    default=opt_consts.DEFAULT_OPTIMIZATION_SAMPLES,
    show_default=True,
    help="Number of samples per simulation during optimization",
)
def optimize_cmd(**kwargs):
    """Optimize the energy mix for a given factory configuration."""
    from timberborn_power_mix.optimization.orchestrator import optimization_orchestrator

    config = parse_optimization_config(**kwargs)
    optimization_orchestrator(config)


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
    battery_height = kwargs.get(BatteryName.BATTERY_HEIGHT, [])

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
    """Parses optimization configuration from kwargs."""
    common_config = parse_common_config(**kwargs)
    return OptimizationConfig(
        **common_config.model_dump(),
        iteration=kwargs[ConfigName.ITERATION],
    )


if __name__ == "__main__":
    cli()
