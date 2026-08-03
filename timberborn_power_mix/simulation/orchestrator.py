import logging

from matplotlib import pyplot as plt

from timberborn_power_mix import helpers
from timberborn_power_mix.plots.canvas import create_simulation_figure
from timberborn_power_mix.simulation import helpers as sim_helpers
from timberborn_power_mix.simulation.engine import (
    jit_multithread_simulation,
    jit_singlethread_simulation,
)
from timberborn_power_mix.simulation.models import SimulationConfig
from timberborn_power_mix.structures import SimulationResult

logger = logging.getLogger(__name__)


def simulation_orchestrator(config: SimulationConfig) -> None:
    """Visualize power and energy profiles for a single configuration."""

    logger.info(f"Running {config.samples} simulations for visualization...")

    res = run_simulation(config)
    create_simulation_figure(config, res)
    plt.show()


def run_simulation(config: SimulationConfig) -> SimulationResult:
    jit_config = config.to_jit_config()
    cached_consts = sim_helpers.calculate_jit_cached_consts(config)

    res: SimulationResult
    if config.threads is None or config.threads > 1:
        threads = helpers.calculate_optimal_threads(config.threads, config.samples)

        res = jit_multithread_simulation(jit_config, threads, cached_consts)
    else:
        res = jit_singlethread_simulation(jit_config, cached_consts)

    return res
