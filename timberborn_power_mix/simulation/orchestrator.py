import logging
import matplotlib.pyplot as plt
from timberborn_power_mix.simulation.engine import (
    run_simulation_multithread,
    run_simulation_singlethread,
)
from timberborn_power_mix.plots.canvas import create_simulation_figure
from timberborn_power_mix.simulation.models import SimulationConfig

logger = logging.getLogger(__name__)


def run_simulation(config: SimulationConfig) -> None:
    """Visualize power and energy profiles for a single configuration."""

    logger.info(f"Running {config.samples} simulations for visualization...")

    if config.threads is None or config.threads > 1:
        res = run_simulation_multithread(config)
    else:
        res = run_simulation_singlethread(config)

    create_simulation_figure(config, res)
    plt.show()
