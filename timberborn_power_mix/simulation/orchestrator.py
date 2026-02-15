import logging
import matplotlib.pyplot as plt
from timberborn_power_mix.simulation.engine import run_simulation
from timberborn_power_mix.plots.canvas import create_simulation_figure
from timberborn_power_mix.cli import parse_simulation_config

logger = logging.getLogger(__name__)


def simulation_orchestrator(**kwargs):
    """Visualize power and energy profiles for a single configuration."""

    config = parse_simulation_config(**kwargs)

    logger.info(f"Running {config.samples} simulations for visualization...")

    res = run_simulation(config)

    create_simulation_figure(config, res)
    plt.show()
