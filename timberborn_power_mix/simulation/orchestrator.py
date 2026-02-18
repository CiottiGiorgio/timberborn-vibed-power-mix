import logging
import matplotlib.pyplot as plt
from timberborn_power_mix.simulation.engine import run_simulation_multithread
from timberborn_power_mix.plots.canvas import create_simulation_figure
from timberborn_power_mix.simulation.models import SimulationConfig

logger = logging.getLogger(__name__)


def simulation_orchestrator(config: SimulationConfig):  # type: ignore[valid-type]
    """Visualize power and energy profiles for a single configuration."""

    logger.info(f"Running {config.samples} simulations for visualization...")

    res = run_simulation_multithread(config)

    create_simulation_figure(config, res)
    plt.show()
