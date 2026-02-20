import logging

import numpy as np
from matplotlib import pyplot as plt

import timberborn_power_mix.simulation.helpers as sim_helpers
import timberborn_power_mix.helpers as helpers
from timberborn_power_mix.plots.canvas import create_simulation_figure
from timberborn_power_mix.simulation.engine import run_simulation_multithreaded
from timberborn_power_mix.simulation.models import SimulationConfig

logger = logging.getLogger(__name__)


def run_simulation(config: SimulationConfig) -> None:
    """Visualize power and energy profiles for a single configuration."""

    logger.info(f"Running {config.samples} simulations for visualization...")
    threads = helpers.calculate_optimal_threads(config.threads, config.samples)

    jit_config = config.to_jit_config()
    jit_config = jit_config._replace(threads=threads)
    cached_consts = sim_helpers.calculate_jit_cached_consts(config)

    # Generate seeds for all samples
    ss = np.random.SeedSequence(config.seed)
    all_seeds = ss.generate_state(config.samples)

    res = run_simulation_multithreaded(
        jit_config,
        cached_consts,
        all_seeds,
    )

    create_simulation_figure(config, res)
    plt.show()
