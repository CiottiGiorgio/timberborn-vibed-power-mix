import logging
import matplotlib.pyplot as plt
from timberborn_power_mix.simulation.core import run_simulation
from timberborn_power_mix.plots.canvas import create_simulation_figure
from timberborn_power_mix.cli import (
    create_cli,
    parse_simulation_config,
)

logger = logging.getLogger(__name__)


def simulate_visualization(**kwargs):
    """Visualize power and energy profiles for a single configuration."""

    config = parse_simulation_config(**kwargs)

    logger.info(f"Running {config.samples} simulations for visualization...")

    res = run_simulation(config)

    create_simulation_figure(
        res.worst_sample, config, res.aggregated_samples.hours_empty_results
    )
    plt.show()


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    cli = create_cli(simulate_visualization)
    cli()


if __name__ == "__main__":
    main()
