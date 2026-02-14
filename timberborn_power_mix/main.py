import logging
import matplotlib.pyplot as plt
from timberborn_power_mix.simulation.core import run_simulation
from timberborn_power_mix.plots.canvas import create_simulation_figure
from timberborn_power_mix.cli import (
    create_cli,
    parse_simulation_config,
    parse_optimization_config,
)
from timberborn_power_mix.optimizer import optimize, find_optimal_solutions

logger = logging.getLogger(__name__)


def simulate_optimization(**kwargs):
    """Runs the optimization process."""
    logger.info("Starting optimization...")
    config = parse_optimization_config(**kwargs)

    results = optimize(
        base_config=config,
        iterations=config.iterations,
        simulations_per_config=config.samples,
    )

    optimal_solutions = find_optimal_solutions(results, max_empty_percent=5.0)

    logger.info(
        "\n--- Optimal Solutions (Satisfying < 5% Empty Time in 95th Percentile Case) ---"
    )
    if not optimal_solutions:
        logger.info("No solutions found satisfying the criteria.")
    else:
        # Print top 5 cheapest solutions
        for i, result in enumerate(optimal_solutions[:5]):
            logger.info(f"Rank {i+1}: {result}")
    logger.info(
        "-----------------------------------------------------------------------------"
    )


def simulate_visualization(**kwargs):
    """Visualize power and energy profiles for a single configuration."""

    config = parse_simulation_config(**kwargs)

    logger.info(f"Running {config.samples} simulations for visualization...")

    res = run_simulation(config)

    create_simulation_figure(res.worst_sample, config, res.hours_empty_results)
    plt.show()


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    cli = create_cli(simulate_visualization, simulate_optimization)
    cli()


if __name__ == "__main__":
    main()
