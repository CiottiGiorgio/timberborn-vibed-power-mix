import logging
import matplotlib.pyplot as plt
from timberborn_power_mix.simulation import simulate_scenario
from timberborn_power_mix.plots.canvas import create_simulation_figure
from timberborn_power_mix.cli import create_cli, parse_params
from timberborn_power_mix.optimizer import optimize, find_optimal_solutions
from timberborn_power_mix.rng import RNGService

logger = logging.getLogger(__name__)


def run_optimization(**kwargs):
    """Runs the optimization process."""
    logger.info("Starting optimization...")
    base_params = parse_params(**kwargs)

    iterations = kwargs.get("iterations", 500)
    samples_per_sim = kwargs.get("samples_per_sim", 2000)

    results = optimize(
        base_params=base_params,
        iterations=iterations,
        simulations_per_config=samples_per_sim,
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


def run_visualization(**kwargs):
    """Visualize power and energy profiles for a single configuration."""

    params = parse_params(**kwargs)

    logger.info(f"Running {params.samples} simulations for visualization...")

    rng_service = RNGService()
    hours_empty_list, worst_run_data, _ = simulate_scenario(params, rng_service)

    create_simulation_figure(worst_run_data, params, hours_empty_list)
    plt.show()


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    cli = create_cli(run_visualization, run_optimization)
    cli()


if __name__ == "__main__":
    main()
