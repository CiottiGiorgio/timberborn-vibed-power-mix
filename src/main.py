import os
from concurrent.futures import ProcessPoolExecutor
from simulation import run_simulation_batch
from plots.canvas import plot_simulation
from cli import create_cli, parse_params
from optimizer import optimize, find_optimal_solutions


def run_optimization(**kwargs):
    """Runs the optimization process."""
    print("Starting optimization...")
    base_params = parse_params(**kwargs)

    iterations = kwargs.get("iterations", 500)
    samples_per_sim = kwargs.get("samples_per_sim", 2000)

    # Note: Bounds are not yet configurable via CLI, using defaults.
    results = optimize(
        base_params=base_params,
        iterations=iterations,
        simulations_per_config=samples_per_sim,
    )

    optimal_solutions = find_optimal_solutions(results, max_empty_percent=5.0)

    print(
        "\n--- Optimal Solutions (Satisfying < 5% Empty Time in 95th Percentile Case) ---"
    )
    if not optimal_solutions:
        print("No solutions found satisfying the criteria.")
    else:
        # Print top 5 cheapest solutions
        for i, result in enumerate(optimal_solutions[:5]):
            print(f"Rank {i+1}: {result}")
    print(
        "-----------------------------------------------------------------------------"
    )


def run_visualization(**kwargs):
    """Visualize power and energy profiles for a single configuration."""

    params = parse_params(**kwargs)
    runs = kwargs.get("samples_per_sim", 1)

    run_empty_hours = []
    worst_run_data = None
    max_hours_empty = -1

    print(f"Running {runs} simulations for visualization...")

    # Determine number of workers
    num_workers = os.cpu_count() or 1

    # Calculate chunk size to split work evenly
    chunk_size = runs // num_workers
    remainder = runs % num_workers

    batches = []
    for i in range(num_workers):
        # Distribute remainder among the first few workers
        count = chunk_size + (1 if i < remainder else 0)
        if count > 0:
            batches.append(count)

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        # Submit batch tasks
        futures = [
            executor.submit(run_simulation_batch, params, batch_size)
            for batch_size in batches
        ]

        for future in futures:
            batch_results = future.result()

            for hours_empty, data in batch_results:
                # Collect data for ALL runs
                run_empty_hours.append(hours_empty)

                if hours_empty > max_hours_empty:
                    max_hours_empty = hours_empty
                    worst_run_data = data

                # If worst_run_data is still None, keep the last one just in case
                if worst_run_data is None:
                    worst_run_data = data

    if worst_run_data:
        plot_simulation(worst_run_data, run_empty_hours, runs)
    else:
        print("No simulation data to plot.")


if __name__ == "__main__":
    cli = create_cli(run_visualization, run_optimization)
    cli()
