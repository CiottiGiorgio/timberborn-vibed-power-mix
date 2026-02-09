import numpy as np
import os
from concurrent.futures import ProcessPoolExecutor
from simulation import simulate_scenario, run_simulation_batch
from plots.canvas import plot_simulation
from cli import create_cli, parse_params


def main(**kwargs):
    """Visualize power and energy profiles for an industrial complex."""

    params = parse_params(**kwargs)
    runs = kwargs.get("runs", 1)

    if runs > 1:
        run_empty_hours = []
        worst_run_data = None
        max_hours_empty = -1

        print(f"Running {runs} simulations...")

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

        plot_simulation(worst_run_data, run_empty_hours, runs)

    else:
        data = simulate_scenario(params)
        # Even for a single run, we can pass the data to show the histogram (it will just be one bar)
        # But usually single run visualization focuses on the time series.
        # However, to be consistent with "Always show the fifth graph", let's pass it.
        # We need to calculate hours_empty for this single run to pass it.
        battery_after_day1 = data.battery_charge[24:]
        hours_empty = np.sum(battery_after_day1 <= 0)
        plot_simulation(data, [hours_empty], 1)


if __name__ == "__main__":
    cli = create_cli(main)
    cli()
