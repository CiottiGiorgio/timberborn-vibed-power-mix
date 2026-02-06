import numpy as np
from concurrent.futures import ProcessPoolExecutor
from simulation import simulate_scenario, run_simulation_task
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

        # Let ProcessPoolExecutor handle worker count automatically (defaults to os.cpu_count())
        with ProcessPoolExecutor() as executor:
            # Submit individual tasks instead of batches
            futures = [
                executor.submit(run_simulation_task, params) for _ in range(runs)
            ]

            for future in futures:
                hours_empty, data = future.result()

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
