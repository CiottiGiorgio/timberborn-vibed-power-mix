from timberborn_power_mix.rng import RNGService
from timberborn_power_mix.models import SimulationParams, EnergyMixParams, FactoryParams
from timberborn_power_mix.simulation import run_simulation_batch
from timberborn_power_mix.plots.canvas import create_simulation_figure
from tests import consts


def generate_reference_simulation_data():
    """
    Runs a deterministic simulation and returns the data for the worst-case scenario.
    """
    # 1. Setup RNG service and seed it
    rng_service = RNGService(seed=42)

    # 2. Setup parameters
    energy_mix = EnergyMixParams(
        power_wheels=0,
        water_wheels=0,
        large_windmills=0,
        windmills=4,
        batteries=1,
        battery_height=1,
    )

    factories = FactoryParams(lumber_mill=1, wood_workshop=1)

    params = SimulationParams(
        days=consts.DEFAULT_DAYS,
        wet_season_days=consts.DEFAULT_WET_SEASON_DAYS,
        dry_season_days=consts.DEFAULT_DRY_SEASON_DAYS,
        badtide_season_days=consts.DEFAULT_BADTIDE_SEASON_DAYS,
        working_hours=consts.DEFAULT_WORKING_HOURS,
        energy_mix=energy_mix,
        factories=factories,
    )

    # 3. Run simulation
    samples = consts.DEFAULT_SAMPLES_PER_SIM
    results = run_simulation_batch(params, samples, rng_service)

    run_empty_hours = []
    worst_run_data = None
    max_hours_empty = -1

    for hours_empty, data in results:
        run_empty_hours.append(hours_empty)
        if hours_empty > max_hours_empty:
            max_hours_empty = hours_empty
            worst_run_data = data

    # If all runs are perfect, just take the last one to generate a plot
    if worst_run_data is None and results:
        worst_run_data = results[-1][1]

    return worst_run_data, run_empty_hours, samples


def generate_reference_figure():
    """
    Generates the reference plot figure.
    """
    worst_run_data, run_empty_hours, samples = generate_reference_simulation_data()
    fig = create_simulation_figure(worst_run_data, run_empty_hours, samples)
    return fig
