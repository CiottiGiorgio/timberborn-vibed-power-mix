import os
import pytest
import rng
from models import SimulationParams, EnergyMixParams, FactoryParams
from simulation import run_simulation_batch
from plots.canvas import create_simulation_figure
from matplotlib.testing.compare import compare_images
import consts


@pytest.fixture
def deterministic_rng():
    """Fixture to seed the RNG before a test."""
    rng.seed_rng(42)


def test_visual_output(deterministic_rng, tmp_path):
    """
    Runs a simulation with a fixed seed and compares the generated plot
    against a reference image.
    """
    # Define paths
    test_dir = os.path.dirname(__file__)
    reference_image_path = os.path.join(test_dir, "reference_visual_output.png")
    generated_image_path = tmp_path / "test_visual_output.png"

    # Check if reference image exists
    if not os.path.exists(reference_image_path):
        pytest.fail(
            f"Reference image not found at {reference_image_path}. "
            "Please generate it by running: python3 scripts/refresh_test_image.py"
        )

    # 1. Setup parameters (Matching 'simulate-simple' run configuration)
    # --lumber-mill 1 --wood-workshop 1 --windmill 4 --battery 1 --battery-height 1
    energy_mix = EnergyMixParams(
        power_wheels=0,
        water_wheels=0,
        large_windmills=0,
        windmills=4,
        batteries=1,
        battery_height=1,
    )

    factories = FactoryParams(counts={"lumber_mill": 1, "wood_workshop": 1})

    # Use constants from the application
    params = SimulationParams(
        days=consts.DEFAULT_DAYS,
        wet_season_days=consts.DEFAULT_WET_SEASON_DAYS,
        dry_season_days=consts.DEFAULT_DRY_SEASON_DAYS,
        badtide_season_days=consts.DEFAULT_BADTIDE_SEASON_DAYS,
        working_hours=consts.DEFAULT_WORKING_HOURS,
        energy_mix=energy_mix,
        factories=factories,
    )

    # 2. Run simulation batch
    samples = consts.DEFAULT_SAMPLES_PER_SIM
    results = run_simulation_batch(params, samples)

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

    # 3. Verify some deterministic outputs (sanity check)
    # Cost calculation:
    # Windmills: 4 * 40 = 160
    # Batteries: 1 * (84 + 1*6) = 90
    # Total: 250
    assert worst_run_data.total_cost == 250.0
    assert max_hours_empty >= 0

    # 4. Generate Plot
    fig = create_simulation_figure(worst_run_data, run_empty_hours, samples)
    fig.savefig(str(generated_image_path))

    # 5. Compare with reference image
    # compare_images returns None if images are identical
    # The `tol` parameter allows for minor differences in rendering between environments
    result = compare_images(reference_image_path, str(generated_image_path), tol=10)

    assert result is None, (
        f"Generated image does not match the reference. "
        f"If the change is intentional, run 'scripts/refresh_test_image.py' "
        f"and commit the new reference image. Differences: {result}"
    )
