import os
import pytest
import rng
from models import SimulationParams, EnergyMixParams, FactoryParams
from simulation import run_simulation_task
from plots.canvas import create_simulation_figure
from matplotlib.testing.compare import compare_images

# Simulation Defaults for Testing
TEST_DAYS = 132
TEST_WORKING_HOURS = 16
TEST_SAMPLES_PER_SIM = 1000

# Season Defaults for Testing
TEST_WET_SEASON_DAYS = 3
TEST_DRY_SEASON_DAYS = 30
TEST_BADTIDE_SEASON_DAYS = 30


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

    factories = FactoryParams(counts={
        "lumber_mill": 1,
        "wood_workshop": 1
    })

    # Use local constants
    params = SimulationParams(
        days=TEST_DAYS,
        wet_season_days=TEST_WET_SEASON_DAYS,
        dry_season_days=TEST_DRY_SEASON_DAYS,
        badtide_season_days=TEST_BADTIDE_SEASON_DAYS,
        working_hours=TEST_WORKING_HOURS,
        energy_mix=energy_mix,
        factories=factories,
    )

    # 2. Run simulation
    hours_empty, data = run_simulation_task(params)

    # 3. Verify some deterministic outputs (sanity check)
    # Cost calculation:
    # Windmills: 4 * 40 = 160
    # Batteries: 1 * (84 + 1*6) = 90
    # Total: 250
    assert data.total_cost == 250.0
    assert hours_empty >= 0

    # 4. Generate Plot
    fig = create_simulation_figure(data, [hours_empty], 1)
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
