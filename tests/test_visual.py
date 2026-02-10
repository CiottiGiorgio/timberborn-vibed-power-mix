import os
import pytest
import rng
from models import SimulationParams, EnergyMixParams, FactoryParams
from simulation import run_simulation_task
from plots.canvas import create_simulation_figure
from matplotlib.testing.compare import compare_images


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

    # 1. Setup parameters
    energy_mix = EnergyMixParams(
        power_wheels=5,
        water_wheels=2,
        large_windmills=5,
        windmills=5,
        batteries=10,
        battery_height=2,
    )

    factories = FactoryParams(counts={"lumber_mill": 2})

    params = SimulationParams(
        days=30,
        wet_season_days=10,
        dry_season_days=5,
        badtide_season_days=5,
        working_hours=16,
        energy_mix=energy_mix,
        factories=factories,
    )

    # 2. Run simulation
    hours_empty, data = run_simulation_task(params)

    # 3. Generate Plot
    fig = create_simulation_figure(data, [hours_empty], 1)
    fig.savefig(str(generated_image_path))

    # 4. Compare with reference image
    # compare_images returns None if images are identical
    # The `tol` parameter allows for minor differences in rendering between environments
    result = compare_images(reference_image_path, str(generated_image_path), tol=10)

    assert result is None, (
        f"Generated image does not match the reference. "
        f"If the change is intentional, run 'scripts/refresh_test_image.py' "
        f"and commit the new reference image. Differences: {result}"
    )
