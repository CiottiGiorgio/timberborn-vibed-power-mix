import os
import pytest
import numpy as np
from matplotlib.testing.compare import compare_images
from tests.helpers import generate_reference_figure, generate_reference_simulation_data
from timberborn_power_mix.simulation.helpers import calculate_total_cost


def test_visual_output(tmp_path):
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

    # 1. Generate Plot from helper
    fig = generate_reference_figure()
    fig.savefig(str(generated_image_path))

    # 2. Verify some deterministic outputs (sanity check)
    worst_run_data, run_empty_hours, config = generate_reference_simulation_data()
    max_hours_empty = max(run_empty_hours) if len(run_empty_hours) > 0 else -1

    # Cost calculation:
    # Windmills: 4 * 40 = 160
    # Batteries: 1 * (84 + 1*6) = 90
    # Total: 250
    assert calculate_total_cost(config.energy_mix) == 250.0
    assert max_hours_empty >= 0

    # 3. Compare with reference image
    # compare_images returns None if images are identical
    # The `tol` parameter allows for minor differences in rendering between environments
    result = compare_images(reference_image_path, str(generated_image_path), tol=10)

    assert result is None, (
        f"Generated image does not match the reference. "
        f"If the change is intentional, run 'scripts/refresh_test_image.py' "
        f"and commit the new reference image. Differences: {result}"
    )
