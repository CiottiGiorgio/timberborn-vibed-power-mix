import pytest
import rng
from models import SimulationParams, EnergyMixParams, FactoryParams
from simulation import run_simulation_task
from plots.canvas import create_simulation_figure


@pytest.fixture
def deterministic_rng():
    """Fixture to seed the RNG before a test."""
    rng.seed_rng(42)


def test_visual_output(deterministic_rng, tmp_path):
    """
    Runs a simulation with a fixed seed and verifies that the output plot is generated.
    """
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

    # 3. Verify some deterministic outputs (sanity check)
    # Cost calculation:
    # Power Wheels: 5 * 50 = 250
    # Water Wheels: 2 * 50 = 100
    # Large Windmills: 5 * 75 = 375
    # Windmills: 5 * 40 = 200
    # Batteries: 10 * (84 + 2*6) = 960
    # Total: 1885
    assert data.total_cost == 1885.0
    assert hours_empty >= 0

    # 4. Generate Plot
    # Use tmp_path fixture from pytest to avoid cluttering the project root
    output_file = tmp_path / "test_visual_output.png"

    fig = create_simulation_figure(data, [hours_empty], 1)
    fig.savefig(str(output_file))

    assert output_file.exists()
    assert output_file.stat().st_size > 0
