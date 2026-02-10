import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), "../src"))

import rng
from models import SimulationParams, EnergyMixParams, FactoryParams
from simulation import run_simulation_task
from plots.canvas import create_simulation_figure

# Simulation Defaults for Testing
TEST_DAYS = 132
TEST_WORKING_HOURS = 16
TEST_SAMPLES_PER_SIM = 1000

# Season Defaults for Testing
TEST_WET_SEASON_DAYS = 3
TEST_DRY_SEASON_DAYS = 30
TEST_BADTIDE_SEASON_DAYS = 30


def refresh_reference_image():
    print("Refreshing reference image for visual tests...")
    
    # 1. Seed the RNG for determinism
    rng.seed_rng(42)

    # 2. Setup parameters (Matching 'simulate-simple' run configuration)
    # --lumber-mill 1 --wood-workshop 1 --windmill 4 --battery 1 --battery-height 1
    energy_mix = EnergyMixParams(
        power_wheels=0,
        water_wheels=0,
        large_windmills=0,
        windmills=4,
        batteries=1,
        battery_height=1
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
        factories=factories
    )

    # 3. Run simulation
    hours_empty, data = run_simulation_task(params)

    # 4. Generate Plot
    # Save to tests/reference_visual_output.png
    output_path = os.path.join(os.path.dirname(__file__), "../tests/reference_visual_output.png")
    output_path = os.path.abspath(output_path)
    
    fig = create_simulation_figure(data, [hours_empty], 1)
    fig.savefig(output_path)
    
    print(f"Reference image saved to: {output_path}")

if __name__ == "__main__":
    refresh_reference_image()
