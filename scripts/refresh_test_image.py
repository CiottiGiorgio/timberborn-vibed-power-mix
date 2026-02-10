import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), "../src"))

import rng
from models import SimulationParams, EnergyMixParams, FactoryParams
from simulation import run_simulation_task
from plots.canvas import create_simulation_figure


def refresh_reference_image():
    print("Refreshing reference image for visual tests...")
    
    # 1. Seed the RNG for determinism
    rng.seed_rng(42)

    # 2. Setup parameters (Must match the test case!)
    energy_mix = EnergyMixParams(
        power_wheels=5,
        water_wheels=2,
        large_windmills=5,
        windmills=5,
        batteries=10,
        battery_height=2
    )
    
    factories = FactoryParams(counts={"lumber_mill": 2})
    
    params = SimulationParams(
        days=30,
        wet_season_days=10,
        dry_season_days=5,
        badtide_season_days=5,
        working_hours=16,
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
