import logging

from scalene import scalene_profiler
from timberborn_power_mix.simulation.orchestrator import run_simulation
from timberborn_power_mix.simulation.models import (
    SimulationConfig,
    EnergyMixConfig,
)
from timberborn_power_mix.models import FactoryConfig
from timberborn_power_mix.machines import (
    FACTORY_DATABASE,
    PRODUCER_DATABASE,
    FactoryName,
    ProducerName,
    BatteryName,
)
from timberborn_power_mix.simulation import consts as sim_consts

# Configure logging
logging.basicConfig(level=logging.INFO)


def run_profiled_simulation():
    # Configuration based on 'simulate-simple' run configuration:
    # --lumber-mills 1 --wood-workshops 1 --windmills 4 --battery-heights 10

    factory_data = {key: 0 for key in FACTORY_DATABASE.keys()}
    factory_data[FactoryName.LUMBER_MILLS] = 1
    factory_data[FactoryName.WOOD_WORKSHOPS] = 1
    factories = FactoryConfig(**factory_data)

    energy_data = {key: 0 for key in PRODUCER_DATABASE.keys()}
    energy_data[BatteryName.BATTERY_HEIGHTS] = [10]
    energy_data[ProducerName.WINDMILLS] = 4
    energy_mix = EnergyMixConfig(**energy_data)

    config = SimulationConfig(
        seed=42,
        threads=None,
        samples=1_000_000,
        days=132,
        working_hours=sim_consts.DEFAULT_WORKING_HOURS,
        wet_days=sim_consts.DEFAULT_WET_SEASON_DAYS,
        dry_days=sim_consts.DEFAULT_DRY_SEASON_DAYS,
        badtide_days=sim_consts.DEFAULT_BADTIDE_SEASON_DAYS,
        factories=factories,
        energy_mix=energy_mix,
    )

    # Warm up Numba to ensure compilation time isn't included in the profile
    print("Warming up Numba (compiling jitted functions)...")
    run_simulation(config)

    num_iterations = 5
    print(
        f"Starting Scalene profiling ({num_iterations} iterations of {config.samples:,} samples)..."
    )
    scalene_profiler.start()

    try:
        for i in range(num_iterations):
            print(f"Iteration {i + 1}/{num_iterations}...")
            run_simulation(config)
    finally:
        scalene_profiler.stop()
    print("Scalene profiling stopped.")


if __name__ == "__main__":
    run_profiled_simulation()
