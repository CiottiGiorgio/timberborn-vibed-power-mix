import logging

import numpy as np
from scalene import scalene_profiler
from timberborn_power_mix.simulation.engine import run_simulation_multithreaded
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
import timberborn_power_mix.simulation.helpers as sim_helpers
import timberborn_power_mix.helpers as helpers

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
        samples=1_000_000,
        days=132,
        wet_days=sim_consts.DEFAULT_WET_SEASON_DAYS,
        dry_days=sim_consts.DEFAULT_DRY_SEASON_DAYS,
        badtide_days=sim_consts.DEFAULT_BADTIDE_SEASON_DAYS,
        working_hours=sim_consts.DEFAULT_WORKING_HOURS,
        energy_mix=energy_mix,
        factories=factories,
        seed=42,
        threads=helpers.calculate_optimal_threads(None, 1_000_000),
    )

    # Warm up Numba to ensure compilation time isn't included in the profile
    print("Warming up Numba (compiling jitted functions)...")
    jit_config = config.to_jit_config()
    cached_consts = sim_helpers.calculate_jit_cached_consts(config)

    # Generate seeds for all samples
    ss = np.random.SeedSequence(config.seed)
    all_seeds = ss.generate_state(config.samples)

    run_simulation_multithreaded(
        jit_config,
        cached_consts,
        all_seeds,
    )

    num_iterations = 5
    print(
        f"Starting Scalene profiling ({num_iterations} iterations of {config.samples:,} samples)..."
    )
    scalene_profiler.start()

    try:
        for i in range(num_iterations):
            print(f"Iteration {i + 1}/{num_iterations}...")
            _result = run_simulation_multithreaded(
                jit_config,
                cached_consts,
                all_seeds,
            )
    finally:
        scalene_profiler.stop()
    print("Scalene profiling stopped.")


if __name__ == "__main__":
    run_profiled_simulation()
