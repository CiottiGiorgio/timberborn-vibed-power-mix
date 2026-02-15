import logging
from scalene import scalene_profiler
from timberborn_power_mix.simulation.core import run_simulation
from timberborn_power_mix.simulation.models import (
    SimulationConfig,
    FactoryConfig,
    EnergyMixConfig,
)
from timberborn_power_mix.machines import (
    FACTORY_DATABASE,
    PRODUCER_DATABASE,
    FactoryName,
    ProducerName,
    BatteryName,
)
from timberborn_power_mix import consts as sim_consts

# Configure logging
logging.basicConfig(level=logging.INFO)


def run_profiled_simulation():
    # Configuration based on 'simulate-simple' run configuration:
    # --lumber-mill 1 --wood-workshop 1 --windmill 4 --battery 1 --battery-height 1

    factory_data = {key: 0 for key in FACTORY_DATABASE.keys()}
    factory_data[FactoryName.LUMBER_MILL] = 1
    factory_data[FactoryName.WOOD_WORKSHOP] = 1
    factories = FactoryConfig(**factory_data)

    energy_data = {key: 0 for key in PRODUCER_DATABASE.keys()}
    energy_data[BatteryName.BATTERY] = 1
    energy_data[BatteryName.BATTERY_HEIGHT] = 1.0
    energy_data[ProducerName.WINDMILL] = 4
    energy_mix = EnergyMixConfig(**energy_data)

    config = SimulationConfig(
        samples=50_000,
        days=132,
        wet_days=sim_consts.DEFAULT_WET_SEASON_DAYS,
        dry_days=sim_consts.DEFAULT_DRY_SEASON_DAYS,
        badtide_days=sim_consts.DEFAULT_BADTIDE_SEASON_DAYS,
        working_hours=sim_consts.DEFAULT_WORKING_HOURS,
        energy_mix=energy_mix,
        factories=factories,
        seed=42,
    )

    # Warm up Numba to ensure compilation time isn't included in the profile
    print("Warming up Numba (compiling jitted functions)...")
    warmup_config = config.model_copy(update={"samples": 1, "days": 1})
    run_simulation(warmup_config)

    num_iterations = 5
    print(
        f"Starting Scalene profiling ({num_iterations} iterations of {config.samples:,} samples)..."
    )
    scalene_profiler.start()

    try:
        for i in range(num_iterations):
            _result = run_simulation(config)
    finally:
        scalene_profiler.stop()
    print("Scalene profiling stopped.")


if __name__ == "__main__":
    run_profiled_simulation()
