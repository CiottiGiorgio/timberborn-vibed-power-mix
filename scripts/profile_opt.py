import logging
from scalene import scalene_profiler
from timberborn_power_mix.optimization.engine import run_optimization
from timberborn_power_mix.optimization.models import OptimizationConfig
from timberborn_power_mix.models import FactoryConfig
from timberborn_power_mix.machines import (
    FACTORY_DATABASE,
    FactoryName,
)
from timberborn_power_mix.simulation import consts as sim_consts

# Configure logging
logging.basicConfig(level=logging.INFO)


def run_profiled_optimization():
    # Configuration based on 'optimize-bot' run configuration:
    # --working-hours 24 --bot-part-factory 3 --bot-assembler 1 --iteration 100

    factory_data = {key: 0 for key in FACTORY_DATABASE.keys()}
    factory_data[FactoryName.BOT_PART_FACTORY] = 3
    factory_data[FactoryName.BOT_ASSEMBLER] = 1
    factories = FactoryConfig(**factory_data)

    config = OptimizationConfig(
        iteration=100,
        samples=5000,
        days=sim_consts.DEFAULT_DAYS,
        wet_days=sim_consts.DEFAULT_WET_SEASON_DAYS,
        dry_days=sim_consts.DEFAULT_DRY_SEASON_DAYS,
        badtide_days=sim_consts.DEFAULT_BADTIDE_SEASON_DAYS,
        working_hours=24,
        factories=factories,
        seed=42,
    )

    num_runs = 5
    print(f"Starting Scalene profiling for optimization ({num_runs} runs)...")
    scalene_profiler.start()

    try:
        for i in range(num_runs):
            run_optimization(config)
    finally:
        scalene_profiler.stop()
    print("Scalene profiling stopped.")


if __name__ == "__main__":
    run_profiled_optimization()
