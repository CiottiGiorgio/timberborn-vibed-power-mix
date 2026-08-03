import logging

from scalene import scalene_profiler

from timberborn_power_mix.machines import (
    FACTORY_DATABASE,
    FactoryName,
)
from timberborn_power_mix.models import FactoryConfig
from timberborn_power_mix.optimization import consts as opt_consts
from timberborn_power_mix.optimization.engine import run_optimization
from timberborn_power_mix.optimization.models import OptimizationConfig
from timberborn_power_mix.simulation import consts as sim_consts

# Configure logging
logging.basicConfig(level=logging.INFO)


def run_profiled_optimization():
    # Configuration based on 'optimize-bot' run configuration:
    # --working-hours 24 --bot-part-factories 3 --bot-assemblers 1 --iterations 10

    factory_data = {key: 0 for key in FACTORY_DATABASE}
    factory_data[FactoryName.BOT_PART_FACTORIES] = 3
    factory_data[FactoryName.BOT_ASSEMBLERS] = 1
    factories = FactoryConfig(**factory_data)

    # Use a smaller number of iterations and samples for profiling
    config = OptimizationConfig(
        iterations=10,
        samples=opt_consts.DEFAULT_OPTIMIZATION_SAMPLES,
        days=sim_consts.DEFAULT_DAYS,
        wet_days=sim_consts.DEFAULT_WET_SEASON_DAYS,
        dry_days=sim_consts.DEFAULT_DRY_SEASON_DAYS,
        badtide_days=sim_consts.DEFAULT_BADTIDE_SEASON_DAYS,
        working_hours=24,
        factories=factories,
        seed=42,
        threads=None,  # Use default (all cores)
    )

    # Warm up Numba to ensure compilation time isn't included in the profile
    print("Warming up Numba (compiling jitted functions)...")
    warmup_config = config.model_copy(update={"iterations": 1, "samples": 1, "days": 1})
    run_optimization(warmup_config)

    num_runs = 2
    print(f"Starting Scalene profiling for optimization ({num_runs} runs)...")
    scalene_profiler.start()

    try:
        for i in range(num_runs):
            print(f"Run {i + 1}/{num_runs}...")
            run_optimization(config)
    finally:
        scalene_profiler.stop()
    print("Scalene profiling stopped.")


if __name__ == "__main__":
    run_profiled_optimization()
