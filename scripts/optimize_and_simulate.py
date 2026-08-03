import logging

from timberborn_power_mix.machines import FACTORY_DATABASE
from timberborn_power_mix.models import FactoryConfig
from timberborn_power_mix.optimization import consts as opt_consts
from timberborn_power_mix.optimization.engine import run_optimization
from timberborn_power_mix.optimization.models import OptimizationConfig
from timberborn_power_mix.simulation import consts as sim_consts
from timberborn_power_mix.simulation.models import SimulationConfig
from timberborn_power_mix.simulation.orchestrator import simulation_orchestrator
from timberborn_power_mix.structures import (
    OptimizeConfigName,
    Percentile,
)

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def main():
    # 1. Define Factory Configuration
    # Initialize all factories to 0, then set specific ones
    factory_data = {name.value: 0 for name in FACTORY_DATABASE}
    factory_data["lumber_mills"] = 1
    factory_data["wood_workshops"] = 1

    factories = FactoryConfig(**factory_data)

    # 2. Define Optimization Configuration using defaults
    opt_config = OptimizationConfig(
        seed=None,
        samples=opt_consts.DEFAULT_OPTIMIZATION_SAMPLES,
        days=sim_consts.DEFAULT_DAYS,
        working_hours=sim_consts.DEFAULT_WORKING_HOURS,
        wet_days=sim_consts.DEFAULT_WET_SEASON_DAYS,
        dry_days=sim_consts.DEFAULT_DRY_SEASON_DAYS,
        badtide_days=sim_consts.DEFAULT_BADTIDE_SEASON_DAYS,
        factories=factories,
        max_time=opt_consts.DEFAULT_MAX_TIME_SECONDS,
        percentile=Percentile.P95,
        target_unreliability=opt_consts.DEFAULT_TARGET_UNRELIABILITY,
    )

    # 3. Run Optimization
    logger.info("--- Phase 1: Optimization ---")
    try:
        opt_res = run_optimization(opt_config)
    except RuntimeError as e:
        logger.error(f"Optimization failed: {e}")
        return

    logger.info(f"Optimization finished! Best wood cost: {opt_res.best_cost}")
    logger.info(f"Selected Mix: {opt_res.best_mix}")

    # 4. Run Simulation with the optimized mix
    logger.info("\n--- Phase 2: Simulation & Visualization ---")

    # Create SimulationConfig from OptimizationConfig and the best mix
    sim_data = opt_config.model_dump()
    sim_data.pop(OptimizeConfigName.MAX_TIME)
    sim_data.pop(OptimizeConfigName.TARGET_UNRELIABILITY)
    sim_data.pop(OptimizeConfigName.PERCENTILE)

    sim_config = SimulationConfig(**sim_data, energy_mix=opt_res.best_mix)

    simulation_orchestrator(sim_config)


if __name__ == "__main__":
    main()
