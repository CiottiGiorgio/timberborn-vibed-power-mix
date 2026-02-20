import logging
from timberborn_power_mix.optimization.engine import run_optimization
from timberborn_power_mix.optimization.models import OptimizationConfig
from timberborn_power_mix.structures import ConfigName

logger = logging.getLogger(__name__)


def optimization_orchestrator(config: OptimizationConfig) -> None:
    """Orchestrates the multi-objective NSGA-II optimization process."""
    iterations = getattr(config, ConfigName.ITERATIONS)
    logger.info(
        f"Starting NSGA-II Multi-Objective Optimization for {iterations} generations..."
    )
    logger.info("Objectives: Minimize Cost & Minimize Battery Stress")

    best_mix, best_cost = run_optimization(config)

    if best_mix:
        logger.info("Optimization finished!")
        logger.info(f"Selected Energy Mix (Total Wood Cost: {best_cost}):")
        for field, value in best_mix.model_dump().items():
            # Handle both numeric counts and lists (like battery_height)
            if (isinstance(value, (int, float)) and value > 0) or (
                isinstance(value, list) and len(value) > 0
            ):
                logger.info(f"  {field}: {value}")
    else:
        logger.warning("Could not find a valid solution within the given iterations.")
