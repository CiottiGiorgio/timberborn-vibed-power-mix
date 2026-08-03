import logging

from timberborn_power_mix.optimization.engine import run_optimization
from timberborn_power_mix.optimization.models import OptimizationConfig
from timberborn_power_mix.structures import OptimizeConfigName

logger = logging.getLogger(__name__)


def optimization_orchestrator(config: OptimizationConfig) -> None:
    """Orchestrates the multi-objective NSGA-II optimization process."""
    max_time = getattr(config, OptimizeConfigName.MAX_TIME)
    logger.info(
        f"Starting NSGA-II Multi-Objective Optimization (Max Time: {max_time}s)..."
    )
    logger.info("Objectives: Minimize Cost & Minimize Battery Stress")

    try:
        res = run_optimization(config)
    except RuntimeError as e:
        logger.error(f"Optimization failed: {e}")
        return

    logger.info("Optimization finished!")
    logger.info(f"Selected Energy Mix (Total Wood Cost: {res.best_cost}):")
    for field, value in res.best_mix.model_dump().items():
        # Handle both numeric counts and lists (like battery_height)
        if (isinstance(value, (int, float)) and value > 0) or (
            isinstance(value, list) and len(value) > 0
        ):
            logger.info(f"  {field}: {value}")
