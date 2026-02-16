import logging
from timberborn_power_mix.optimization.engine import run_optimization
from timberborn_power_mix.optimization.models import OptimizationConfig
from timberborn_power_mix.models import ConfigName

logger = logging.getLogger(__name__)


def optimization_orchestrator(opt_config: OptimizationConfig):
    """Orchestrates the optimization process and logs the results."""
    iterations = getattr(opt_config, ConfigName.ITERATION)
    logger.info(
        f"Starting AGGRESSIVE guided simulated annealing for {iterations} iterations..."
    )

    best_mix, best_cost = run_optimization(opt_config)

    if best_mix:
        logger.info("Optimization finished!")
        logger.info(f"Best Energy Mix (Cost: {best_cost}):")
        for field, value in best_mix.model_dump().items():
            if value > 0:
                logger.info(f"  {field}: {value}")
    else:
        logger.warning(
            "Could not find a feasible solution within the given iterations."
        )
