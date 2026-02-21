import logging
import numpy as np
from typing import Tuple, Optional, Dict, Any
from concurrent.futures.thread import ThreadPoolExecutor

from pymoo.core.problem import ElementwiseProblem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.sampling.rnd import IntegerRandomSampling
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.optimize import minimize
from pymoo.termination import get_termination


import timberborn_power_mix.simulation.helpers as sim_helpers
from timberborn_power_mix.simulation.models import SimulationConfig, EnergyMixConfig
from timberborn_power_mix.optimization.models import OptimizationConfig
from timberborn_power_mix.simulation.engine import (
    jit_singlethread_simulation_no_plots,
)
from timberborn_power_mix.machines import PRODUCER_DATABASE, BatteryName
from timberborn_power_mix import helpers
import timberborn_power_mix.optimization.helpers as opt_helpers
from timberborn_power_mix.optimization import consts as opt_consts

logger = logging.getLogger(__name__)

# TODO:
# - check that we can better type the return types passed around in the optimization engine
# - check that we can make more tests on the simulation engine on a more modular level (unit tests, etc.)
# - check that we can make tests for the optimization engine
# - find a good strategy to run tests automatically
# - write ci/cd for tests and linting (not packaging)


class PowerMixProblem(ElementwiseProblem):
    """
    Multi-objective optimization problem for finding the optimal power mix.

    Objectives:
    1. Minimize total wood cost.
    2. Minimize unreliability (95th percentile of working hours empty).
    """

    def __init__(self, opt_config: OptimizationConfig, **kwargs: Any):
        self.opt_config = opt_config
        self.sim_config_base = opt_config.model_dump()
        self.sim_config_base.pop("max_time")
        self.sim_config_base.pop("seed")

        self.producers = list(PRODUCER_DATABASE.keys())
        self.n_producers = len(self.producers)

        # Variables:
        # [0...n_producers-1]: Producer counts
        # [n_producers]: Number of batteries
        # [n_producers+1]: Uniform battery height
        n_var = self.n_producers + 2

        xl = np.zeros(n_var, dtype=int)
        xu = np.zeros(n_var, dtype=int)

        # Producer bounds
        xu[: self.n_producers] = opt_consts.MAX_MACHINES_PER_TYPE

        # Num batteries bounds
        xu[self.n_producers] = opt_consts.MAX_BATTERIES

        # Uniform battery height bounds
        xl[self.n_producers + 1] = 1
        xu[self.n_producers + 1] = opt_consts.MAX_BATTERY_HEIGHT

        super().__init__(n_var=n_var, n_obj=2, xl=xl, xu=xu, **kwargs)

    def _evaluate(
        self, x: np.ndarray, out: Dict[str, Any], *args: Any, **kwargs: Any
    ) -> None:
        # 1. Reconstruct EnergyMixConfig
        mix_data: Dict[str, Any] = {}
        for i, producer in enumerate(self.producers):
            mix_data[producer.value] = int(x[i])

        num_batteries = int(x[self.n_producers])
        uniform_height = int(x[self.n_producers + 1])
        mix_data[BatteryName.BATTERY_HEIGHTS.value] = [uniform_height] * num_batteries

        mix = EnergyMixConfig(**mix_data)

        # 2. Run Simulation
        eval_seed = hash(tuple(x)) % (2**32)
        config = SimulationConfig(
            **self.sim_config_base, energy_mix=mix, seed=eval_seed
        )
        result = jit_singlethread_simulation_no_plots(
            config.to_jit_config(), sim_helpers.calculate_jit_cached_consts(config)
        )

        # 3. Calculate Objectives
        cost = float(opt_helpers.calculate_total_wood_cost(mix))

        # Objective 2: Minimize Unreliability (95th percentile of working hours empty)
        total_working_hours = (
            self.sim_config_base["days"] * self.sim_config_base["working_hours"]
        )
        hours_empty_pct = float(result / total_working_hours)

        out["F"] = [cost, hours_empty_pct]
        out["mix"] = mix


def run_optimization(
    config: OptimizationConfig,
) -> Tuple[Optional[EnergyMixConfig], float]:
    """
    Main NSGA-II Loop using pymoo with parallel evaluation.

    Finds the Pareto front of cost vs. unreliability and selects the solution
    closest to the target unreliability threshold.
    """
    pop_size = opt_consts.POPULATION_SIZE
    n_threads = helpers.calculate_optimal_threads(config.threads, pop_size)

    with ThreadPoolExecutor(max_workers=n_threads) as executor:
        problem = PowerMixProblem(config, elementwise_runner=executor.map)

        algorithm = NSGA2(
            pop_size=pop_size,
            sampling=IntegerRandomSampling(),
            crossover=SBX(
                prob=opt_consts.CROSSOVER_PROBABILITY,
                eta=opt_consts.CROSSOVER_ETA,
                vtype=float,
            ),
            mutation=PM(
                prob=opt_consts.MUTATION_PROBABILITY,
                eta=opt_consts.MUTATION_ETA,
                vtype=float,
            ),
            eliminate_duplicates=True,
        )

        # Stop after either the generation limit or a hardcoded time limit
        termination = get_termination("time", config.max_time)

        res = minimize(
            problem,
            algorithm,
            termination,
            seed=config.seed,
            save_history=False,
            verbose=True,
        )

    if res.opt is None:
        return None, 0.0

    # Selection Logic:
    # Find the solution closest to the target unreliability (e.g. 5%)
    best_sol = min(
        res.opt, key=lambda sol: abs(sol.F[1] - opt_consts.TARGET_UNRELIABILITY)
    )

    best_mix = best_sol.get("mix")
    best_cost = best_sol.F[0]

    logger.info(
        f"Optimization complete. Selected solution with {best_sol.F[1]:.2%} "
        f"unreliability at cost {best_cost}."
    )

    return best_mix, best_cost
