import logging
import numpy as np
from typing import Dict, Any
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
from timberborn_power_mix.structures import ConfigName
from timberborn_power_mix.optimization.structures import OptimizationResult

logger = logging.getLogger(__name__)


class PowerMixProblem(ElementwiseProblem):
    """
    Multi-objective optimization problem for finding the optimal power mix.

    Objectives:
    1. Minimize total wood cost.
    2. Minimize unreliability (95th percentile of working hours empty).
    """

    def __init__(self, opt_config: OptimizationConfig, **kwargs: Any):
        self.opt_config = opt_config
        self.producers = list(PRODUCER_DATABASE.keys())
        self.n_producers = len(self.producers)

        # Decision Variables:
        # [0...n_producers-1]: Producer counts
        # [n_producers]: Number of batteries
        # [n_producers+1]: Uniform battery height
        num_variables = self.n_producers + 2

        lower_bounds = np.zeros(num_variables, dtype=int)
        upper_bounds = np.zeros(num_variables, dtype=int)

        # Producer bounds
        upper_bounds[: self.n_producers] = opt_consts.MAX_MACHINES_PER_TYPE

        # Num batteries bounds
        upper_bounds[self.n_producers] = opt_consts.MAX_BATTERIES

        # Uniform battery height bounds
        lower_bounds[self.n_producers + 1] = 1
        upper_bounds[self.n_producers + 1] = opt_consts.MAX_BATTERY_HEIGHT

        super().__init__(
            n_var=num_variables, n_obj=2, xl=lower_bounds, xu=upper_bounds, **kwargs
        )

    def _decision_vector_to_mix(self, decision_vector: np.ndarray) -> EnergyMixConfig:
        """Converts a decision vector into an EnergyMixConfig."""
        mix_data: Dict[str, Any] = {}
        for i, producer in enumerate(self.producers):
            mix_data[producer.value] = int(decision_vector[i])

        num_batteries = int(decision_vector[self.n_producers])
        uniform_height = int(decision_vector[self.n_producers + 1])
        mix_data[BatteryName.BATTERY_HEIGHTS.value] = [uniform_height] * num_batteries

        return EnergyMixConfig(**mix_data)

    def _evaluate(
        self,
        decision_vector: np.ndarray,
        out: Dict[str, Any],
        *args: Any,
        **kwargs: Any,
    ) -> None:
        # 1. Reconstruct EnergyMixConfig
        mix = self._decision_vector_to_mix(decision_vector)

        # 2. Run Simulation
        eval_seed = hash(tuple(decision_vector)) % (2**32)

        # Create SimulationConfig by merging opt_config and mix
        # We exclude max_time and seed from the base config
        sim_base_data = self.opt_config.model_dump()
        sim_base_data.pop(ConfigName.MAX_TIME)
        sim_base_data.pop(ConfigName.SEED)

        config = SimulationConfig(**sim_base_data, energy_mix=mix, seed=eval_seed)

        result = jit_singlethread_simulation_no_plots(
            config.to_jit_config(), sim_helpers.calculate_jit_cached_consts(config)
        )

        # 3. Calculate Objectives
        cost = float(opt_helpers.calculate_total_wood_cost(mix))

        # Objective 2: Minimize Unreliability (95th percentile of working hours empty)
        total_working_hours = getattr(self.opt_config, ConfigName.DAYS) * getattr(
            self.opt_config, ConfigName.WORKING_HOURS
        )
        hours_empty_pct = float(result / total_working_hours)

        out["F"] = [cost, hours_empty_pct]
        out["mix"] = mix


def run_optimization(
    config: OptimizationConfig,
) -> OptimizationResult:
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
        return OptimizationResult(None, 0.0, 0.0)

    # Selection Logic:
    # Find the solution closest to the target unreliability (e.g. 5%)
    best_sol = min(
        res.opt, key=lambda sol: abs(sol.F[1] - opt_consts.TARGET_UNRELIABILITY)
    )

    best_mix = best_sol.get("mix")
    best_cost = best_sol.F[0]
    unreliability = best_sol.F[1]

    logger.info(
        f"Optimization complete. Selected solution with {unreliability:.2%} "
        f"unreliability at cost {best_cost}."
    )

    return OptimizationResult(
        best_mix=best_mix, best_cost=best_cost, unreliability=unreliability
    )
