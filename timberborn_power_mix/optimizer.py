import random
import numpy as np
from typing import List, Dict, Tuple, Optional
from timberborn_power_mix.models import SimulationParams, EnergyMixParams
from timberborn_power_mix.simulation import run_simulation_batch
from timberborn_power_mix import consts


class OptimizationResult:
    def __init__(
        self,
        params: EnergyMixParams,
        cost: float,
        p95_hours_empty: float,
        total_hours: int,
        energy_surplus: float,
    ):
        self.params = params
        self.cost = cost
        self.p95_hours_empty = p95_hours_empty
        self.total_hours = total_hours
        self.energy_surplus = energy_surplus

    @property
    def p95_empty_percent(self) -> float:
        return (
            (self.p95_hours_empty / self.total_hours) * 100.0
            if self.total_hours > 0
            else 0.0
        )

    @property
    def is_valid(self) -> bool:
        # Valid if reliability is met AND we are not in energy deficit
        return self.p95_empty_percent <= 5.0 and self.energy_surplus >= 0

    def calculate_score(self) -> float:
        """
        Calculates a score to minimize.
        Hierarchy of needs:
        1. Positive Energy Surplus (Production >= Consumption)
        2. Reliability (P95 Empty <= 5%)
        3. Cost
        """
        if self.energy_surplus < 0:
            # Huge penalty for energy deficit
            return 1e12 + abs(self.energy_surplus)

        if self.p95_empty_percent > 5.0:
            # Large penalty for reliability issues
            # We add the percentage to provide a gradient towards 5%
            return 1e9 + self.p95_empty_percent

        # If valid, score is the cost
        return self.cost

    def __repr__(self):
        valid_str = "VALID" if self.is_valid else "INVALID"
        return (
            f"[{valid_str}] Cost: {self.cost:.1f}, "
            f"P95 Empty: {self.p95_empty_percent:.1f}%, "
            f"Surplus: {self.energy_surplus:.1f}, "
            f"Params: {self.params}"
        )


def get_random_params(bounds: Dict[str, Tuple[int, int]]) -> EnergyMixParams:
    return EnergyMixParams(
        power_wheels=random.randint(*bounds.get("power_wheels", (0, 0))),
        water_wheels=random.randint(*bounds.get("water_wheels", (0, 0))),
        large_windmills=random.randint(*bounds.get("large_windmills", (0, 0))),
        windmills=random.randint(*bounds.get("windmills", (0, 0))),
        batteries=random.randint(*bounds.get("batteries", (0, 0))),
        battery_height=random.randint(*bounds.get("battery_height", (1, 1))),
    )


def mutate_params(
    result: OptimizationResult, bounds: Dict[str, Tuple[int, int]]
) -> EnergyMixParams:
    """
    Creates a neighbor by changing one parameter, with bias based on the current state.
    """
    params = result.params
    new_params = params.model_copy(deep=True)

    # Determine bias
    # If we are in deficit or unreliable, we desperately need to ADD stuff.
    # If we are valid (safe), we should try to REMOVE stuff to lower cost (as per user request).

    if result.energy_surplus < 0:
        # Deficit: Bias towards increasing producers
        bias = 1  # Increase
        prob_bias = 0.9
        fields = ["power_wheels", "water_wheels", "large_windmills", "windmills"]
    elif result.p95_empty_percent > 5.0:
        # Unreliable: Bias towards increasing producers or storage
        bias = 1  # Increase
        prob_bias = 0.8
        fields = [
            "power_wheels",
            "water_wheels",
            "large_windmills",
            "windmills",
            "batteries",
            "battery_height",
        ]
    else:
        # Valid (Safe): Bias towards decreasing cost (removing stuff)
        # This directly addresses "Think of any value below the 5% ... as an opportunity to decrease the cost"
        bias = -1  # Decrease
        prob_bias = 0.8
        fields = [
            "power_wheels",
            "water_wheels",
            "large_windmills",
            "windmills",
            "batteries",
            "battery_height",
        ]

    # Pick a field
    field = random.choice(fields)

    # Determine direction
    if random.random() < prob_bias:
        change = bias
    else:
        change = -bias  # Go against bias occasionally to escape local optima

    current_val = getattr(new_params, field)
    min_val, max_val = bounds.get(field, (0, 100))

    # Apply change respecting bounds
    new_val = current_val + change
    new_val = max(min_val, min(new_val, max_val))

    setattr(new_params, field, new_val)

    return new_params


def evaluate_config(
    base_params: SimulationParams,
    mix_params: EnergyMixParams,
    samples_per_config: int,
    total_hours: int,
) -> OptimizationResult:

    current_params = base_params.model_copy(deep=True)
    current_params.energy_mix = mix_params

    batch_results = run_simulation_batch(current_params, samples=samples_per_config)

    hours_empty_list = [r[0] for r in batch_results]
    p95_empty = np.percentile(hours_empty_list, 95)

    # Calculate average surplus at the end of simulation across samples
    surpluses = []
    for _, data in batch_results:
        prod = data.energy_production[-1]
        cons = data.energy_consumption[-1]
        surpluses.append(prod - cons)

    avg_surplus = np.mean(surpluses)
    cost = batch_results[0][1].total_cost

    return OptimizationResult(mix_params, cost, p95_empty, total_hours, avg_surplus)


def optimize(
    base_params: SimulationParams,
    iterations: int = consts.DEFAULT_OPTIMIZATION_ITERATIONS,
    simulations_per_config: int = consts.DEFAULT_SAMPLES_PER_SIM,
    bounds: Optional[Dict[str, Tuple[int, int]]] = None,
) -> List[OptimizationResult]:
    """
    Performs a Local Search (Random Mutation Hill Climbing) to find optimal configurations.
    """

    if bounds is None:
        bounds = {
            "power_wheels": (0, 20),
            "water_wheels": (0, 20),
            "large_windmills": (0, 30),
            "windmills": (0, 30),
            "batteries": (0, 20),
            "battery_height": (1, 20),
        }

    total_hours = base_params.days * consts.HOURS_PER_DAY

    # Start with a random configuration
    current_params = get_random_params(bounds)
    current_result = evaluate_config(
        base_params, current_params, simulations_per_config, total_hours
    )

    history = [current_result]
    best_valid_result = current_result if current_result.is_valid else None

    print(f"Starting Local Search with {iterations} iterations...")
    print(f"Initial: {current_result}")

    for i in range(iterations):
        # Mutate based on current state
        candidate_params = mutate_params(current_result, bounds)
        candidate_result = evaluate_config(
            base_params, candidate_params, simulations_per_config, total_hours
        )

        # Compare
        current_score = current_result.calculate_score()
        candidate_score = candidate_result.calculate_score()

        # We accept if score improves
        if candidate_score < current_score:
            current_result = candidate_result
            history.append(current_result)

            if current_result.is_valid:
                if (
                    best_valid_result is None
                    or current_result.cost < best_valid_result.cost
                ):
                    best_valid_result = current_result

        # Optional: Simulated Annealing could be added here to accept worse solutions with some probability

        if (i + 1) % 10 == 0:
            print(
                f"Iteration {i + 1}/{iterations}. Current Score: {current_score:.0f}. Best Valid Cost: {best_valid_result.cost if best_valid_result else 'None'}"
            )

    return history


def find_optimal_solutions(
    results: List[OptimizationResult], max_empty_percent: float = 5.0
) -> List[OptimizationResult]:
    """
    Returns the best valid solutions found during the search.
    """
    # Filter for valid solutions (Surplus >= 0 and P95 Empty <= 5%)
    valid_solutions = [r for r in results if r.is_valid]

    # Sort by cost (cheapest first)
    valid_solutions.sort(key=lambda x: x.cost)

    # Remove duplicates (by params)
    unique_solutions = []
    seen_params = set()
    for sol in valid_solutions:
        # Create a tuple of values to use as key
        p = sol.params
        key = (
            p.power_wheels,
            p.water_wheels,
            p.large_windmills,
            p.windmills,
            p.batteries,
            p.battery_height,
        )
        if key not in seen_params:
            seen_params.add(key)
            unique_solutions.append(sol)

    return unique_solutions
