from typing import NamedTuple

from timberborn_power_mix.simulation.models import EnergyMixConfig


class OptimizationResult(NamedTuple):
    """Result of the multi-objective optimization process."""

    best_mix: EnergyMixConfig
    best_cost: int
    unreliability: float
