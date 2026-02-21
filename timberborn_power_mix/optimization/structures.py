from typing import NamedTuple, Optional

from timberborn_power_mix.simulation.models import EnergyMixConfig


class OptimizationResult(NamedTuple):
    """Result of the multi-objective optimization process."""

    best_mix: Optional[EnergyMixConfig]
    best_cost: float
    unreliability: float
