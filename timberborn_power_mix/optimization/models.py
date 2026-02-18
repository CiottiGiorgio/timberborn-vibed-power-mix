from typing import List
from pydantic import create_model, BaseModel
from timberborn_power_mix.models import ConfigName, CommonConfig
from timberborn_power_mix.simulation.models import EnergyMixConfig

"""
This module defines the configuration models for the power optimization.
Many models are created dynamically using Pydantic's `create_model` to stay in sync 
with the machine databases defined in `machines.py`.

The dynamic structures effectively look like this:

class FactoryConfig(BaseModel):
    lumber_mills: int = 0
    gear_workshops: int = 0
    ... (all other factories)

class OptimizationConfig(BaseModel):
    seed: Optional[int] = None
    samples: int
    days: int
    working_hours: int
    wet_days: int
    dry_days: int
    badtide_days: int
    factories: FactoryConfig
    iterations: int
"""

OptimizationConfig = create_model(
    "OptimizationConfig",
    **{str(ConfigName.ITERATIONS): int},
    __base__=BaseModel,
)

# Flatten CommonConfig into OptimizationConfig
for name, field in CommonConfig.model_fields.items():
    OptimizationConfig.model_fields[name] = field
OptimizationConfig.model_rebuild(force=True)


class Individual:
    """Represents a single power grid configuration in the population."""

    def __init__(self, mix: EnergyMixConfig):  # type: ignore[valid-type]
        self.mix = mix
        self.cost: float = 0.0
        self.battery_stress: float = 0.0  # Objective 1: Minimize
        self.hours_empty_pct: float = 0.0  # Selection criteria

        self.rank: int = 0
        self.crowding_distance: float = 0.0
        self.domination_count: int = 0
        self.dominated_solutions: List["Individual"] = []

    def set_results(
        self, cost: float, battery_stress: float, hours_empty_pct: float
    ) -> None:
        """Sets the evaluation results calculated externally."""
        self.cost = cost
        self.battery_stress = battery_stress
        self.hours_empty_pct = hours_empty_pct
