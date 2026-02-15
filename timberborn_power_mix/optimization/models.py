from pydantic import create_model, BaseModel
from timberborn_power_mix.models import ConfigName, CommonConfig

"""
This module defines the configuration models for the power optimization.
Many models are created dynamically using Pydantic's `create_model` to stay in sync 
with the machine databases defined in `machines.py`.

The dynamic structures effectively look like this:

class FactoryConfig(BaseModel):
    lumber_mill: int = 0
    gear_workshop: int = 0
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
    iteration: int
"""

OptimizationConfig = create_model(
    "OptimizationConfig",
    **{ConfigName.ITERATION: int},
    __base__=BaseModel,
)

# Flatten CommonConfig into OptimizationConfig
for name, field in CommonConfig.model_fields.items():
    OptimizationConfig.model_fields[name] = field
OptimizationConfig.model_rebuild(force=True)
