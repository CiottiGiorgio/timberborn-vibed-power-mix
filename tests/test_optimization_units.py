import numpy as np
import pytest
from timberborn_power_mix.optimization import engine
from timberborn_power_mix.optimization.models import OptimizationConfig
from timberborn_power_mix.models import FactoryConfig
from timberborn_power_mix.machines import FACTORY_DATABASE, BatteryName
from timberborn_power_mix.simulation.models import EnergyMixConfig


@pytest.fixture
def opt_config():
    factories_data = {key.value: 0 for key in FACTORY_DATABASE.keys()}
    factories_data["lumber_mills"] = 1
    factories = FactoryConfig(**factories_data)

    return OptimizationConfig(
        seed=42,
        samples=10,
        days=10,
        working_hours=16,
        wet_days=5,
        dry_days=2,
        badtide_days=1,
        factories=factories,
        max_time=5,
    )


def test_power_mix_problem_initialization(opt_config):
    problem = engine.PowerMixProblem(opt_config)
    assert problem.n_var == len(problem.producers) + 2
    assert problem.n_obj == 2
    assert problem.xl is not None
    assert problem.xu is not None


def test_decision_vector_to_mix_conversion(opt_config):
    problem = engine.PowerMixProblem(opt_config)
    n_producers = len(problem.producers)

    # Create a dummy decision vector
    decision_vector = np.zeros(problem.n_var, dtype=int)
    decision_vector[0] = 5  # First producer
    decision_vector[n_producers] = 3  # Num batteries
    decision_vector[n_producers + 1] = 10  # Battery height

    mix = problem._decision_vector_to_mix(decision_vector)

    assert isinstance(mix, EnergyMixConfig)
    # Check first producer count
    producer_name = problem.producers[0].value
    assert getattr(mix, producer_name) == 5

    # Check batteries
    heights = getattr(mix, BatteryName.BATTERY_HEIGHTS.value)
    assert len(heights) == 3
    assert all(h == 10 for h in heights)


def test_evaluate_reproducibility(opt_config):
    problem = engine.PowerMixProblem(opt_config)
    decision_vector = np.zeros(problem.n_var, dtype=int)
    decision_vector[0] = 2
    decision_vector[len(problem.producers)] = 1
    decision_vector[len(problem.producers) + 1] = 5

    out1 = {}
    problem._evaluate(decision_vector, out1)

    out2 = {}
    problem._evaluate(decision_vector, out2)

    assert out1["F"] == out2["F"]
    assert out1["mix"] == out2["mix"]


def test_run_optimization_smoke(opt_config):
    # Short run to ensure everything is wired correctly
    opt_config.max_time = 1
    res = engine.run_optimization(opt_config)

    assert res.best_mix is not None or res.best_cost == 0.0
    assert isinstance(res.best_cost, float)
    assert isinstance(res.unreliability, float)
