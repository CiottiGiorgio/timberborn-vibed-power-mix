import numpy as np
import pytest
from timberborn_power_mix.simulation import core, helpers, engine
from timberborn_power_mix.structures import (
    ProducerGroup,
    JitSimulationConfig,
    JitSimulationCachedConsts,
)


@pytest.fixture
def sample_jit_inputs():
    config = JitSimulationConfig(
        seed=42,
        samples=10,
        days=10,
        working_hours=16,
        wet_days=5,
        dry_days=2,
        badtide_days=1,
    )

    sim_consts = JitSimulationCachedConsts(
        total_consumption_rate=1000,
        total_battery_capacity=5000,
        large_windmills=ProducerGroup(2, 400),
        windmills=ProducerGroup(5, 120),
        power_wheels=ProducerGroup(3, 50),
        water_wheels=ProducerGroup(4, 180),
    )
    return config, sim_consts


def test_jit_stochastic_simulation_no_sample_reproducibility(sample_jit_inputs):
    _, sim_consts = sample_jit_inputs
    seed = 12345
    total_hours = 240
    base_surplus = np.random.randint(-500, 500, size=total_hours).astype(np.int64)
    is_working_hour = np.ones(total_hours, dtype=np.bool_)

    res1 = core.jit_stochastic_simulation_no_sample(
        seed,
        total_hours,
        base_surplus,
        is_working_hour,
        sim_consts.total_battery_capacity,
        sim_consts.large_windmills,
        sim_consts.windmills,
    )
    res2 = core.jit_stochastic_simulation_no_sample(
        seed,
        total_hours,
        base_surplus,
        is_working_hour,
        sim_consts.total_battery_capacity,
        sim_consts.large_windmills,
        sim_consts.windmills,
    )

    assert res1 == res2
    assert isinstance(res1, int)


def test_jit_stochastic_simulation_reproducibility(sample_jit_inputs):
    _, sim_consts = sample_jit_inputs
    seed = 12345
    total_hours = 240
    base_surplus = np.random.randint(-500, 500, size=total_hours).astype(np.int64)
    base_power_production = np.random.randint(0, 1000, size=total_hours).astype(
        np.uint32
    )

    res1 = core.jit_stochastic_simulation(
        seed,
        total_hours,
        base_surplus,
        base_power_production,
        sim_consts.total_battery_capacity,
        sim_consts.large_windmills,
        sim_consts.windmills,
    )
    res2 = core.jit_stochastic_simulation(
        seed,
        total_hours,
        base_surplus,
        base_power_production,
        sim_consts.total_battery_capacity,
        sim_consts.large_windmills,
        sim_consts.windmills,
    )

    np.testing.assert_array_equal(res1.power_production, res2.power_production)
    np.testing.assert_array_equal(res1.battery_charge, res2.battery_charge)


def test_jit_simulation_prelude_reproducibility(sample_jit_inputs):
    config, sim_consts = sample_jit_inputs

    res1 = helpers.jit_simulation_prelude(config, sim_consts)
    res2 = helpers.jit_simulation_prelude(config, sim_consts)

    for a, b in zip(res1, res2):
        if isinstance(a, np.ndarray):
            np.testing.assert_array_equal(a, b)
        else:
            assert a == b


def test_jit_batched_simulation_reproducibility(sample_jit_inputs):
    _, sim_consts = sample_jit_inputs
    seeds = np.array([1, 2, 3, 4, 5], dtype=np.uint32)
    total_hours = 240
    base_surplus = np.random.randint(-500, 500, size=total_hours).astype(np.int64)
    is_working_hour = np.ones(total_hours, dtype=np.bool_)

    res1 = engine.jit_batched_simulation(
        seeds, total_hours, base_surplus, is_working_hour, sim_consts
    )
    res2 = engine.jit_batched_simulation(
        seeds, total_hours, base_surplus, is_working_hour, sim_consts
    )

    np.testing.assert_array_equal(res1, res2)


def test_jit_singlethread_simulation_no_plots_reproducibility(sample_jit_inputs):
    config, sim_consts = sample_jit_inputs

    res1 = engine.jit_singlethread_simulation_no_plots(
        config, sim_consts, percentile=95
    )
    res2 = engine.jit_singlethread_simulation_no_plots(
        config, sim_consts, percentile=95
    )

    assert res1 == res2
