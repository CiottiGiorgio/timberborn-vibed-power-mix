import numpy as np
import pytest
from timberborn_power_mix.simulation.models import SimulationConfig, EnergyMixConfig
from timberborn_power_mix.simulation.engine import (
    run_simulation_singlethread,
    run_simulation_multithread,
)
from timberborn_power_mix.machines import (
    BatteryName,
    ProducerName,
    FACTORY_DATABASE,
    PRODUCER_DATABASE,
    FactoryName,
)
from timberborn_power_mix.models import FactoryConfig


@pytest.fixture
def simulation_config():
    """Creates a standard simulation configuration for testing."""
    mix_data: dict[str, int | list[int]] = {key: 0 for key in PRODUCER_DATABASE.keys()}
    mix_data[ProducerName.WATER_WHEELS] = 5
    mix_data[ProducerName.WINDMILLS] = 5
    mix_data[ProducerName.LARGE_WINDMILLS] = 2
    mix_data[ProducerName.POWER_WHEELS] = 5
    mix_data[BatteryName.BATTERY_HEIGHTS] = [5, 5, 5]
    energy_mix = EnergyMixConfig(**mix_data)

    factories_data = {key: 0 for key in FACTORY_DATABASE.keys()}
    factories_data[FactoryName.LUMBER_MILLS] = 2
    factories = FactoryConfig(**factories_data)

    return SimulationConfig(
        seed=42,
        threads=4,
        samples=100,
        days=30,
        working_hours=16,
        wet_days=5,
        dry_days=2,
        badtide_days=1,
        factories=factories,
        energy_mix=energy_mix,
    )


def test_single_vs_multithread_consistency(simulation_config):
    """
    Verifies that single-threaded and multi-threaded simulations produce
    identical results when initialized with the same seed.
    """
    # Run single-threaded simulation
    res_single = run_simulation_singlethread(simulation_config)

    # Run multi-threaded simulation
    res_multi = run_simulation_multithread(simulation_config)

    # 1. Check Aggregated Metrics (Hours Empty)
    # We sort them because the order might differ in multithreading due to chunking/aggregation
    # although our current implementation concatenates them in order of chunks.
    # To be safe and robust, we compare sorted arrays or ensure strict ordering.
    # Our implementation:
    #   Single: seeds generated 0..N
    #   Multi: seeds generated 0..N, split into chunks, processed, then concatenated.
    #   So the order SHOULD be preserved exactly.

    np.testing.assert_array_equal(
        res_single.aggregated_samples.hours_empty_results,
        res_multi.aggregated_samples.hours_empty_results,
        err_msg="Aggregated hours_empty_results differ between single and multi-thread runs",
    )

    # 2. Check p95 Sample Data
    # The p95 sample is reconstructed using the same seed logic.
    # If the aggregated results are identical, the p95 index and seed should be identical,
    # and thus the reconstructed sample should be identical.

    np.testing.assert_array_equal(
        res_single.p95_sample.power_production,
        res_multi.p95_sample.power_production,
        err_msg="p95 sample power_production differs",
    )

    np.testing.assert_array_equal(
        res_single.p95_sample.battery_charge,
        res_multi.p95_sample.battery_charge,
        err_msg="p95 sample battery_charge differs",
    )

    # 3. Check Power Consumption Profile
    np.testing.assert_array_equal(
        res_single.aggregated_samples.power_consumption,
        res_multi.aggregated_samples.power_consumption,
        err_msg="Power consumption profile differs",
    )


def test_seed_determinism(simulation_config):
    """
    Verifies that running the same simulation twice (single-threaded)
    with the same seed produces identical results.
    """
    res1 = run_simulation_singlethread(simulation_config)
    res2 = run_simulation_singlethread(simulation_config)

    np.testing.assert_array_equal(
        res1.aggregated_samples.hours_empty_results,
        res2.aggregated_samples.hours_empty_results,
        err_msg="Simulation is not deterministic across repeated runs",
    )
