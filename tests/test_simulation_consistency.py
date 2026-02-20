import numpy as np
import pytest
from unittest.mock import patch
from timberborn_power_mix.simulation.models import SimulationConfig, EnergyMixConfig
from timberborn_power_mix.simulation.engine import (
    run_simulation_singlethread,
    run_simulation_multithread,
)
from timberborn_power_mix.simulation.orchestrator import simulation_orchestrator
from timberborn_power_mix.machines import (
    BatteryName,
    ProducerName,
    PRODUCER_DATABASE,
    FACTORY_DATABASE,
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
    np.testing.assert_array_equal(
        res_single.aggregated_samples.hours_empty_results,
        res_multi.aggregated_samples.hours_empty_results,
        err_msg="Aggregated hours_empty_results differ between single and multi-thread runs",
    )

    # 2. Check p95 Sample Data
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


@patch("timberborn_power_mix.simulation.orchestrator.run_simulation_multithread")
@patch("timberborn_power_mix.simulation.orchestrator.run_simulation_singlethread")
@patch("timberborn_power_mix.simulation.orchestrator.create_simulation_figure")
@patch("timberborn_power_mix.simulation.orchestrator.plt.show")
def test_orchestrator_calls_correct_engine(
    mock_show, mock_create_fig, mock_single, mock_multi, simulation_config
):
    """
    Verifies that simulation_orchestrator calls the correct engine function
    based on the number of threads in the configuration.
    """
    # Case 1: threads > 1 -> should call multithread
    config_multi = simulation_config.model_copy(update={"threads": 4})
    simulation_orchestrator(config_multi)

    mock_multi.assert_called_once_with(config_multi)
    mock_single.assert_not_called()

    # Reset mocks
    mock_multi.reset_mock()
    mock_single.reset_mock()

    # Case 2: threads = 1 -> should call singlethread
    config_single = simulation_config.model_copy(update={"threads": 1})
    simulation_orchestrator(config_single)

    mock_single.assert_called_once_with(config_single)
    mock_multi.assert_not_called()

    # Reset mocks
    mock_multi.reset_mock()
    mock_single.reset_mock()

    # Case 3: threads = None -> should call multithread
    config_none = simulation_config.model_copy(update={"threads": None})
    simulation_orchestrator(config_none)

    mock_multi.assert_called_once_with(config_none)
    mock_single.assert_not_called()
