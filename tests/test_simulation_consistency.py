import numpy as np
import pytest
from unittest.mock import patch
from timberborn_power_mix.simulation.models import SimulationConfig, EnergyMixConfig
from timberborn_power_mix.simulation.engine import (
    jit_singlethread_simulation,
    jit_multithread_simulation,
)
from timberborn_power_mix.simulation.orchestrator import (
    simulation_orchestrator,
    run_simulation,
)
from timberborn_power_mix.machines import (
    BatteryName,
    ProducerName,
    PRODUCER_DATABASE,
    FACTORY_DATABASE,
    FactoryName,
)
from timberborn_power_mix.models import FactoryConfig
from timberborn_power_mix.structures import (
    SimulationResult,
    SimulationSample,
    AggregatedSamples,
)
import timberborn_power_mix.simulation.helpers as sim_helpers
import timberborn_power_mix.helpers as helpers
import timberborn_power_mix.simulation.consts as sim_consts


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
    jit_config = simulation_config.to_jit_config()
    sim_consts_jit = sim_helpers.calculate_jit_cached_consts(simulation_config)

    # Run single-threaded simulation
    res_single = jit_singlethread_simulation(jit_config, sim_consts_jit)

    # Run multi-threaded simulation
    threads = helpers.calculate_optimal_threads(
        simulation_config.threads, simulation_config.samples
    )
    res_multi = jit_multithread_simulation(jit_config, threads, sim_consts_jit)

    # 1. Check Aggregated Metrics (Lost Hours)
    np.testing.assert_array_equal(
        res_single.aggregated_samples.lost_working_hours_results,
        res_multi.aggregated_samples.lost_working_hours_results,
        err_msg="Aggregated lost_working_hours_results differ between single and multi-thread runs",
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
    Verifies that running the same simulation twice
    with the same seed produces identical results.
    """
    res1 = run_simulation(simulation_config)
    res2 = run_simulation(simulation_config)

    np.testing.assert_array_equal(
        res1.aggregated_samples.lost_working_hours_results,
        res2.aggregated_samples.lost_working_hours_results,
        err_msg="Simulation is not deterministic across repeated runs",
    )


@patch("timberborn_power_mix.simulation.orchestrator.jit_multithread_simulation")
@patch("timberborn_power_mix.simulation.orchestrator.jit_singlethread_simulation")
@patch("timberborn_power_mix.simulation.orchestrator.create_simulation_figure")
@patch("timberborn_power_mix.simulation.orchestrator.plt.show")
def test_orchestrator_calls_correct_engine(
    _mock_show, _mock_create_fig, mock_single, mock_multi, simulation_config
):
    """
    Verifies that run_simulation calls the correct engine function
    based on the number of threads in the configuration.
    """
    # Create a dummy result to avoid errors in orchestrator
    total_hours = simulation_config.days * sim_consts.HOURS_PER_DAY
    dummy_res = SimulationResult(
        p95_sample=SimulationSample(
            power_production=np.zeros(total_hours, dtype=np.uint32),
            battery_charge=np.zeros(total_hours, dtype=np.uint32),
        ),
        aggregated_samples=AggregatedSamples(
            power_consumption=np.zeros(total_hours, dtype=np.uint32),
            lost_working_hours_results=np.zeros(
                simulation_config.samples, dtype=np.float64
            ),
        ),
    )
    mock_multi.return_value = dummy_res
    mock_single.return_value = dummy_res

    # Case 1: threads > 1 -> should call multithread
    config_multi = simulation_config.model_copy(update={"threads": 4})
    simulation_orchestrator(config_multi)

    mock_multi.assert_called_once()
    mock_single.assert_not_called()

    # Reset mocks
    mock_multi.reset_mock()
    mock_single.reset_mock()

    # Case 2: threads = 1 -> should call singlethread
    config_single = simulation_config.model_copy(update={"threads": 1})
    simulation_orchestrator(config_single)

    mock_single.assert_called_once()
    mock_multi.assert_not_called()

    # Reset mocks
    mock_multi.reset_mock()
    mock_single.reset_mock()

    # Case 3: threads = None -> should call multithread
    config_none = simulation_config.model_copy(update={"threads": None})
    simulation_orchestrator(config_none)

    mock_multi.assert_called_once()
    mock_single.assert_not_called()
