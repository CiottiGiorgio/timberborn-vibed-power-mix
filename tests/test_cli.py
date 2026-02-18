from click.testing import CliRunner
from timberborn_power_mix.cli import cli
from unittest.mock import patch


def test_simulate_no_options():
    """
    Ensures that 'tb-power-mix simulate' runs successfully with no options,
    relying on CLI-defined defaults.
    """
    runner = CliRunner()

    # We patch the orchestrator and plt.show to avoid actually running the simulation
    # and opening a window during tests.
    with (
        patch(
            "timberborn_power_mix.simulation.orchestrator.simulation_orchestrator"
        ) as mock_orchestrator,
        patch("matplotlib.pyplot.show"),
    ):
        result = runner.invoke(cli, ["simulate"])

        # Check that the command executed successfully
        assert result.exit_code == 0

        # Verify that the orchestrator was called
        assert mock_orchestrator.called

        # Verify that the config passed to the orchestrator has the expected defaults
        config = mock_orchestrator.call_args[0][0]

        # Check a few default values from sim_consts (via cli.py)
        assert config.samples > 0
        assert config.days > 0
        assert config.working_hours > 0


def test_optimize_no_options():
    """
    Ensures that 'tb-power-mix optimize' runs successfully with no options,
    relying on CLI-defined defaults.
    """
    runner = CliRunner()

    with patch(
        "timberborn_power_mix.optimization.orchestrator.optimization_orchestrator"
    ) as mock_orchestrator:
        result = runner.invoke(cli, ["optimize"])

        assert result.exit_code == 0
        assert mock_orchestrator.called

        config = mock_orchestrator.call_args[0][0]
        assert config.iterations > 0
        assert config.samples > 0
