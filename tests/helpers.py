import numpy as np
from timberborn_power_mix.simulation.models import (
    SimulationConfig,
    EnergyMixConfig,
    FactoryConfig,
)
from timberborn_power_mix.simulation.core import run_simulation
from timberborn_power_mix.plots.canvas import create_simulation_figure
from timberborn_power_mix.machines import (
    FACTORY_DATABASE,
    PRODUCER_DATABASE,
    BatteryName,
    FactoryName,
    ProducerName,
)
from tests import consts


def generate_reference_simulation_data():
    """
    Runs a deterministic simulation and returns the data for the worst-case scenario.
    """
    # 1. Setup parameters
    # We must provide all fields because the models don't have defaults
    factory_data = {key: 0 for key in FACTORY_DATABASE.keys()}
    factory_data[FactoryName.LUMBER_MILL] = 1
    factory_data[FactoryName.WOOD_WORKSHOP] = 1
    factories = FactoryConfig(**factory_data)

    energy_data = {key: 0 for key in PRODUCER_DATABASE.keys()}
    energy_data[BatteryName.BATTERY] = 1
    energy_data[BatteryName.BATTERY_HEIGHT] = 1.0
    energy_data[ProducerName.WINDMILL] = 4
    energy_mix = EnergyMixConfig(**energy_data)

    config = SimulationConfig(
        samples=consts.DEFAULT_SAMPLES_PER_SIM,
        days=consts.DEFAULT_DAYS,
        wet_days=consts.DEFAULT_WET_SEASON_DAYS,
        dry_days=consts.DEFAULT_DRY_SEASON_DAYS,
        badtide_days=consts.DEFAULT_BADTIDE_SEASON_DAYS,
        working_hours=consts.DEFAULT_WORKING_HOURS,
        energy_mix=energy_mix,
        factories=factories,
    )

    # 2. Run simulation
    # We seed numpy for determinism in the simulation
    np.random.seed(42)
    res = run_simulation(config)

    return res.worst_sample, res.hours_empty_results, config


def generate_reference_figure():
    """
    Generates the reference plot figure.
    """
    worst_run_data, run_empty_hours, config = generate_reference_simulation_data()
    fig = create_simulation_figure(worst_run_data, config, run_empty_hours)
    return fig
