from timberborn_power_mix.machines import (
    PRODUCER_DATABASE,
    BatteryName,
    battery_cost,
)
from timberborn_power_mix.simulation.models import EnergyMixConfig


def calculate_total_wood_cost(energy_mix: EnergyMixConfig) -> float:
    """Calculates total cost by iterating over the mix fields."""
    mix_dict = energy_mix.model_dump()

    # Sum costs for all producers in the database
    producer_cost = sum(
        mix_dict[name] * spec.cost for name, spec in PRODUCER_DATABASE.items()
    )

    # Add battery costs from the list of heights
    heights = mix_dict[BatteryName.BATTERY_HEIGHT]
    battery_total_cost = sum(battery_cost(h) for h in heights)

    return producer_cost + battery_total_cost
