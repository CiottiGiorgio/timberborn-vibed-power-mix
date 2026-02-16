from timberborn_power_mix.machines import (
    PRODUCER_DATABASE,
    ProducerName,
    BatteryName,
    battery_cost,
)
from timberborn_power_mix.simulation.models import EnergyMixConfig


def calculate_total_wood_cost(energy_mix: EnergyMixConfig) -> float:
    """Calculates the total wood cost of all machines in the energy mix."""
    wheel_spec = PRODUCER_DATABASE[ProducerName.WATER_WHEEL]
    windmill_spec = PRODUCER_DATABASE[ProducerName.WINDMILL]
    large_windmill_spec = PRODUCER_DATABASE[ProducerName.LARGE_WINDMILL]
    power_wheel_spec = PRODUCER_DATABASE[ProducerName.POWER_WHEEL]

    num_batteries = getattr(energy_mix, BatteryName.BATTERY)
    num_water_wheels = getattr(energy_mix, ProducerName.WATER_WHEEL)
    num_windmills = getattr(energy_mix, ProducerName.WINDMILL)
    num_large_windmills = getattr(energy_mix, ProducerName.LARGE_WINDMILL)
    num_power_wheels = getattr(energy_mix, ProducerName.POWER_WHEEL)
    battery_height = getattr(energy_mix, BatteryName.BATTERY_HEIGHT)

    return (
        (num_power_wheels * power_wheel_spec.cost)
        + (num_water_wheels * wheel_spec.cost)
        + (num_large_windmills * large_windmill_spec.cost)
        + (num_windmills * windmill_spec.cost)
        + (num_batteries * battery_cost(battery_height))
    )
