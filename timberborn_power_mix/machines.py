from enum import StrEnum
from typing import NamedTuple


class MachineSpec(NamedTuple):
    power: int
    cost: int


class FactoryName(StrEnum):
    LUMBER_MILLS = "lumber_mills"
    GEAR_WORKSHOPS = "gear_workshops"
    STEEL_FACTORIES = "steel_factories"
    WOOD_WORKSHOPS = "wood_workshops"
    PAPER_MILLS = "paper_mills"
    PRINTING_PRESSES = "printing_presses"
    OBSERVATORIES = "observatories"
    BOT_PART_FACTORIES = "bot_part_factories"
    BOT_ASSEMBLERS = "bot_assemblers"
    EXPLOSIVES_FACTORIES = "explosives_factories"
    GRILLMISTS = "grillmists"
    CENTRIFUGES = "centrifuges"


class ProducerName(StrEnum):
    WATER_WHEELS = "water_wheels"
    WINDMILLS = "windmills"
    LARGE_WINDMILLS = "large_windmills"
    POWER_WHEELS = "power_wheels"


class BatteryName(StrEnum):
    BATTERY_HEIGHTS = "battery_heights"


# Consumers
FACTORY_DATABASE: dict[FactoryName, MachineSpec] = {
    FactoryName.LUMBER_MILLS: MachineSpec(power=50, cost=0),
    FactoryName.GEAR_WORKSHOPS: MachineSpec(power=120, cost=0),
    FactoryName.STEEL_FACTORIES: MachineSpec(power=200, cost=0),
    FactoryName.WOOD_WORKSHOPS: MachineSpec(power=250, cost=0),
    FactoryName.PAPER_MILLS: MachineSpec(power=80, cost=0),
    FactoryName.PRINTING_PRESSES: MachineSpec(power=150, cost=0),
    FactoryName.OBSERVATORIES: MachineSpec(power=200, cost=0),
    FactoryName.BOT_PART_FACTORIES: MachineSpec(power=150, cost=0),
    FactoryName.BOT_ASSEMBLERS: MachineSpec(power=250, cost=0),
    FactoryName.EXPLOSIVES_FACTORIES: MachineSpec(power=150, cost=0),
    FactoryName.GRILLMISTS: MachineSpec(power=60, cost=0),
    FactoryName.CENTRIFUGES: MachineSpec(power=200, cost=0),
}

# Producers
PRODUCER_DATABASE: dict[ProducerName, MachineSpec] = {
    ProducerName.WATER_WHEELS: MachineSpec(power=150, cost=50),
    ProducerName.WINDMILLS: MachineSpec(power=150, cost=40),
    ProducerName.LARGE_WINDMILLS: MachineSpec(power=300, cost=75),
    ProducerName.POWER_WHEELS: MachineSpec(power=50, cost=50),
}


class BatterySpec(NamedTuple):
    base_capacity: int
    capacity_per_height: int
    base_cost: int
    cost_per_height: int


GRAVITY_BATTERY = BatterySpec(
    base_capacity=4000,
    capacity_per_height=2000,
    base_cost=84,
    cost_per_height=6,
)


def battery_capacity(height: int) -> int:
    return GRAVITY_BATTERY.base_capacity + (
        height * GRAVITY_BATTERY.capacity_per_height
    )


def battery_cost(height: int) -> int:
    return GRAVITY_BATTERY.base_cost + (height * GRAVITY_BATTERY.cost_per_height)
