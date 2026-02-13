from enum import Enum
from typing import NamedTuple, Dict


class MachineSpec(NamedTuple):
    power: int
    cost: int


class FactoryName(str, Enum):
    LUMBER_MILL = "lumber_mill"
    GEAR_WORKSHOP = "gear_workshop"
    STEEL_FACTORY = "steel_factory"
    WOOD_WORKSHOP = "wood_workshop"
    PAPER_MILL = "paper_mill"
    PRINTING_PRESS = "printing_press"
    OBSERVATORY = "observatory"
    BOT_PART_FACTORY = "bot_part_factory"
    BOT_ASSEMBLER = "bot_assembler"
    EXPLOSIVES_FACTORY = "explosives_factory"
    GRILLMIST = "grillmist"
    CENTRIFUGE = "centrifuge"


class ProducerName(str, Enum):
    WATER_WHEEL = "water_wheel"
    WINDMILL = "windmill"
    LARGE_WINDMILL = "large_windmill"
    POWER_WHEEL = "power_wheel"


class BatteryName(str, Enum):
    BATTERY = "battery"
    BATTERY_HEIGHT = "battery_height"


# Consumers
FACTORY_DATABASE: Dict[FactoryName, MachineSpec] = {
    FactoryName.LUMBER_MILL: MachineSpec(power=50, cost=0),
    FactoryName.GEAR_WORKSHOP: MachineSpec(power=120, cost=0),
    FactoryName.STEEL_FACTORY: MachineSpec(power=200, cost=0),
    FactoryName.WOOD_WORKSHOP: MachineSpec(power=250, cost=0),
    FactoryName.PAPER_MILL: MachineSpec(power=80, cost=0),
    FactoryName.PRINTING_PRESS: MachineSpec(power=150, cost=0),
    FactoryName.OBSERVATORY: MachineSpec(power=200, cost=0),
    FactoryName.BOT_PART_FACTORY: MachineSpec(power=150, cost=0),
    FactoryName.BOT_ASSEMBLER: MachineSpec(power=250, cost=0),
    FactoryName.EXPLOSIVES_FACTORY: MachineSpec(power=150, cost=0),
    FactoryName.GRILLMIST: MachineSpec(power=60, cost=0),
    FactoryName.CENTRIFUGE: MachineSpec(power=200, cost=0),
}

# Producers
PRODUCER_DATABASE: Dict[ProducerName, MachineSpec] = {
    ProducerName.WATER_WHEEL: MachineSpec(power=150, cost=50),
    ProducerName.WINDMILL: MachineSpec(power=150, cost=40),
    ProducerName.LARGE_WINDMILL: MachineSpec(power=300, cost=75),
    ProducerName.POWER_WHEEL: MachineSpec(power=50, cost=50),
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


def battery_capacity(height: float) -> float:
    return GRAVITY_BATTERY.base_capacity + (
        height * GRAVITY_BATTERY.capacity_per_height
    )


def battery_cost(height: float) -> float:
    return GRAVITY_BATTERY.base_cost + (height * GRAVITY_BATTERY.cost_per_height)
