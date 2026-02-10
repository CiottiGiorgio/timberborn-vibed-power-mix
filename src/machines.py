from typing import NamedTuple, Dict, Iterator, Tuple, Union


class MachineSpec(NamedTuple):
    power: int
    cost: int
    is_producer: bool = False


# Consumers
lumber_mill = MachineSpec(power=50, cost=0)
gear_workshop = MachineSpec(power=120, cost=0)
steel_factory = MachineSpec(power=200, cost=0)
wood_workshop = MachineSpec(power=250, cost=0)
paper_mill = MachineSpec(power=80, cost=0)
printing_press = MachineSpec(power=150, cost=0)
observatory = MachineSpec(power=200, cost=0)
bot_part_factory = MachineSpec(power=150, cost=0)
bot_assembler = MachineSpec(power=250, cost=0)
explosives_factory = MachineSpec(power=150, cost=0)
grillmist = MachineSpec(power=60, cost=0)
centrifuge = MachineSpec(power=200, cost=0)

# Producers
power_wheel = MachineSpec(power=50, cost=50, is_producer=True)
water_wheel = MachineSpec(power=150, cost=50, is_producer=True)
large_windmill = MachineSpec(power=300, cost=75, is_producer=True)
windmill = MachineSpec(power=150, cost=40, is_producer=True)


class BatterySpec:
    base_capacity = 4000
    capacity_per_height = 2000
    base_cost = 84
    cost_per_height = 6

    @classmethod
    def calculate_capacity(cls, height: Union[int, float]) -> float:
        return cls.base_capacity + (height * cls.capacity_per_height)

    @classmethod
    def calculate_cost(cls, height: Union[int, float]) -> float:
        return cls.base_cost + (height * cls.cost_per_height)


# Registry
ALL_MACHINES: Dict[str, MachineSpec] = {
    "lumber_mill": lumber_mill,
    "gear_workshop": gear_workshop,
    "steel_factory": steel_factory,
    "wood_workshop": wood_workshop,
    "paper_mill": paper_mill,
    "printing_press": printing_press,
    "observatory": observatory,
    "bot_part_factory": bot_part_factory,
    "bot_assembler": bot_assembler,
    "explosives_factory": explosives_factory,
    "grillmist": grillmist,
    "centrifuge": centrifuge,
    "power_wheel": power_wheel,
    "water_wheel": water_wheel,
    "large_windmill": large_windmill,
    "windmill": windmill,
}


def iter_consumers() -> Iterator[Tuple[str, MachineSpec]]:
    for name, spec in ALL_MACHINES.items():
        if not spec.is_producer:
            yield name, spec


def iter_producers() -> Iterator[Tuple[str, MachineSpec]]:
    for name, spec in ALL_MACHINES.items():
        if spec.is_producer:
            yield name, spec
