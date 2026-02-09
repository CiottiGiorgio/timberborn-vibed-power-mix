from dataclasses import dataclass
from typing import Iterator, Tuple


@dataclass(frozen=True)
class MachineSpec:
    power: int
    cost: int
    is_producer: bool = False


@dataclass(frozen=True)
class BatterySpec:
    base_capacity: int
    capacity_per_height: int
    base_cost: int
    cost_per_height: int

    def calculate_capacity(self, height: int) -> int:
        return self.base_capacity + (height * self.capacity_per_height)

    def calculate_cost(self, height: int) -> int:
        return self.base_cost + (height * self.cost_per_height)


class MachineDatabase:
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

    power_wheel = MachineSpec(power=50, cost=50, is_producer=True)
    water_wheel = MachineSpec(power=150, cost=50, is_producer=True)
    large_windmill = MachineSpec(power=300, cost=75, is_producer=True)
    windmill = MachineSpec(power=150, cost=40, is_producer=True)

    battery = BatterySpec(
        base_capacity=4000, capacity_per_height=2000, base_cost=84, cost_per_height=6
    )

    @classmethod
    def iter_machines(cls) -> Iterator[Tuple[str, MachineSpec]]:
        return (
            (name, value)
            for name, value in cls.__dict__.items()
            if isinstance(value, MachineSpec)
        )

    @classmethod
    def iter_consumers(cls) -> Iterator[Tuple[str, MachineSpec]]:
        return (
            (name, value)
            for name, value in cls.iter_machines()
            if not value.is_producer
        )

    @classmethod
    def iter_producers(cls) -> Iterator[Tuple[str, MachineSpec]]:
        return (
            (name, value) for name, value in cls.iter_machines() if value.is_producer
        )
