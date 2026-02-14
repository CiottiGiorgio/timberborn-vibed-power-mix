from enum import StrEnum


class ConfigName(StrEnum):
    SAMPLES = "samples"
    DAYS = "days"
    WORKING_HOURS = "working_hours"
    WET_DAYS = "wet_days"
    DRY_DAYS = "dry_days"
    BADTIDE_DAYS = "badtide_days"
    ITERATIONS = "iterations"
    FACTORIES = "factories"
    ENERGY_MIX = "energy_mix"
    SEED = "seed"
