from enum import StrEnum

HOURS_PER_DAY = 24

# Simulation Defaults
DEFAULT_SAMPLES = 1000
DEFAULT_DAYS = 132
DEFAULT_WORKING_HOURS = 16

# Season Defaults
DEFAULT_WET_SEASON_DAYS = 3
DEFAULT_DRY_SEASON_DAYS = 30
DEFAULT_BADTIDE_SEASON_DAYS = 30

# Wind Simulation
WIND_DURATION_MIN_HOURS = 5
WIND_DURATION_MAX_HOURS = 13  # Exclusive for randint (5-12 hours)
LARGE_WINDMILL_THRESHOLD = 0.20
WINDMILL_THRESHOLD = 0.30

# Optimization Defaults
DEFAULT_OPTIMIZATION_ITERATIONS = 150


class ConfigKey(StrEnum):
    SAMPLES = "samples"
    DAYS = "days"
    WORKING_HOURS = "working_hours"
    WET_DAYS = "wet_days"
    DRY_DAYS = "dry_days"
    BADTIDE_DAYS = "badtide_days"
    ITERATIONS = "iterations"
    FACTORIES = "factories"
    ENERGY_MIX = "energy_mix"
