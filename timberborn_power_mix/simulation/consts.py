HOURS_PER_DAY = 24

# Simulation Defaults
DEFAULT_SIMULATION_SAMPLES = 10_000
DEFAULT_DAYS = 132
DEFAULT_WORKING_HOURS = 16

# Season Defaults
DEFAULT_WET_SEASON_DAYS = 3
DEFAULT_DRY_SEASON_DAYS = 30
DEFAULT_BADTIDE_SEASON_DAYS = 30

# Optimization Defaults
DEFAULT_OPTIMIZATION_ITERATIONS = 20

# Wind Simulation
WIND_DURATION_MIN_HOURS = 5
WIND_DURATION_MAX_HOURS = 13  # Exclusive for randint (5-12 hours)

# Wind strength is simulated using integers (0-1000) for performance.
# Integer math avoids floating-point overhead and ensures consistent
# behavior across different CPU architectures.
WIND_STRENGTH_MAX = 1000
LARGE_WINDMILL_THRESHOLD = 200  # Equivalent to 0.20
WINDMILL_THRESHOLD = 300  # Equivalent to 0.30
