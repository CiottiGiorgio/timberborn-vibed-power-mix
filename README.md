# Timberborn Vibed Power Mix

A simulation and optimization tool for power management in Timberborn. This tool helps you determine the optimal mix of power sources and batteries to sustain your factories through wet, dry, and badtide seasons.

## Installation

Ensure you have Python 3.13+ installed. This project uses [Poetry](https://python-poetry.org/) for dependency management.

```bash
poetry install
```

## Usage

The tool provides a CLI with two main commands: `run` and `optimize`.

### Run Simulation

Simulate a specific configuration of power producers, consumers, and batteries. This will generate a plot showing power generation, consumption, and battery levels over time.

```bash
poetry run python timberborn_power_mix/main.py run \
    --large-windmill 5 \
    --windmill 2 \
    --battery 3 \
    --battery-height 10 \
    --lumber-mill 2 \
    --gear-workshop 1 \
    --samples-per-sim 100
```

**Key Parameters:**
- `--[machine-name] [count]`: Specify the number of producers (e.g., `--large-windmill`, `--water-wheel`) or consumers (e.g., `--lumber-mill`, `--steel-factory`).
- `--battery [count]`: Number of gravity batteries.
- `--battery-height [meters]`: Height of the gravity batteries. Can be a single integer or a comma-separated list (e.g., `10,15,10`).
- `--samples-per-sim [count]`: Number of Monte Carlo simulations to run (default: 1000).
- `--days [count]`: Duration of the simulation in days.

### Optimize

Find the most cost-effective power mix that meets your factory's demands with minimal downtime.

```bash
poetry run python timberborn_power_mix/main.py optimize \
    --lumber-mill 5 \
    --gear-workshop 2 \
    --iterations 500
```

**Key Parameters:**
- `--iterations [count]`: Number of optimization iterations to run.
- `--[consumer-name] [count]`: Specify the factories you need to power.

## Data Insights

The simulation accounts for:
- **Intermittent Production**: Power generation fluctuates due to wind strength, seasonal water flow changes, and working hours.
- **Seasons**:
  - **Wet Season**: Water wheels function normally.
  - **Dry Season**: Water wheels stop working.
  - **Badtide**: Water wheels continue to function (assuming contaminated water still flows).
- **Working Hours**: Factories only consume power during specified working hours (default: 16h/day).
- **Battery Physics**: Gravity battery capacity scales with height.

The optimizer looks for solutions where the factories are unpowered for less than 5% of the time in the 95th percentile worst-case scenario.

### Visualizing Performance
The generated plots help you understand the dynamics of your power grid:
- **Sufficiency & Timing**: Verify if you produce enough power and, crucially, if you produce it *at the right time* relative to your working hours.
- **Battery Reaction**: Observe how your battery system reacts to the generation of power over time, smoothing out intermittent production.
- **Surplus & Deficits**: Check if your setup correctly captures energy surplus during high-production windows and if it has enough depth to cover deficits.
