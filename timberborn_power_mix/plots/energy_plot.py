import numpy as np
from matplotlib import ticker
from matplotlib.axes import Axes

from timberborn_power_mix.simulation import consts as sim_consts


def plot_energy(
    ax: Axes,
    time_days: np.ndarray,
    energy_production: np.ndarray,
    energy_consumption: np.ndarray,
    days: int,
) -> None:
    ax.plot(
        time_days,
        energy_production,
        label="Cumulative Energy Produced",
        color="#1f77b4",
        linewidth=2,
    )
    ax.plot(
        time_days,
        energy_consumption,
        label="Cumulative Energy Consumed",
        color="#ff7f0e",
        linewidth=2,
    )

    # Calculate average power (slope) using linear regression (Least Squares)
    # Fit y = mx + c
    # m (slope) will be in Energy/Day.
    # To get Power (Energy/Hour), we divide m by 24.

    slope_prod, intercept_prod = np.polyfit(time_days, energy_production, 1)
    avg_prod_power = slope_prod / sim_consts.HOURS_PER_DAY

    slope_cons, intercept_cons = np.polyfit(time_days, energy_consumption, 1)
    avg_cons_power = slope_cons / sim_consts.HOURS_PER_DAY

    # Create lines for plotting covering the full range
    x_vals = np.array([0, days])
    y_vals_prod = slope_prod * x_vals + intercept_prod
    y_vals_cons = slope_cons * x_vals + intercept_cons

    ax.plot(
        x_vals,
        y_vals_prod,
        color="#1f77b4",
        linestyle="--",
        linewidth=1.5,
        alpha=0.7,
        label=f"OLS Fit Prod ({avg_prod_power:.1f} hp)",
    )

    ax.plot(
        x_vals,
        y_vals_cons,
        color="#ff7f0e",
        linestyle="--",
        linewidth=1.5,
        alpha=0.7,
        label=f"OLS Fit Cons ({avg_cons_power:.1f} hp)",
    )

    ax.set_ylabel("Energy (hph)")
    ax.set_title("Energy Profile", pad=20)
    ax.legend(loc="upper left")
    ax.grid(True, linestyle="--", alpha=0.7)

    # Format y-axis with thousands separator
    ax.yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:,.0f}"))
