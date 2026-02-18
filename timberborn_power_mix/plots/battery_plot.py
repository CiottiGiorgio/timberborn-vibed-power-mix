import matplotlib.ticker as ticker
from matplotlib.axes import Axes
import numpy as np
from typing import List


def plot_battery(
    ax: Axes,
    time_days: np.ndarray,
    battery_charge: np.ndarray,
    total_battery_capacity: float,
    num_batteries: int,
    battery_heights: List[float],
) -> None:
    ax.plot(
        time_days,
        battery_charge,
        label="Battery Charge",
        color="#9467bd",
        linewidth=2,
    )
    ax.axhline(
        total_battery_capacity, color="black", linestyle="--", label="Max Capacity"
    )
    ax.fill_between(time_days, battery_charge, 0, color="#9467bd", alpha=0.3)

    ax.set_ylabel("Stored Energy (hph)")
    ax.set_xlabel("Time (days)")

    if num_batteries > 0:
        avg_height = sum(battery_heights) / num_batteries
        height_str = f"Avg {avg_height:.1f}"
    else:
        height_str = "0"

    ax.set_title(f"Battery Status ({num_batteries} Batteries @ {height_str}m)", pad=20)
    ax.legend(loc="upper right")
    ax.grid(True, linestyle="--", alpha=0.7)

    # Format y-axis with thousands separator
    ax.yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:,.0f}"))
