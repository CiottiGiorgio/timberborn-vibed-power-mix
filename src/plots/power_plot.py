import numpy as np
import matplotlib.ticker as ticker


def plot_power(
    ax,
    time_days,
    power_production,
    power_consumption,
    season_boundaries,
    days,
    water_wheels,
    large_windmills,
    windmills,
):
    # Simplified label as details are now in the text box
    production_label = "Total Production"

    ax.plot(
        time_days,
        power_production,
        label=production_label,
        color="#1f77b4",
        linewidth=1,
        alpha=0.8,
    )
    consumption_label = "Total Consumption"
    ax.plot(
        time_days,
        power_consumption,
        label=consumption_label,
        color="#ff7f0e",
        linewidth=2,
    )

    # Calculate means
    mean_production = np.mean(power_production)
    mean_consumption = np.mean(power_consumption)

    # Add horizontal lines for means
    ax.axhline(
        y=mean_production,
        color="#1f77b4",
        linestyle="--",
        linewidth=1.5,
        alpha=0.8,
        label=f"Avg Prod ({mean_production:.1f} hp)",
    )
    ax.axhline(
        y=mean_consumption,
        color="#ff7f0e",
        linestyle="--",
        linewidth=1.5,
        alpha=0.8,
        label=f"Avg Cons ({mean_consumption:.1f} hp)",
    )

    ax.set_ylabel("Power (hp)")
    ax.set_title("Power Profile", pad=20)
    ax.legend(loc="upper right")
    ax.grid(True, linestyle="--", alpha=0.7)

    # Format y-axis with thousands separator
    ax.yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:,.0f}"))
