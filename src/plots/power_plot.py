import numpy as np

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
        label="Mean Production",
    )
    ax.axhline(
        y=mean_consumption,
        color="#ff7f0e",
        linestyle="--",
        linewidth=1.5,
        alpha=0.8,
        label="Mean Consumption",
    )

    ax.set_ylabel("Power (hp)")
    ax.set_title(f"Power Profile ({days} Days)", pad=35)
    ax.legend(loc="upper right")
    ax.grid(True, linestyle="--", alpha=0.7)

    # Add season labels at the top of the first plot
    for i, (start_day, label) in enumerate(season_boundaries):
        end_day = (
            season_boundaries[i + 1][0] if i + 1 < len(season_boundaries) else days
        )
        mid_point = (start_day + end_day) / 2
        if mid_point < days:
            ax.text(
                mid_point,
                1.01,
                label,
                transform=ax.get_xaxis_transform(),
                ha="center",
                va="bottom",
                fontsize=10,
                fontweight="bold",
                alpha=0.6,
            )
