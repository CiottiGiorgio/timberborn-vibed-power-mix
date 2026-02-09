import consts


def plot_energy(ax, time_days, energy_production, energy_consumption, days):
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

    # Calculate average power (slope)
    # Total energy / total hours
    total_hours = days * consts.HOURS_PER_DAY

    # Ensure we use the last value of the cumulative array
    total_produced = energy_production[-1]
    total_consumed = energy_consumption[-1]

    avg_prod_power = total_produced / total_hours
    avg_cons_power = total_consumed / total_hours

    # Create lines: y = avg_power * time_hours = avg_power * (time_days * 24)
    # We can just use linear interpolation between (0,0) and (days, total)

    ax.plot(
        [0, days],
        [0, total_produced],
        color="#1f77b4",
        linestyle="--",
        linewidth=1.5,
        alpha=0.7,
        label=f"Avg Prod Slope ({avg_prod_power:.1f} hp)",
    )

    ax.plot(
        [0, days],
        [0, total_consumed],
        color="#ff7f0e",
        linestyle="--",
        linewidth=1.5,
        alpha=0.7,
        label=f"Avg Cons Slope ({avg_cons_power:.1f} hp)",
    )

    ax.set_ylabel("Energy (hph)")
    ax.set_title(f"Energy Profile ({days} Days)")
    ax.legend(loc="upper left")
    ax.grid(True, linestyle="--", alpha=0.7)
