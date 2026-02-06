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

    ax.set_ylabel("Energy (hph)")
    ax.set_title(f"Energy Profile ({days} Days)")
    ax.legend(loc="upper right")
    ax.grid(True, linestyle="--", alpha=0.7)
