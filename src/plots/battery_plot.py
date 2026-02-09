def plot_battery(
    ax, time_days, battery_charge, total_battery_capacity, batteries, battery_height
):
    ax.plot(
        time_days,
        battery_charge,
        label=f"Battery Charge (Max: {total_battery_capacity} hph)",
        color="#9467bd",
        linewidth=2,
    )
    ax.axhline(
        total_battery_capacity, color="black", linestyle="--", label="Max Capacity"
    )
    ax.fill_between(time_days, battery_charge, 0, color="#9467bd", alpha=0.3)

    ax.set_ylabel("Stored Energy (hph)")
    ax.set_xlabel("Time (days)")

    if isinstance(battery_height, list):
        avg_height = sum(battery_height) / len(battery_height) if battery_height else 0
        height_str = f"Avg {avg_height:.1f}"
    else:
        height_str = str(battery_height)

    ax.set_title(f"Battery Status ({batteries} Batteries @ {height_str}m)")
    ax.legend(loc="upper right")
    ax.grid(True, linestyle="--", alpha=0.7)
