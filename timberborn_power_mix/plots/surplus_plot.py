from collections.abc import Sequence
from typing import cast

import numpy as np
from matplotlib import ticker
from matplotlib.axes import Axes


def plot_surplus(
    ax: Axes,
    time_days: np.ndarray,
    power_surplus: np.ndarray,
    effective_balance: np.ndarray,
) -> None:
    ax.plot(
        time_days,
        power_surplus,
        color="#D3D3D3",
        linewidth=1,
        alpha=0.4,
        label="Original Surplus/Deficit",
    )
    ax.fill_between(time_days, power_surplus, 0, color="#D3D3D3", alpha=0.15)
    ax.fill_between(
        time_days,
        effective_balance,
        0,
        where=cast(Sequence[bool], effective_balance > 0),
        facecolor="green",
        alpha=0.6,
        label="Unstored Surplus",
    )
    ax.fill_between(
        time_days,
        effective_balance,
        0,
        where=cast(Sequence[bool], effective_balance < 0),
        facecolor="red",
        alpha=0.6,
        label="Uncovered Deficit",
    )

    ax.axhline(0, color="black", linewidth=1, linestyle="-")
    ax.set_ylabel("Effective Surplus (hp)")
    ax.set_title("Power Surplus Profile (After Battery Buffering)", pad=20)
    ax.legend(loc="upper right")
    ax.grid(True, linestyle="--", alpha=0.7)

    # Format y-axis with thousands separator
    ax.yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:,.0f}"))
