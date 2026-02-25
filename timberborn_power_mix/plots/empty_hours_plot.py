from typing import Any, cast
import numpy as np
from matplotlib.axes import Axes
import matplotlib.ticker as ticker


def plot_empty_hours_percentage(
    ax: Axes,
    run_lost_hours: np.ndarray,
    total_working_hours: int,
) -> None:
    # Calculate percentages
    # run_lost_hours is now a float array representing "equivalent lost hours"
    # (sum of productivity loss).
    # So (run_lost_hours / total_working_hours) * 100 is the percentage of
    # total potential productivity lost.

    run_lost_percentages = (run_lost_hours / total_working_hours) * 100

    # Create bins of 1% width (default)
    if len(run_lost_percentages) > 0:
        max_val = np.max(run_lost_percentages)
        # Ensure we have at least one bin if max_val is 0
        if max_val == 0:
            bins = np.arange(0, 2, 1, dtype=float)  # 0-1%
        else:
            bins = np.arange(0, max_val + 2, 1, dtype=float)

        # Plot the histogram
        n_raw, bins_out, patches_raw = ax.hist(
            run_lost_percentages,
            bins=cast(Any, bins),
            color="red",
            alpha=0.7,
            edgecolor="black",
        )
        n = cast(Any, n_raw)
        patches = cast(Any, patches_raw)

        # Style the first bar (0-1%) to be grey with diagonal hatching
        if len(patches) > 0:
            patches[0].set_facecolor("lightgrey")
            patches[0].set_hatch("//")
            patches[0].set_edgecolor("black")

        # Scale the y-axis ignoring the first bin (0-1%)
        # The first bin corresponds to n[0]
        if len(n) > 1:
            max_freq_excluding_first = max(n[1:])
            # Add some headroom (e.g., 10%)
            if max_freq_excluding_first > 0:
                ax.set_ylim(0, float(max_freq_excluding_first * 1.1))

        # Calculate percentiles and mean for ALL samples
        if len(run_lost_percentages) > 0:
            p5 = float(np.percentile(run_lost_percentages, 5))
            p50 = float(np.percentile(run_lost_percentages, 50))
            p95 = float(np.percentile(run_lost_percentages, 95))
            mean_val = float(np.mean(run_lost_percentages))

            # Add vertical lines for percentiles
            ax.axvline(
                p5,
                color="blue",
                linestyle="--",
                linewidth=1.5,
                label=f"5th % ({p5:.1f}%)",
            )
            ax.axvline(
                p50,
                color="green",
                linestyle="--",
                linewidth=1.5,
                label=f"Median ({p50:.1f}%)",
            )
            ax.axvline(
                mean_val,
                color="orange",
                linestyle="-",
                linewidth=1.5,
                label=f"Mean ({mean_val:.1f}%)",
            )
            ax.axvline(
                p95,
                color="purple",
                linestyle="--",
                linewidth=1.5,
                label=f"95th % ({p95:.1f}%)",
            )
            ax.legend(loc="upper right")

            # Add explanatory text for 95th percentile
            explanation_text = (
                f"In 95% of all cases, the system loses less\n"
                f"than {p95:.1f}% of total potential productivity."
            )

            # Place text below the legend (using relative coordinates)
            # We use transform=ax.transAxes to position relative to the axes
            ax.text(
                0.98,
                0.60,  # Position slightly below the legend (adjust as needed)
                explanation_text,
                transform=ax.transAxes,
                fontsize=9,
                verticalalignment="top",
                horizontalalignment="right",
                bbox=dict(
                    boxstyle="round", facecolor="white", alpha=0.8, edgecolor="gray"
                ),
            )

    ax.set_title("Distribution of productivity loss")
    ax.set_xlabel("Percentage of productivity lost (%)")
    ax.set_ylabel("Number of Samples")
    ax.grid(True, linestyle="--", alpha=0.5)

    # Format y-axis with thousands separator
    ax.yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:,.0f}"))
    # Ensure y-axis ticks are integers
    ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
