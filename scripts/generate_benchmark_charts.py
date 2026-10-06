"""Generate the benchmark bar charts embedded in README.md.

One-off asset-generation script, not part of the runtime pipeline. Re-run
whenever the underlying benchmark numbers change:

    uv run python scripts/generate_benchmark_charts.py

Colors are the first three validated categorical slots from the dataviz
skill's reference palette (pre-validated for colorblind-safe all-pairs
comparison in both light/dark modes, so no separate validation step is
needed for these 2-3-series charts).
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

SURFACE = "#fcfcfb"
PRIMARY_INK = "#0b0b0b"
SECONDARY_INK = "#52514e"
MUTED_INK = "#898781"
GRIDLINE = "#e1e0d9"
BASELINE = "#c3c2b7"

CATEGORICAL = ["#2a78d6", "#eb6834", "#1baf7a"]  # blue, orange, aqua

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "docs" / "images"


def grouped_bar_chart(
    metrics: list[str],
    series: dict[str, list[float]],
    title: str,
    output_filename: str,
) -> None:
    """Render one grouped bar chart (metrics on the x-axis, one bar group
    per series) and save it to OUTPUT_DIR / output_filename.

    `series` is {category_name: [value_per_metric]}, values aligned with
    `metrics` by position. Colors are assigned to series in the order
    given, using CATEGORICAL's fixed slot order (never reassigned by
    value/rank).
    """
    n_metrics = len(metrics)
    n_series = len(series)
    bar_width = 0.8 / n_series
    x = np.arange(n_metrics)

    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    for i, (label, values) in enumerate(series.items()):
        offset = (i - (n_series - 1) / 2) * bar_width
        bars = ax.bar(
            x + offset,
            values,
            width=bar_width * 0.88,
            color=CATEGORICAL[i % len(CATEGORICAL)],
            label=label,
        )
        for bar, value in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.015,
                f"{value:.2f}",
                ha="center",
                va="bottom",
                fontsize=9,
                color=SECONDARY_INK,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(metrics, color=PRIMARY_INK, fontsize=11)
    ax.set_ylim(0, 1.0)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.tick_params(axis="y", colors=MUTED_INK, labelsize=9)
    ax.tick_params(axis="x", length=0)

    ax.yaxis.grid(True, color=GRIDLINE, linewidth=1, zorder=0)
    ax.set_axisbelow(True)

    for spine_name in ("top", "right", "left"):
        ax.spines[spine_name].set_visible(False)
    ax.spines["bottom"].set_color(BASELINE)

    ax.set_title(title, color=PRIMARY_INK, fontsize=13, pad=16, loc="left")
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.08),
        ncol=n_series,
        frameon=False,
        labelcolor=SECONDARY_INK,
        fontsize=10,
    )

    fig.tight_layout()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_DIR / output_filename, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    grouped_bar_chart(
        metrics=["Recall@10", "MRR", "nDCG@10"],
        series={
            "Naive dense": [0.536, 0.381, 0.413],
            "Hybrid": [0.607, 0.534, 0.532],
            "+ Cross-encoder rerank": [0.607, 0.607, 0.589],
        },
        title="Retrieval ladder",
        output_filename="retrieval_ladder.png",
    )

    grouped_bar_chart(
        metrics=["Recall@10", "MRR", "nDCG@10"],
        series={
            "restricted (sees everything)": [0.536, 0.381, 0.413],
            "internal (RBAC-filtered)": [0.429, 0.310, 0.340],
        },
        title="RBAC pre-filter results",
        output_filename="rbac_results.png",
    )

    grouped_bar_chart(
        metrics=["Faithfulness", "Answer Relevance", "Answer Correctness"],
        series={
            "Claude Haiku 4.5": [0.95, 0.85, 0.50],
            "gpt-oss-120b": [0.91, 0.91, 0.59],
        },
        title="Generation-model comparison",
        output_filename="generation_comparison.png",
    )

    print(f"Charts written to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
