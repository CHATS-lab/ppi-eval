from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


METHOD_COLORS = {
    "human_only": "#4C78A8",
    "auto_only": "#F58518",
    "ppi": "#54A24B",
    "ppi_bleu": "#B279A2",
    "human_z": "#4C78A8",
    "human_perm": "#72B7B2",
    "ppi_z": "#54A24B",
    "ppi_perm": "#E45756",
}


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def savefig(fig: plt.Figure, path: Path) -> None:
    ensure_dir(path.parent)
    fig.savefig(path.with_suffix(".png"), dpi=200)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def plot_ci_summary(summary: pd.DataFrame, output_prefix: Path, title: str | None = None) -> None:
    datasets = list(summary["dataset"].drop_duplicates())
    for value_col, ylabel, suffix, ref in [
        ("mean_interval_width", "Average interval width", "width", None),
        ("mean_coverage_rate", "Empirical coverage", "coverage", 0.95),
    ]:
        if len(datasets) > 3:
            nrows, ncols, figsize = 2, 3, (12, 6.8)
        else:
            nrows, ncols, figsize = 1, max(1, len(datasets)), (4.0 * max(1, len(datasets)), 3.9)
        fig, axes = plt.subplots(nrows, ncols, figsize=figsize, sharex=True)
        axes = np.asarray(axes).ravel()
        for ax, dataset in zip(axes, datasets, strict=False):
            sub = summary[summary["dataset"] == dataset]
            for method, group in sub.groupby("method", sort=False):
                group = group.sort_values("L")
                ax.plot(
                    group["L"],
                    group[value_col],
                    marker="o",
                    linewidth=1.7,
                    markersize=3.2,
                    label=group["method_label"].iloc[0],
                    color=METHOD_COLORS.get(method, None),
                )
            if ref is not None:
                ax.axhline(ref, color="#666666", linestyle="--", linewidth=1.0)
            ax.set_title(dataset, fontsize=10)
            ax.set_xlabel("L")
            ax.set_ylabel(ylabel)
            ax.grid(alpha=0.25)
        for ax in axes[len(datasets) :]:
            ax.axis("off")
        handles, labels = axes[0].get_legend_handles_labels()
        if handles:
            legend_cols = min(4, len(labels)) if len(datasets) > 3 else min(3, len(labels))
            fig.legend(handles, labels, loc="lower center", ncol=legend_cols, frameon=False, fontsize=8)
        if title:
            fig.suptitle(title if suffix == "width" else title.replace("width", "coverage"), fontsize=13)
        fig.tight_layout(rect=(0, 0.08 if handles else 0, 1, 0.94 if title else 1))
        savefig(fig, output_prefix.parent / f"{output_prefix.name}_{suffix}.pdf")
