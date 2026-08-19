#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ppi_mt_eval.intervals import ppi_perm_reject
from ppi_mt_eval.plotting import ensure_dir, savefig
from ppi_mt_eval.progress import iter_progress, progress_bar


COLORS = {
    "ppi_perm": "#E45756",
    "ppi_perm_uncentered": "#E45756",
}
LABELS = {
    "ppi_perm": "PPI Perm.",
    "ppi_perm_uncentered": "PPI Perm. (uncentered)",
}
LINESTYLES = {
    "ppi_perm": "-",
    "ppi_perm_uncentered": "--",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Type I error simulation for centered vs uncentered PPI permutation tests."
    )
    p.add_argument("--output-dir", type=Path, default=Path("results/type_i_error_metric_mean_shift_simulation"))
    p.add_argument("--rhos", nargs="+", type=float, default=[0.3, 0.7])
    p.add_argument("--metric-means", nargs="+", type=float, default=[0.0, 1.0, 5.0])
    p.add_argument("--labeled-sizes", nargs="+", type=int, default=list(range(20, 201, 20)))
    p.add_argument("--unlabeled-size", type=int, default=800)
    p.add_argument("--num-trials", type=int, default=10000)
    p.add_argument("--num-permutations", type=int, default=1000)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--seed", type=int, default=20260704)
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--plot-only", action="store_true")
    p.add_argument("--no-progress", action="store_true")
    return p.parse_args()


def sample_null(
    rng: np.random.Generator,
    rows: int,
    size: int,
    rho: float,
    metric_mean: float,
) -> tuple[np.ndarray, np.ndarray]:
    sample = rng.multivariate_normal(
        mean=np.array([0.0, metric_mean], dtype=float),
        cov=np.array([[1.0, rho], [rho, 1.0]], dtype=float),
        size=(rows, size),
    )
    return sample[:, :, 0], sample[:, :, 1]


def run_condition(args, seed: int, rho: float, metric_mean: float, l_size: int) -> list[dict[str, object]]:
    rng = np.random.default_rng(seed)
    u_size = args.unlabeled_size
    methods = ("ppi_perm", "ppi_perm_uncentered")
    counts = {method: 0 for method in methods}
    done = 0
    with progress_bar(
        enabled=not args.no_progress,
        total=args.num_trials,
        desc=f"rho={rho} mean_f={metric_mean:g} L={l_size}",
        unit="trial",
        leave=False,
    ) as bar:
        while done < args.num_trials:
            batch = min(args.batch_size, args.num_trials - done)
            d, f = sample_null(rng, batch, l_size + u_size, rho, metric_mean)
            y_l = d[:, :l_size]
            f_l = f[:, :l_size]
            f_u = f[:, l_size:]
            signs_l = rng.choice(np.array([-1, 1], dtype=np.int8), size=(batch, args.num_permutations, l_size))
            signs_u = rng.choice(np.array([-1, 1], dtype=np.int8), size=(batch, args.num_permutations, u_size))
            counts["ppi_perm"] += int(
                ppi_perm_reject(y_l, f_l, f_u, signs_l, signs_u, args.alpha, center_metric=True).sum()
            )
            counts["ppi_perm_uncentered"] += int(
                ppi_perm_reject(y_l, f_l, f_u, signs_l, signs_u, args.alpha, center_metric=False).sum()
            )
            done += batch
            bar.update(batch)
    return [
        {
            "rho": rho,
            "metric_mean": metric_mean,
            "labeled_size": l_size,
            "unlabeled_size": u_size,
            "method": method,
            "method_label": LABELS[method],
            "rejection_count": count,
            "num_trials": args.num_trials,
            "num_permutations": args.num_permutations,
            "type_i_error": count / args.num_trials,
        }
        for method, count in counts.items()
    ]


def normalize_method_names(summary: pd.DataFrame) -> pd.DataFrame:
    summary = summary.copy()
    old_centered = summary["method"].eq("ppi_perm_centered")
    if "method_label" in summary:
        labels = summary["method_label"].astype(str)
    else:
        labels = pd.Series("", index=summary.index)
    old_uncentered = summary["method"].eq("ppi_perm") & labels.str.contains("uncentered", case=False, na=False)
    summary.loc[old_centered, "method"] = "ppi_perm"
    summary.loc[old_uncentered, "method"] = "ppi_perm_uncentered"
    summary["method_label"] = summary["method"].map(LABELS)
    return summary


def plot(summary: pd.DataFrame, args) -> None:
    summary = normalize_method_names(summary)
    figures_dir = args.output_dir / "figures"
    ensure_dir(figures_dir)
    fig, axes = plt.subplots(len(args.rhos), len(args.metric_means), figsize=(12.2, 6.8), sharex=True, sharey=True)
    axes = np.asarray(axes).reshape(len(args.rhos), len(args.metric_means))
    for row_idx, rho in enumerate(args.rhos):
        for col_idx, metric_mean in enumerate(args.metric_means):
            ax = axes[row_idx, col_idx]
            subset = summary[
                np.isclose(summary["rho"].astype(float), rho)
                & np.isclose(summary["metric_mean"].astype(float), metric_mean)
            ]
            for method in ("ppi_perm", "ppi_perm_uncentered"):
                line = subset[subset["method"] == method].sort_values("labeled_size")
                ax.plot(
                    line["labeled_size"],
                    line["type_i_error"],
                    marker="o",
                    linewidth=1.8,
                    markersize=3.5,
                    color=COLORS[method],
                    linestyle=LINESTYLES[method],
                    label=LABELS[method],
                )
            ax.axhline(args.alpha, color="#555555", linestyle="--", linewidth=1.0, label=r"$\alpha=0.05$")
            ax.set_title(rf"$\rho={rho}$, $\mathbb{{E}}[f]={metric_mean:g}$", fontsize=10)
            ax.grid(alpha=0.25)
            if row_idx == len(args.rhos) - 1:
                ax.set_xlabel("Labeled examples (L)")
            if col_idx == 0:
                ax.set_ylabel("Empirical Type I error")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    savefig(fig, figures_dir / "type_i_error_metric_mean_ppi_perm_centering_comparison.pdf")


def main() -> None:
    args = parse_args()
    ensure_dir(args.output_dir)
    if args.plot_only:
        summary = normalize_method_names(pd.read_csv(args.output_dir / "ppi_perm_centering_comparison_summary.csv"))
    else:
        seed_sequence = np.random.SeedSequence(args.seed)
        conditions = [
            (rho, metric_mean, l_size)
            for rho in args.rhos
            for metric_mean in args.metric_means
            for l_size in args.labeled_sizes
        ]
        child_seeds = seed_sequence.spawn(len(conditions))
        rows = []
        for (rho, metric_mean, l_size), child_seed in iter_progress(
            list(zip(conditions, child_seeds, strict=True)),
            enabled=not args.no_progress,
            desc="conditions",
            unit="condition",
        ):
            rows.extend(run_condition(args, int(child_seed.generate_state(1)[0]), rho, metric_mean, l_size))
            print(f"completed rho={rho} mean_f={metric_mean:g} L={l_size}", flush=True)
        summary = normalize_method_names(pd.DataFrame(rows))
        summary.to_csv(args.output_dir / "ppi_perm_centering_comparison_summary.csv", index=False)
        summary.to_csv(args.output_dir / "type_i_error_summary.csv", index=False)
        (args.output_dir / "config.json").write_text(json.dumps(vars(args), default=str, indent=2), encoding="utf-8")
    plot(summary, args)
    print(summary.groupby("method")["type_i_error"].agg(["mean", "max"]).round(4).to_string())


if __name__ == "__main__":
    main()
