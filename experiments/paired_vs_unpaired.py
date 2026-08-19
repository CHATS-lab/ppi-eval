#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ppi_mt_eval.config import load_config, select_specs
from ppi_mt_eval.data import filtered_metric_columns, is_reference_system, read_scores
from ppi_mt_eval.plotting import ensure_dir, savefig
from ppi_mt_eval.variance import corr, human_paired_unpaired_ratio, ppi_paired_unpaired_ratio


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Paired-vs-unpaired variance comparisons.")
    p.add_argument("--config", type=Path, default=Path("configs/paper_datasets.json"))
    p.add_argument("--dataset-dir", type=Path, default=Path("datasets"))
    p.add_argument("--output-dir", type=Path, default=Path("results/paired_vs_unpaired"))
    p.add_argument("--datasets", nargs="*", default=None)
    return p.parse_args()


def run_dataset(spec, config, dataset_dir: Path):
    df = read_scores(spec.path(dataset_dir))
    df = df.loc[~df["system-name"].map(is_reference_system)].copy()
    metric_cols = filtered_metric_columns(df, spec.human_col, spec, config)
    human = df.pivot_table(index="seg-id", columns="system-name", values=spec.human_col, aggfunc="mean")
    systems = [system for system in human.columns if human[system].notna().any()]
    systems = list(human[systems].mean(axis=0).sort_values(ascending=False).index)
    human = human[systems]
    metric_tables = {
        metric: df.pivot_table(index="seg-id", columns="system-name", values=metric, aggfunc="mean").reindex(columns=systems)
        for metric in metric_cols
    }
    human_rows = []
    pp_rows = []
    corr_rows = []
    human_global = human.dropna(axis=0, how="any")
    human_systems = list(human_global.mean(axis=0).sort_values(ascending=False).index)
    for system_a, system_b in combinations(human_systems, 2):
        y1 = human_global[system_a].to_numpy(float)
        y2 = human_global[system_b].to_numpy(float)
        human_rows.append(
            {
                "dataset": spec.label,
                "test_set": spec.test_set,
                "language_pair": spec.language_pair,
                "human_col": spec.human_col,
                "system_a": system_a,
                "system_b": system_b,
                "aligned_segments": int(human_global.shape[0]),
                "relative_variance_increase": human_paired_unpaired_ratio(y1, y2),
            }
        )
    for ia, ib in combinations(range(len(systems)), 2):
        system_a, system_b = systems[ia], systems[ib]
        for metric in metric_cols:
            table = metric_tables[metric]
            aligned = pd.DataFrame(
                {
                    "y_a": human[system_a],
                    "y_b": human[system_b],
                    "f_a": table[system_a],
                    "f_b": table[system_b],
                }
            ).dropna()
            if aligned.shape[0] < 3:
                continue
            y1m = aligned["y_a"].to_numpy(float)
            y2m = aligned["y_b"].to_numpy(float)
            f1 = aligned["f_a"].to_numpy(float)
            f2 = aligned["f_b"].to_numpy(float)
            pp_rows.append(
                {
                    "dataset": spec.label,
                    "test_set": spec.test_set,
                    "language_pair": spec.language_pair,
                    "human_col": spec.human_col,
                    "system_a": system_a,
                    "system_b": system_b,
                    "metric": metric,
                    "aligned_segments": int(aligned.shape[0]),
                    "relative_variance_increase": ppi_paired_unpaired_ratio(y1m, y2m, f1, f2),
                }
            )
            c_delta = corr(y1m - y2m, f1 - f2)
            c_sep = np.nanmean([corr(y1m, f1), corr(y2m, f2)])
            corr_rows.append(
                {
                    "dataset": spec.label,
                    "metric": metric,
                    "corr_delta_abs": abs(c_delta) if np.isfinite(c_delta) else np.nan,
                    "corr_sep_abs": abs(c_sep) if np.isfinite(c_sep) else np.nan,
                    "delta_smaller": abs(c_delta) < abs(c_sep) if np.isfinite(c_delta) and np.isfinite(c_sep) else np.nan,
                }
            )
    return human_rows, pp_rows, corr_rows


def summarize(df: pd.DataFrame, value_col: str = "relative_variance_increase") -> pd.DataFrame:
    rows = []
    for dataset, group in df.groupby("dataset"):
        vals = group[value_col].replace([np.inf, -np.inf], np.nan).dropna()
        rows.append(
            {
                "dataset": dataset,
                "num_cases": int(vals.shape[0]),
                "mean": float(vals.mean()),
                "positive_percent": float((vals > 0).mean() * 100.0),
            }
        )
    return pd.DataFrame(rows)


def plot_hist(df: pd.DataFrame, output: Path, title: str, color: str, bins: int, ylabel: str) -> None:
    datasets = list(df["dataset"].drop_duplicates())
    fig, axes = plt.subplots(2, 3, figsize=(12, 6.8), sharex=False, sharey=False)
    axes = axes.ravel()
    for ax, dataset in zip(axes, datasets, strict=False):
        vals = df[df["dataset"] == dataset]["relative_variance_increase"].replace([np.inf, -np.inf], np.nan).dropna()
        ax.hist(vals, bins=bins, color=color, edgecolor="white", alpha=0.9)
        ax.axvline(0, color="#B22222", linestyle="--", linewidth=1.4)
        percent_positive = 100 * (vals > 0).mean()
        ax.set_title(f"{dataset}\n{percent_positive:.1f}% positive", fontsize=10)
        ax.set_xlabel("Relative variance increase")
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=0.25)
    for ax in axes[len(datasets) :]:
        ax.axis("off")
    fig.suptitle(title)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    savefig(fig, output)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    ensure_dir(args.output_dir)
    h_rows, p_rows, c_rows = [], [], []
    for spec in select_specs(config, args.datasets):
        h, p, c = run_dataset(spec, config, args.dataset_dir)
        h_rows.extend(h)
        p_rows.extend(p)
        c_rows.extend(c)
        print(f"completed {spec.label}", flush=True)
    human = pd.DataFrame(h_rows)
    pp = pd.DataFrame(p_rows)
    corr_df = pd.DataFrame(c_rows)
    human.to_csv(args.output_dir / "human_pairwise_variance.csv", index=False)
    pp.to_csv(args.output_dir / "ppi_pairwise_metric_variance.csv", index=False)
    summarize(human).to_csv(args.output_dir / "human_summary.csv", index=False)
    summarize(pp).to_csv(args.output_dir / "ppi_summary.csv", index=False)
    corr_summary = corr_df.groupby("dataset", as_index=False).agg(
        mean_corr_delta_abs=("corr_delta_abs", "mean"),
        mean_corr_sep_abs=("corr_sep_abs", "mean"),
        delta_smaller_percent=("delta_smaller", lambda x: float(np.nanmean(x) * 100.0)),
    )
    corr_summary.to_csv(args.output_dir / "ppi_corr_summary.csv", index=False)
    (args.output_dir / "config.json").write_text(json.dumps(vars(args), default=str, indent=2), encoding="utf-8")
    plot_hist(
        human,
        args.output_dir / "paired_vs_unpaired_human_variance_histograms.pdf",
        "Distribution of relative variance increase from using an unpaired design",
        color="#4C78A8",
        bins=18,
        ylabel="System pairs",
    )
    plot_hist(
        pp,
        args.output_dir / "paired_vs_unpaired_pp_variance_histograms.pdf",
        "Distribution of relative PPI variance increase from using an unpaired design",
        color="#F58518",
        bins=24,
        ylabel="System pair / metric cases",
    )
    print(summarize(human).to_string(index=False))
    print(summarize(pp).to_string(index=False))


if __name__ == "__main__":
    main()
