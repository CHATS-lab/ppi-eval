#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ppi_mt_eval.config import load_config, select_specs
from ppi_mt_eval.data import read_scores, wide_tables
from ppi_mt_eval.intervals import auto_z_reject, human_z_reject, ppi_z_reject
from ppi_mt_eval.plotting import METHOD_COLORS, ensure_dir, savefig


def parse_sample_sizes(values: list[str]) -> list[tuple[int, int]]:
    out = []
    for value in values:
        l_size, u_size = value.split(":", 1)
        out.append((int(l_size), int(u_size)))
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Basic empirical power analysis.")
    p.add_argument("--config", type=Path, default=Path("configs/paper_datasets.json"))
    p.add_argument("--dataset-dir", type=Path, default=Path("datasets"))
    p.add_argument("--output-dir", type=Path, default=Path("results/basic_power"))
    p.add_argument("--datasets", nargs="*", default=None)
    p.add_argument("--sample-sizes", nargs="+", default=["40:800", "80:800"])
    p.add_argument("--num-trials", type=int, default=1000)
    p.add_argument("--seed", type=int, default=20260513)
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--plot-only", action="store_true")
    return p.parse_args()


def run_dataset(spec, config, sample_sizes, args, rng) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    metric = config.representative_metric_by_testset[spec.test_set]
    df = read_scores(spec.path(args.dataset_dir))
    human_table, metric_tables, systems = wide_tables(df, spec.human_col, [metric])
    metric_table = metric_tables[metric]
    rows = []
    skipped_pairs = 0
    min_required = max(l_size + u_size for l_size, u_size in sample_sizes)
    for system_a, system_b in combinations(systems, 2):
        aligned = pd.DataFrame(
            {
                "human_a": human_table[system_a],
                "human_b": human_table[system_b],
                "metric_a": metric_table[system_a],
                "metric_b": metric_table[system_b],
            }
        ).dropna()
        if aligned.shape[0] < min_required:
            skipped_pairs += 1
            continue
        human_diff = (aligned["human_a"] - aligned["human_b"]).to_numpy(float)
        metric_diff = (aligned["metric_a"] - aligned["metric_b"]).to_numpy(float)
        true_effect = float(np.mean(human_diff))
        ordered_a, ordered_b = system_a, system_b
        if not np.isfinite(true_effect) or np.isclose(true_effect, 0.0):
            skipped_pairs += 1
            continue
        if true_effect < 0:
            human_diff = -human_diff
            metric_diff = -metric_diff
            true_effect = -true_effect
            ordered_a, ordered_b = system_b, system_a
        for l_size, u_size in sample_sizes:
            counts = {m: 0 for m in ["human_only", "auto_only", "ppi"]}
            for _ in range(args.num_trials):
                sampled = rng.choice(human_diff.size, size=l_size + u_size, replace=False)
                labeled = sampled[:l_size]
                unlabeled = sampled[l_size:]
                counts["human_only"] += int(human_z_reject(human_diff[None, labeled], args.alpha)[0])
                counts["auto_only"] += int(auto_z_reject(metric_diff[None, sampled], args.alpha)[0])
                counts["ppi"] += int(
                    ppi_z_reject(
                        human_diff[None, labeled],
                        metric_diff[None, labeled],
                        metric_diff[None, unlabeled],
                        args.alpha,
                    )[0]
                )
            for method, count in counts.items():
                rows.append(
                    {
                        "dataset": spec.label,
                        "test_set": spec.test_set,
                        "language_pair": spec.language_pair,
                        "system_a": ordered_a,
                        "system_b": ordered_b,
                        "true_effect": true_effect,
                        "aligned_segments": int(aligned.shape[0]),
                        "L": l_size,
                        "U": u_size,
                        "method": method,
                        "num_trials": args.num_trials,
                        "valid_trials": args.num_trials,
                        "num_rejections": count,
                        "power": float(count / args.num_trials),
                    }
                )
    meta = [{"dataset": spec.label, "num_pairs": len({(r["system_a"], r["system_b"]) for r in rows}), "skipped_pairs": skipped_pairs}]
    return rows, meta


def plot_power(pairwise: pd.DataFrame, output_dir: Path) -> None:
    labels = {"human_only": "Human-only", "auto_only": "Auto-only", "ppi": "PPI"}
    colors = {"human_only": METHOD_COLORS["human_only"], "auto_only": METHOD_COLORS["auto_only"], "ppi": METHOD_COLORS["ppi"]}
    for dataset in pairwise["dataset"].drop_duplicates():
        dataset_df = pairwise[pairwise["dataset"] == dataset]
        labeled_sizes = sorted(dataset_df["L"].unique())
        fig, axes = plt.subplots(1, len(labeled_sizes), figsize=(10.5, 3.6), sharey=True)
        axes = np.asarray([axes]).ravel() if len(labeled_sizes) == 1 else np.asarray(axes).ravel()
        for ax, l_size in zip(axes, labeled_sizes, strict=False):
            subset_l = dataset_df[dataset_df["L"] == l_size]
            for method in labels:
                g = subset_l[subset_l["method"] == method].sort_values("true_effect")
                ax.plot(
                    g["true_effect"],
                    g["power"],
                    marker="o",
                    markersize=2.2,
                    linewidth=1.2,
                    label=labels[method],
                    color=colors[method],
                    alpha=0.9,
                )
            ax.set_title(f"L={l_size}, U={int(subset_l['U'].iloc[0])}")
            ax.set_xlabel("True human effect")
            ax.grid(alpha=0.25)
        axes[0].set_ylabel("Empirical power")
        axes[-1].legend(loc="lower right", fontsize=8)
        fig.suptitle(f"{dataset}: power by true effect size", fontsize=12)
        fig.tight_layout(rect=(0, 0, 1, 0.92))
        tag = dataset.replace(" ", "_").replace("-", "_")
        savefig(fig, output_dir / f"power_curves_{tag}.pdf")

    for l_size, group_l in pairwise.groupby("L"):
        datasets = list(group_l["dataset"].drop_duplicates())
        if len(datasets) > 3:
            fig, axes = plt.subplots(2, 3, figsize=(12, 6.8), sharey=True)
        else:
            fig, axes = plt.subplots(1, len(datasets), figsize=(4.1 * len(datasets), 3.6), sharey=True)
        axes = np.asarray(axes).ravel()
        for ax, dataset in zip(axes, datasets, strict=False):
            sub = group_l[group_l["dataset"] == dataset]
            for method, g in sub.groupby("method"):
                g = g.sort_values("true_effect")
                ax.plot(
                    g["true_effect"],
                    g["power"],
                    marker="o",
                    markersize=2.0,
                    linewidth=1.1,
                    label=labels[method],
                    color=colors[method],
                    alpha=0.9,
                )
            ax.set_title(dataset)
            ax.set_xlabel("True human effect")
            ax.set_ylabel("Empirical power")
            ax.set_ylim(-0.03, 1.03)
            ax.grid(alpha=0.25)
        for ax in axes[len(datasets) :]:
            ax.axis("off")
        axes[-1].legend(loc="lower right", fontsize=8)
        fig.suptitle(f"Power curves for L={l_size}, U=800", fontsize=13)
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        savefig(fig, output_dir / f"power_curves_L{l_size}.pdf")


def main() -> None:
    args = parse_args()
    ensure_dir(args.output_dir)
    if args.plot_only:
        pairwise = pd.read_csv(args.output_dir / "power_pairwise.csv")
    else:
        config = load_config(args.config)
        rng = np.random.default_rng(args.seed)
        rows: list[dict[str, Any]] = []
        meta: list[dict[str, Any]] = []
        for spec in select_specs(config, args.datasets):
            r, m = run_dataset(spec, config, parse_sample_sizes(args.sample_sizes), args, rng)
            rows.extend(r)
            meta.extend(m)
            print(f"completed {spec.label}", flush=True)
        pairwise = pd.DataFrame(rows)
        pairwise.to_csv(args.output_dir / "power_pairwise.csv", index=False)
        pairwise.groupby(["dataset", "L", "method"], as_index=False)["power"].mean().to_csv(
            args.output_dir / "power_summary.csv", index=False
        )
        pd.DataFrame(meta).to_csv(args.output_dir / "dataset_meta.csv", index=False)
        (args.output_dir / "config.json").write_text(json.dumps(vars(args), default=str, indent=2), encoding="utf-8")
    plot_power(pairwise, args.output_dir)


if __name__ == "__main__":
    main()
