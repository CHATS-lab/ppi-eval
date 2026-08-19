#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ppi_mt_eval.config import DatasetSpec, load_config, select_specs
from ppi_mt_eval.data import read_scores, wide_tables
from ppi_mt_eval.intervals import human_z_intervals, ppi_intervals
from ppi_mt_eval.plotting import ensure_dir, plot_ci_summary


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Unified CI simulations for paper experiments.")
    p.add_argument("--config", type=Path, default=Path("configs/paper_datasets.json"))
    p.add_argument("--dataset-dir", type=Path, default=Path("datasets"))
    p.add_argument("--output-dir", type=Path, default=Path("results/confidence_intervals"))
    p.add_argument("--mode", choices=["basic", "gemba"], default="basic")
    p.add_argument("--datasets", nargs="*", default=None)
    p.add_argument("--labeled-sizes", nargs="+", type=int, default=list(range(20, 201, 20)))
    p.add_argument("--unlabeled-size", type=int, default=800)
    p.add_argument("--num-runs", type=int, default=1000)
    p.add_argument("--seed", type=int, default=20260704)
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--plot-only", action="store_true")
    return p.parse_args()


def build_common_pair_matrices(human_table, metric_tables, systems):
    common_index = human_table.dropna().index
    for table in metric_tables.values():
        common_index = common_index.intersection(table.dropna().index)
    human_array = human_table.loc[common_index, systems].to_numpy(float).T
    metric_arrays = {
        metric: table.loc[common_index, systems].to_numpy(float).T
        for metric, table in metric_tables.items()
    }
    pair_specs = []
    human_pairs = []
    metric_pairs = {metric: [] for metric in metric_tables}
    for idx_a, system_a in enumerate(systems):
        for idx_b in range(idx_a + 1, len(systems)):
            hdiff = human_array[idx_a] - human_array[idx_b]
            pair_specs.append(
                {
                    "system_a": system_a,
                    "system_b": systems[idx_b],
                    "true_effect": float(np.mean(hdiff)),
                    "aligned_segments": int(hdiff.size),
                }
            )
            human_pairs.append(hdiff)
            for metric, metric_array in metric_arrays.items():
                metric_pairs[metric].append(metric_array[idx_a] - metric_array[idx_b])
    return pair_specs, np.vstack(human_pairs), {metric: np.vstack(values) for metric, values in metric_pairs.items()}


def metric_plan(mode: str, spec: DatasetSpec, config) -> list[tuple[str, str, str]]:
    if mode == "gemba":
        metric = config.gemba_metrics.get(spec.key)
        if metric is None:
            return []
        return [("auto_only", metric, "Auto-only"), ("ppi", metric, "PPI")]
    metricx = config.representative_metric_by_testset[spec.test_set]
    return [("ppi", metricx, "PPI + MetricX"), ("ppi_bleu", config.bleu_metric, "PPI + BLEU")]


def run_dataset(
    spec: DatasetSpec,
    config,
    args: argparse.Namespace,
    rng: np.random.Generator,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    plan = metric_plan(args.mode, spec, config)
    if not plan:
        return [], [], []
    metric_cols = sorted({metric for _, metric, _ in plan})
    df = read_scores(spec.path(args.dataset_dir))
    human_table, metric_tables, systems = wide_tables(df, spec.human_col, metric_cols)
    first_pairs, first_human, metric_pair_map = build_common_pair_matrices(human_table, metric_tables, systems)
    max_sample = max(args.labeled_sizes) + args.unlabeled_size
    if first_human.shape[1] < max_sample:
        raise ValueError(f"{spec.label} has {first_human.shape[1]} aligned segments, needs {max_sample}.")

    run_rows: list[dict[str, Any]] = []
    pair_acc: dict[tuple[int, str, int], dict[str, Any]] = {}
    total = first_human.shape[1]
    for l_size in args.labeled_sizes:
        for run_idx in range(args.num_runs):
            sampled_idx = rng.choice(total, size=l_size + args.unlabeled_size, replace=False)
            labeled_idx = sampled_idx[:l_size]
            unlabeled_idx = sampled_idx[l_size:]
            true_effects = np.array([float(p["true_effect"]) for p in first_pairs], dtype=float)
            method_results = {
                "human_only": (
                    "Human-only",
                    human_z_intervals(first_human, labeled_idx, true_effects, args.alpha),
                )
            }
            for method, metric, label in plan:
                human_pairs = first_human
                metric_pairs = metric_pair_map[metric]
                true_effects = np.array([float(p["true_effect"]) for p in first_pairs], dtype=float)
                if args.mode == "gemba" and method == "auto_only":
                    result = human_z_intervals(metric_pairs, sampled_idx, true_effects, args.alpha)
                else:
                    result = ppi_intervals(
                        human_pairs, metric_pairs, labeled_idx, unlabeled_idx, true_effects, args.alpha
                    )
                method_results[method] = (label, result)
            for method, (label, (widths, covers)) in method_results.items():
                run_rows.append(
                    {
                        "dataset": spec.label,
                        "test_set": spec.test_set,
                        "language_pair": spec.language_pair,
                        "L": l_size,
                        "U": args.unlabeled_size,
                        "method": method,
                        "method_label": label,
                        "run_idx": run_idx,
                        "mean_interval_width": float(np.mean(widths)),
                        "coverage_rate": float(np.mean(covers)),
                    }
                )
                pairs = first_pairs
                for pair_idx, pair in enumerate(pairs):
                    key = (pair_idx, method, l_size)
                    acc = pair_acc.setdefault(
                        key,
                        {
                            "dataset": spec.label,
                            "test_set": spec.test_set,
                            "language_pair": spec.language_pair,
                            "system_a": pair["system_a"],
                            "system_b": pair["system_b"],
                            "true_effect": pair["true_effect"],
                            "aligned_segments": pair["aligned_segments"],
                            "L": l_size,
                            "U": args.unlabeled_size,
                            "method": method,
                            "method_label": label,
                            "valid_trials": 0,
                            "width_sum": 0.0,
                            "coverage_sum": 0.0,
                        },
                    )
                    acc["valid_trials"] += 1
                    acc["width_sum"] += float(widths[pair_idx])
                    acc["coverage_sum"] += float(covers[pair_idx])
    pair_rows = []
    for acc in pair_acc.values():
        n = acc.pop("valid_trials")
        width = acc.pop("width_sum")
        cover = acc.pop("coverage_sum")
        pair_rows.append({**acc, "valid_trials": n, "mean_interval_width": width / n, "coverage_rate": cover / n})
    meta = [
        {
            "dataset": spec.label,
            "test_set": spec.test_set,
            "language_pair": spec.language_pair,
            "num_systems": len(systems),
            "num_pairs": len(first_pairs),
            "aligned_segments": int(total),
        }
    ]
    return run_rows, pair_rows, meta


def summarize(run_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (dataset, l_size, method), group in run_df.groupby(["dataset", "L", "method"], sort=False):
        rows.append(
            {
                "dataset": dataset,
                "test_set": group["test_set"].iloc[0],
                "language_pair": group["language_pair"].iloc[0],
                "L": int(l_size),
                "U": int(group["U"].iloc[0]),
                "method": method,
                "method_label": group["method_label"].iloc[0],
                "num_runs": int(group.shape[0]),
                "mean_interval_width": float(group["mean_interval_width"].mean()),
                "median_interval_width": float(group["mean_interval_width"].median()),
                "mean_coverage_rate": float(group["coverage_rate"].mean()),
                "median_coverage_rate": float(group["coverage_rate"].median()),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    if args.seed is None:
        args.seed = 20260704 if args.mode == "gemba" else 20260514
    config = load_config(args.config)
    if args.plot_only:
        summary = pd.read_csv(args.output_dir / "interval_summary.csv")
    else:
        rng = np.random.default_rng(args.seed)
        specs = select_specs(config, args.datasets)
        run_rows: list[dict[str, Any]] = []
        pair_rows: list[dict[str, Any]] = []
        meta_rows: list[dict[str, Any]] = []
        for spec in specs:
            rr, pr, mr = run_dataset(spec, config, args, rng)
            run_rows.extend(rr)
            pair_rows.extend(pr)
            meta_rows.extend(mr)
            print(f"completed {spec.label}", flush=True)
        ensure_dir(args.output_dir)
        pd.DataFrame(run_rows).to_csv(args.output_dir / "interval_run_level.csv", index=False)
        pd.DataFrame(pair_rows).to_csv(args.output_dir / "interval_pairwise.csv", index=False)
        pd.DataFrame(meta_rows).to_csv(args.output_dir / "dataset_meta.csv", index=False)
        summary = summarize(pd.DataFrame(run_rows))
        summary.to_csv(args.output_dir / "interval_summary.csv", index=False)
        (args.output_dir / "config.json").write_text(json.dumps(vars(args), default=str, indent=2), encoding="utf-8")
    plot_ci_summary(summary, args.output_dir / f"{args.mode}_interval", title="Average interval width")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
