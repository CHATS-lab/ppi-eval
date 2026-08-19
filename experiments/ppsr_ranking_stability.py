#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import copy
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from ppi_mt_eval.config import load_config, select_specs
from ppi_mt_eval.plotting import ensure_dir, savefig
from ppi_mt_eval.progress import iter_progress, progress_bar
from ppsr_discriminative_power import (
    META_LABELS,
    SYSTEM_META_METRICS,
    compute_meta_score,
    load_matrices,
    pairwise_p_values_from_signs,
    random_signs,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Ranking stability of system-level meta-metrics.")
    p.add_argument("--config", type=Path, default=Path("configs/paper_datasets.json"))
    p.add_argument("--dataset-dir", type=Path, default=Path("datasets"))
    p.add_argument("--output-dir", type=Path, default=Path("results/ppsr_ranking_stability"))
    p.add_argument("--datasets", nargs="*", default=None)
    p.add_argument("--labeled-sizes", nargs="+", type=int, default=list(range(100, 1001, 100)))
    p.add_argument("--num-trials", type=int, default=1000)
    p.add_argument("--num-permutations", type=int, default=1000)
    p.add_argument("--seed", type=int, default=20260517)
    p.add_argument("--meta-metrics", nargs="+", default=list(SYSTEM_META_METRICS), choices=sorted(META_LABELS))
    p.add_argument("--figure-prefix", default="saving_ratio_ranking_stability")
    p.add_argument("--y-min", type=float, default=-0.05)
    p.add_argument("--y-max", type=float, default=1.05)
    p.add_argument("--max-metrics", type=int, default=None, help="Optional debug limit on automatic metrics.")
    p.add_argument("--num-workers", type=int, default=1, help="Number of datasets to process in parallel.")
    p.add_argument("--plot-only", action="store_true")
    p.add_argument("--no-progress", action="store_true", help="Disable tqdm progress bars.")
    return p.parse_args()


META_PLOT_LABELS = {
    "input_r": "Group-by-Item",
    "global_r": "No-Grouping",
    "system_r": "Group-by-System",
    "pdp": "PDP",
    "pearson": "Pearson r",
    "spearman": "Spearman rho",
    "kendall": "Kendall tau-b",
    "spa": "SPA",
    "ppsr": "PPSR",
}

META_COLORS = {
    "pearson": "#4C78A8",
    "spearman": "#F58518",
    "kendall": "#E45756",
    "spa": "#B279A2",
    "ppsr": "#54A24B",
    "input_r": "#4C78A8",
    "global_r": "#F58518",
    "system_r": "#B279A2",
    "pdp": "#E45756",
}

SYSTEM_PLOT_ORDER = ("pearson", "spearman", "kendall", "spa", "ppsr")


def score_all(
    human: np.ndarray,
    mats: dict[str, np.ndarray],
    metric_names: list[str],
    signs: np.ndarray,
    meta_metrics: tuple[str, ...],
) -> dict[str, np.ndarray]:
    human_pvals = pairwise_p_values_from_signs(human, signs)
    scores = {name: [] for name in meta_metrics}
    for metric in metric_names:
        for meta_metric in meta_metrics:
            scores[meta_metric].append(compute_meta_score(meta_metric, human, mats[metric], human_pvals, signs))
    return {name: np.asarray(values, dtype=float) for name, values in scores.items()}


def finite_kendall(reference: np.ndarray, sampled: np.ndarray) -> float:
    mask = np.isfinite(reference) & np.isfinite(sampled)
    if int(mask.sum()) < 2:
        return np.nan
    value = stats.kendalltau(reference[mask], sampled[mask], variant="b").statistic
    return float(value) if np.isfinite(value) else np.nan


def run_dataset(spec, config, args, dataset_index: int):
    human, mats, systems, aligned_index = load_matrices(spec, config, args.dataset_dir)
    metric_names = list(mats)
    if args.max_metrics is not None:
        metric_names = metric_names[: args.max_metrics]
        mats = {metric: mats[metric] for metric in metric_names}
    too_large = [l_size for l_size in args.labeled_sizes if l_size > human.shape[1]]
    if too_large:
        raise ValueError(f"{spec.label} has {human.shape[1]} aligned segments; invalid L values: {too_large}")

    full_rng = np.random.default_rng(args.seed + 100_003 * dataset_index)
    full_signs = random_signs(full_rng, args.num_permutations, human.shape[1])
    meta_metrics = tuple(args.meta_metrics)
    full_scores = score_all(human, mats, metric_names, full_signs, meta_metrics)
    trial_seed = args.seed + 10_000_019 * dataset_index

    rows = []
    full_rows = []
    for metric_idx, metric in enumerate(metric_names):
        for meta_metric in meta_metrics:
            full_rows.append(
                {
                    "dataset": spec.label,
                    "test_set": spec.test_set,
                    "language_pair": spec.language_pair,
                    "metric": metric,
                    "meta_metric": meta_metric,
                    "meta_metric_label": META_LABELS[meta_metric],
                    "full_score": full_scores[meta_metric][metric_idx],
                    "aligned_segments": human.shape[1],
                    "num_systems": human.shape[0],
                }
            )
    total_trials = len(args.labeled_sizes) * args.num_trials
    with progress_bar(
        enabled=not args.no_progress,
        total=total_trials,
        desc=f"{spec.label} trials",
        unit="trial",
        leave=False,
    ) as bar:
        for l_size in args.labeled_sizes:
            for trial in range(args.num_trials):
                rng = np.random.default_rng(trial_seed + 1_000_003 * l_size + trial)
                sampled_cols = rng.choice(human.shape[1], size=l_size, replace=False)
                sampled_human = human[:, sampled_cols]
                sampled_mats = {name: matrix[:, sampled_cols] for name, matrix in mats.items()}
                signs = random_signs(rng, args.num_permutations, l_size)
                sampled_scores = score_all(sampled_human, sampled_mats, metric_names, signs, meta_metrics)
                for meta_metric in meta_metrics:
                    tau = finite_kendall(full_scores[meta_metric], sampled_scores[meta_metric])
                    rows.append(
                        {
                            "dataset": spec.label,
                            "test_set": spec.test_set,
                            "language_pair": spec.language_pair,
                            "labeled_size": l_size,
                            "trial": trial,
                            "meta_metric": meta_metric,
                            "meta_metric_label": META_LABELS[meta_metric],
                            "kendall_tau_b": tau,
                            "num_permutations": args.num_permutations,
                            "num_metrics": len(metric_names),
                            "aligned_segments": human.shape[1],
                        }
                    )
                bar.update(1)
    meta = {
        "dataset": spec.label,
        "test_set": spec.test_set,
        "language_pair": spec.language_pair,
        "human_col": spec.human_col,
        "num_systems": len(systems),
        "num_metrics": len(metric_names),
        "aligned_segments": len(aligned_index),
        "num_trials": args.num_trials,
        "num_permutations": args.num_permutations,
    }
    return pd.DataFrame(rows), pd.DataFrame(full_rows), meta


def summarize(trials: pd.DataFrame) -> pd.DataFrame:
    rows = []
    group_cols = ["dataset", "test_set", "language_pair", "labeled_size", "meta_metric", "meta_metric_label"]
    for key, group in trials.groupby(group_cols, sort=False):
        values = group["kendall_tau_b"].replace([np.inf, -np.inf], np.nan).dropna()
        rows.append(
            {
                **dict(zip(group_cols, key, strict=True)),
                "mean_kendall_tau_b": float(values.mean()) if len(values) else np.nan,
                "std_kendall_tau_b": float(values.std(ddof=1)) if len(values) > 1 else np.nan,
                "num_trials": int(group.shape[0]),
                "num_finite_trials": int(values.shape[0]),
            }
        )
    return pd.DataFrame(rows)


def plot(summary: pd.DataFrame, output: Path, figure_prefix: str, y_limits: tuple[float, float]) -> None:
    datasets = list(summary["dataset"].drop_duplicates())
    present_metrics = set(summary["meta_metric"].drop_duplicates())
    metric_order = [metric for metric in SYSTEM_PLOT_ORDER if metric in present_metrics]
    metric_order.extend(metric for metric in summary["meta_metric"].drop_duplicates() if metric not in metric_order)
    ncols = 3
    nrows = int(np.ceil(len(datasets) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(13, 3.2 * nrows), sharex=True, sharey=True)
    axes = np.asarray(axes).ravel()
    for ax, dataset in zip(axes, datasets, strict=False):
        sub = summary[summary["dataset"] == dataset]
        for metric in metric_order:
            group = sub[sub["meta_metric"] == metric]
            if group.empty:
                continue
            group = group.sort_values("labeled_size")
            ax.plot(
                group["labeled_size"],
                group["mean_kendall_tau_b"],
                marker="o",
                linewidth=1.8,
                markersize=3.5,
                label=META_PLOT_LABELS.get(metric, metric),
                color=META_COLORS.get(metric),
            )
        ax.set_title(dataset)
        ax.grid(alpha=0.25)
        ax.set_ylim(*y_limits)
    for ax in axes[len(datasets) :]:
        ax.axis("off")
    for ax in axes[-ncols:]:
        ax.set_xlabel("Sampled inputs (L)")
    for idx in range(0, len(axes), ncols):
        axes[idx].set_ylabel("Average Kendall's $\\tau_b$")
    handles, labs = axes[0].get_legend_handles_labels()
    fig.legend(handles, labs, loc="lower center", ncol=5, frameon=False)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    combined_name = f"{figure_prefix}_all_datasets.pdf"
    savefig(fig, output / combined_name)
    for dataset in datasets:
        tag = dataset.replace(" ", "_").replace("-", "_")
        fig, ax = plt.subplots(figsize=(6.4, 4.0))
        sub = summary[summary["dataset"] == dataset]
        for metric in metric_order:
            group = sub[sub["meta_metric"] == metric]
            if group.empty:
                continue
            group = group.sort_values("labeled_size")
            ax.plot(
                group["labeled_size"],
                group["mean_kendall_tau_b"],
                marker="o",
                linewidth=1.9,
                markersize=4,
                label=META_PLOT_LABELS.get(metric, metric),
                color=META_COLORS.get(metric),
            )
        ax.set_title(f"Ranking stability: {dataset}")
        ax.set_xlabel("Sampled inputs (L)")
        ax.set_ylabel("Average Kendall's $\\tau_b$")
        ax.grid(alpha=0.25)
        ax.set_ylim(*y_limits)
        ax.legend(fontsize=8, ncol=3)
        figure_name = f"{figure_prefix}_{tag}.pdf"
        savefig(fig, output / figure_name)


def run_datasets(specs, config, args):
    def run_serial():
        results = []
        for dataset_index, spec in enumerate(
            iter_progress(specs, enabled=not args.no_progress, desc="datasets", unit="dataset")
        ):
            results.append((dataset_index, *run_dataset(spec, config, args, dataset_index)))
            print(f"completed {spec.label}", flush=True)
        return results

    if args.num_workers <= 1:
        return run_serial()

    worker_args = copy(args)
    worker_args.no_progress = True
    results = []
    try:
        with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
            futures = {
                executor.submit(run_dataset, spec, config, worker_args, dataset_index): (dataset_index, spec)
                for dataset_index, spec in enumerate(specs)
            }
            for future in iter_progress(
                as_completed(futures),
                enabled=not args.no_progress,
                total=len(futures),
                desc="datasets",
                unit="dataset",
            ):
                dataset_index, spec = futures[future]
                results.append((dataset_index, *future.result()))
                print(f"completed {spec.label}", flush=True)
    except PermissionError as exc:
        print(f"ProcessPoolExecutor unavailable ({exc}); falling back to serial execution.", flush=True)
        return run_serial()
    return sorted(results, key=lambda item: item[0])


def main() -> None:
    args = parse_args()
    ensure_dir(args.output_dir)
    if args.plot_only:
        summary = pd.read_csv(args.output_dir / "ranking_stability_summary.csv")
    else:
        config = load_config(args.config)
        specs = list(select_specs(config, args.datasets))
        results = run_datasets(specs, config, args)
        trial_frames = [trials for _idx, trials, _full, _meta in results]
        full_frames = [full for _idx, _trials, full, _meta in results]
        meta_rows = [meta for _idx, _trials, _full, meta in results]
        trial_df = pd.concat(trial_frames, ignore_index=True)
        summary = summarize(trial_df)
        trial_df.to_csv(args.output_dir / "ranking_stability_trials.csv", index=False)
        summary.to_csv(args.output_dir / "ranking_stability_summary.csv", index=False)
        pd.concat(full_frames, ignore_index=True).to_csv(args.output_dir / "full_dataset_meta_scores.csv", index=False)
        pd.DataFrame(meta_rows).to_csv(args.output_dir / "dataset_meta.csv", index=False)
        (args.output_dir / "config.json").write_text(json.dumps(vars(args), default=str, indent=2), encoding="utf-8")
    plot(summary, args.output_dir, args.figure_prefix, (args.y_min, args.y_max))


if __name__ == "__main__":
    main()
