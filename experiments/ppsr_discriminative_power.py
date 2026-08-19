#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import copy
from itertools import combinations
from pathlib import Path
from typing import Callable

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib"))

import numpy as np
import pandas as pd

from ppi_mt_eval.config import load_config, select_specs
from ppi_mt_eval.data import filtered_metric_columns, is_reference_system, read_scores
from ppi_mt_eval.meta import META_METRICS as META_SCORE_FUNCTIONS
from ppi_mt_eval.meta import pairwise_p_values_from_signs
from ppi_mt_eval.plotting import ensure_dir
from ppi_mt_eval.progress import iter_progress


SYSTEM_META_METRICS = ("pearson", "spearman", "kendall", "spa", "ppsr")
SEGMENT_META_METRICS = ("input_r", "global_r", "system_r", "pdp", "ppsr")
META_METRICS = SYSTEM_META_METRICS
META_LABELS = {
    "input_r": "Group-by-Item r",
    "global_r": "No-Grouping r",
    "system_r": "Group-by-System r",
    "pdp": "PDP",
    "pearson": "Pearson r",
    "spearman": "Spearman rho",
    "kendall": "Kendall tau-b",
    "spa": "SPA",
    "ppsr": "PPSR",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Discriminative power of PPSR and baseline meta-metrics.")
    parser.add_argument("--config", type=Path, default=Path("configs/paper_datasets.json"))
    parser.add_argument("--dataset-dir", type=Path, default=Path("datasets"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/ppsr_discriminative_power"))
    parser.add_argument("--datasets", nargs="*", default=None)
    parser.add_argument("--num-permutations", type=int, default=1000)
    parser.add_argument(
        "--spa-inner-permutations",
        type=int,
        default=1000,
        help="Retained for CLI compatibility; SPA is computed through the shared meta module.",
    )
    parser.add_argument("--seed", type=int, default=20260516)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--meta-metrics", nargs="+", default=list(SYSTEM_META_METRICS), choices=sorted(META_LABELS))
    parser.add_argument(
        "--num-workers",
        type=int,
        default=1,
        help="Number of datasets to process in parallel.",
    )
    parser.add_argument("--max-metrics", type=int, default=None, help="Optional debug limit on automatic metrics.")
    parser.add_argument("--no-progress", action="store_true", help="Disable tqdm progress bars.")
    return parser.parse_args()


def random_masks(rng: np.random.Generator, rows: int, cols: int) -> np.ndarray:
    return rng.random(size=(rows, cols)) < 0.5


def random_signs(rng: np.random.Generator, rows: int, cols: int) -> np.ndarray:
    signs = rng.random(size=(rows, cols), dtype=np.float32)
    np.rint(signs, out=signs, casting="same_kind")
    signs *= 2.0
    signs -= 1.0
    return signs


def finite(value: float) -> float:
    return float(value) if np.isfinite(value) else float("nan")


def compute_meta_score(
    meta_metric: str,
    human: np.ndarray,
    metric: np.ndarray,
    human_pvals: np.ndarray | None = None,
    signs: np.ndarray | None = None,
) -> float:
    if meta_metric not in META_SCORE_FUNCTIONS:
        raise ValueError(meta_metric)
    if meta_metric == "spa":
        return finite(META_SCORE_FUNCTIONS[meta_metric](human, metric, human_pvals, signs))
    return finite(META_SCORE_FUNCTIONS[meta_metric](human, metric))


def load_matrices(spec, config, dataset_dir: Path):
    df = read_scores(spec.path(dataset_dir))
    metric_cols = filtered_metric_columns(df, spec.human_col, spec, config)
    for col in [spec.human_col, *metric_cols]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.loc[~df["system-name"].map(is_reference_system)].copy()

    human = df.pivot_table(index="seg-id", columns="system-name", values=spec.human_col, aggfunc="mean")
    systems = [system for system in human.columns if human[system].notna().any()]
    systems = list(human[systems].mean(axis=0).sort_values(ascending=False).index)
    human = human[systems]

    metric_tables: dict[str, pd.DataFrame] = {}
    retained: list[str] = []
    common_index = human.dropna().index
    for metric in metric_cols:
        table = df.pivot_table(index="seg-id", columns="system-name", values=metric, aggfunc="mean")
        if not set(systems).issubset(table.columns):
            continue
        table = table[systems]
        values = table.to_numpy(float)
        if values.size == 0 or np.all(~np.isfinite(values)):
            continue
        common_index = common_index.intersection(table.dropna().index)
        metric_tables[metric] = table
        retained.append(metric)

    human_matrix = human.loc[common_index, systems].to_numpy(float).T
    metric_mats = {
        metric: metric_tables[metric].loc[common_index, systems].to_numpy(float).T
        for metric in retained
    }
    return human_matrix, metric_mats, systems, common_index


def perm_inputs_pvalue(
    better: np.ndarray,
    worse: np.ndarray,
    score_fn: Callable[[np.ndarray], float],
    observed_delta: float,
    masks: np.ndarray,
) -> float:
    count = 0
    for mask in masks:
        swapped_better = better.copy()
        swapped_worse = worse.copy()
        swapped_better[:, mask] = worse[:, mask]
        swapped_worse[:, mask] = better[:, mask]
        delta = score_fn(swapped_better) - score_fn(swapped_worse)
        if np.isfinite(delta) and delta >= observed_delta:
            count += 1
    return (count + 1.0) / (len(masks) + 1.0)


def run_dataset(spec, config, args, dataset_index: int):
    human, mats, systems, aligned_index = load_matrices(spec, config, args.dataset_dir)
    metric_names = list(mats)
    if args.max_metrics is not None:
        metric_names = metric_names[: args.max_metrics]
        mats = {metric: mats[metric] for metric in metric_names}

    rng = np.random.default_rng(args.seed + 100_003 * dataset_index)
    signs_inner = random_signs(rng, args.spa_inner_permutations, human.shape[1])
    masks = random_masks(rng, args.num_permutations, human.shape[1])
    human_pvals = pairwise_p_values_from_signs(human, signs_inner)
    meta_metrics = tuple(args.meta_metrics)

    score_rows = []
    score_lookup: dict[tuple[str, str], float] = {}
    for metric in iter_progress(
        metric_names,
        enabled=not args.no_progress,
        desc=f"{spec.label} scores",
        unit="metric",
        leave=False,
    ):
        for meta_metric in meta_metrics:
            score = compute_meta_score(meta_metric, human, mats[metric], human_pvals, signs_inner)
            score_lookup[(meta_metric, metric)] = score
            score_rows.append(
                {
                    "dataset": spec.label,
                    "test_set": spec.test_set,
                    "language_pair": spec.language_pair,
                    "metric": metric,
                    "meta_metric": meta_metric,
                    "meta_metric_label": META_LABELS[meta_metric],
                    "score": score,
                    "aligned_segments": int(human.shape[1]),
                    "num_systems": int(human.shape[0]),
                }
            )

    sig_rows = []
    metric_pairs = list(combinations(metric_names, 2))
    for meta_metric in meta_metrics:
        for metric_a, metric_b in iter_progress(
            metric_pairs,
            enabled=not args.no_progress,
            desc=f"{spec.label} {meta_metric}",
            unit="pair",
            leave=False,
        ):
            score_a = score_lookup[(meta_metric, metric_a)]
            score_b = score_lookup[(meta_metric, metric_b)]
            if not np.isfinite(score_a) or not np.isfinite(score_b):
                better, worse = metric_a, metric_b
                observed = np.nan
                p_value = np.nan
            else:
                better, worse = (metric_a, metric_b) if score_a >= score_b else (metric_b, metric_a)
                observed = abs(score_a - score_b)

                def score_fn(matrix: np.ndarray, meta_metric: str = meta_metric) -> float:
                    return compute_meta_score(meta_metric, human, matrix, human_pvals, signs_inner)

                p_value = perm_inputs_pvalue(mats[better], mats[worse], score_fn, observed, masks)

            sig_rows.append(
                {
                    "dataset": spec.label,
                    "test_set": spec.test_set,
                    "language_pair": spec.language_pair,
                    "meta_metric": meta_metric,
                    "meta_metric_label": META_LABELS[meta_metric],
                    "metric_1": metric_a,
                    "metric_2": metric_b,
                    "better_metric": better,
                    "worse_metric": worse,
                    "observed_delta": observed,
                    "p_value": p_value,
                    "significant": bool(np.isfinite(p_value) and p_value <= args.alpha),
                    "num_permutations": args.num_permutations,
                    "spa_inner_permutations": args.spa_inner_permutations if meta_metric == "spa" else 0,
                }
            )

    meta = {
        "dataset": spec.label,
        "test_set": spec.test_set,
        "language_pair": spec.language_pair,
        "human_col": spec.human_col,
        "num_systems": len(systems),
        "num_metrics": len(metric_names),
        "aligned_segments": int(len(aligned_index)),
        "num_metric_pairs": len(metric_pairs),
    }
    return pd.DataFrame(score_rows), pd.DataFrame(sig_rows), meta


def summarize_scores(score_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (dataset, meta_metric), group in score_df.groupby(["dataset", "meta_metric"], sort=False):
        finite_scores = group["score"].replace([np.inf, -np.inf], np.nan).dropna().round(12)
        rows.append(
            {
                "dataset": dataset,
                "test_set": group["test_set"].iloc[0],
                "language_pair": group["language_pair"].iloc[0],
                "meta_metric": meta_metric,
                "meta_metric_label": group["meta_metric_label"].iloc[0],
                "num_metrics": int(group["metric"].nunique()),
                "num_finite_scores": int(finite_scores.shape[0]),
                "num_distinct_values": int(finite_scores.nunique()),
            }
        )
    return pd.DataFrame(rows)


def summarize_significance(sig_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (dataset, meta_metric), group in sig_df.groupby(["dataset", "meta_metric"], sort=False):
        finite_p = group["p_value"].replace([np.inf, -np.inf], np.nan).dropna()
        rows.append(
            {
                "dataset": dataset,
                "test_set": group["test_set"].iloc[0],
                "language_pair": group["language_pair"].iloc[0],
                "meta_metric": meta_metric,
                "meta_metric_label": group["meta_metric_label"].iloc[0],
                "num_metric_pairs": int(group.shape[0]),
                "num_finite_p_values": int(finite_p.shape[0]),
                "num_significant": int(group["significant"].sum()),
                "pct_significant": float(100.0 * group["significant"].mean()),
            }
        )
    return pd.DataFrame(rows)


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
    config = load_config(args.config)
    ensure_dir(args.output_dir)
    specs = list(select_specs(config, args.datasets))
    results = run_datasets(specs, config, args)
    score_frames = [scores for _idx, scores, _sig, _meta in results]
    sig_frames = [sig for _idx, _scores, sig, _meta in results]
    meta_rows = [meta for _idx, _scores, _sig, meta in results]

    score_df = pd.concat(score_frames, ignore_index=True)
    sig_df = pd.concat(sig_frames, ignore_index=True)
    distinct = summarize_scores(score_df)
    sig_summary = summarize_significance(sig_df)

    score_df.to_csv(args.output_dir / "metric_scores.csv", index=False)
    score_df.pivot_table(
        index=["dataset", "test_set", "language_pair", "metric"],
        columns="meta_metric",
        values="score",
        aggfunc="first",
    ).reset_index().to_csv(args.output_dir / "meta_metric_scores.csv", index=False)
    distinct.to_csv(args.output_dir / "distinct_value_summary.csv", index=False)
    sig_df.to_csv(args.output_dir / "pairwise_significance.csv", index=False)
    sig_summary.to_csv(args.output_dir / "significance_summary.csv", index=False)

    compat = distinct.merge(
        sig_summary,
        on=["dataset", "test_set", "language_pair", "meta_metric", "meta_metric_label"],
    )
    compat.rename(
        columns={
            "num_distinct_values": "distinct_values",
            "num_significant": "significant_comparisons",
            "num_metric_pairs": "max_significant_comparisons",
        }
    ).assign(max_distinct_values=lambda df: df["num_metrics"]).to_csv(
        args.output_dir / "discriminative_power_summary.csv",
        index=False,
    )
    pd.DataFrame(meta_rows).to_csv(args.output_dir / "dataset_meta.csv", index=False)
    (args.output_dir / "config.json").write_text(
        json.dumps(vars(args), default=str, indent=2),
        encoding="utf-8",
    )
    print(sig_summary.to_string(index=False))


if __name__ == "__main__":
    main()
