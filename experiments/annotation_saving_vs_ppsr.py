#!/usr/bin/env python3
"""Compare empirical annotation savings with PPSR rankings.

For each dataset, system pair, and metric, this script estimates the minimum
number of human annotations needed to reach 80% power under the human-only and
prediction-powered paired Z-tests. It then correlates total empirical savings
with PPSR.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats as scipy_stats
from scipy.optimize import brentq

from ppi_mt_eval.config import DatasetSpec, load_config, select_specs
from ppi_mt_eval.data import HUMAN_SEG_COLS, filtered_metric_columns, is_reference_system, read_scores
from ppi_mt_eval.intervals import human_z_reject, ppi_z_reject

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover
    tqdm = None


DEFAULT_TARGET_POWER = 0.80
DEFAULT_ALPHA = 0.05
DEFAULT_NUM_TRIALS = 1000
DEFAULT_SEED = 20260521
DEFAULT_CONFIG = Path("configs/paper_datasets.json")
METHOD_COLORS = {
    "total_saved_annotations": "#4C78A8",
    "ppsr": "#54A24B",
}


@dataclass(frozen=True)
class PairTask:
    dataset: str
    test_set: str
    language_pair: str
    pair_index: int
    system_a: str
    system_b: str
    human_diff: np.ndarray
    metric_diffs: dict[str, np.ndarray]
    metric_names: tuple[str, ...]
    num_trials: int
    alpha: float
    target_power: float
    seed: int
    brent_xtol: float
    min_labeled: int


@dataclass(frozen=True)
class ThresholdResult:
    threshold: float
    status: str
    lower_power: float
    upper_power: float


def load_dataset_configs(config_path: Path, dataset_specs: list[str]) -> list[DatasetSpec]:
    return select_specs(load_config(config_path), dataset_specs)


def load_dataset_matrices(
    spec: DatasetSpec,
    dataset_dir: Path,
    config_path: Path,
    max_metrics: int,
) -> tuple[np.ndarray, dict[str, np.ndarray], list[str], list[Any], dict[str, Any]]:
    config = load_config(config_path)
    df = read_scores(spec.path(dataset_dir))
    metric_cols = filtered_metric_columns(df, spec.human_col, spec, config)
    if max_metrics > 0:
        metric_cols = metric_cols[:max_metrics]
    for col in [spec.human_col, *metric_cols]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.loc[~df["system-name"].map(is_reference_system)].copy()

    human_table = df.pivot_table(index="seg-id", columns="system-name", values=spec.human_col, aggfunc="mean")
    systems = [system for system in human_table.columns if human_table[system].notna().any()]
    systems = list(human_table[systems].mean(axis=0).sort_values(ascending=False).index)
    human_table = human_table[systems]

    metric_tables: dict[str, pd.DataFrame] = {}
    retained_metrics: list[str] = []
    common_index = human_table.dropna().index
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
        retained_metrics.append(metric)

    human = human_table.loc[common_index, systems].to_numpy(float).T
    metrics = {
        metric: metric_tables[metric].loc[common_index, systems].to_numpy(float).T
        for metric in retained_metrics
    }
    meta = {
        "dataset": spec.label,
        "test_set": spec.test_set,
        "language_pair": spec.language_pair,
        "human_col": spec.human_col,
        "num_systems": len(systems),
        "num_metrics": len(retained_metrics),
        "aligned_segments": int(len(common_index)),
        "raw_metric_candidates": int(len(metric_cols)),
    }
    return human, metrics, systems, list(common_index), meta


def progress(iterable, **kwargs):
    if tqdm is None:
        return iterable
    return tqdm(iterable, **kwargs)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--dataset-dir", type=Path, default=Path("datasets"))
    parser.add_argument(
        "--ppsr-scores",
        type=Path,
        default=Path("results/ppsr_discriminative_power/metric_scores.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/annotation_saving_vs_ppsr"),
    )
    parser.add_argument(
        "--datasets",
        nargs="*",
        default=None,
    )
    parser.add_argument("--num-trials", type=int, default=DEFAULT_NUM_TRIALS)
    parser.add_argument("--alpha", type=float, default=DEFAULT_ALPHA)
    parser.add_argument("--target-power", type=float, default=DEFAULT_TARGET_POWER)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--brent-xtol", type=float, default=1.0)
    parser.add_argument("--min-labeled", type=int, default=2)
    parser.add_argument("--max-metrics", type=int, default=0)
    parser.add_argument("--max-pairs-per-dataset", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    parser.add_argument("--estimate-only", action="store_true")
    parser.add_argument("--skip-estimate", action="store_true")
    parser.add_argument("--pilot-trials", type=int, default=100)
    parser.add_argument("--pilot-pairs-per-dataset", type=int, default=3)
    parser.add_argument("--pilot-max-metrics", type=int, default=0)
    parser.add_argument(
        "--postprocess-existing",
        action="store_true",
        help="Reuse output-dir/pair_metric_thresholds.csv and recompute filtered savings/PPSR summaries.",
    )
    return parser.parse_args()


class PairPowerEvaluator:
    def __init__(
        self,
        human_diff: np.ndarray,
        metric_diffs: dict[str, np.ndarray],
        permutations: np.ndarray,
        alpha: float,
        target_power: float,
    ) -> None:
        self.human_diff = human_diff
        self.metric_diffs = metric_diffs
        self.permutations = permutations
        self.alpha = alpha
        self.target_power = target_power
        self.cache: dict[tuple[str, str, int], float] = {}

    def power(self, method: str, metric: str, labeled_size: int) -> float:
        key = (method, metric, int(labeled_size))
        if key in self.cache:
            return self.cache[key]
        labeled_idx = self.permutations[:, :labeled_size]
        if method == "human":
            values = self.human_diff[labeled_idx]
            power = float(np.mean(human_z_reject(values, self.alpha)))
        elif method == "ppi":
            metric_diff = self.metric_diffs[metric]
            unlabeled_idx = self.permutations[:, labeled_size:]
            rejections = ppi_z_reject(
                self.human_diff[labeled_idx],
                metric_diff[labeled_idx],
                metric_diff[unlabeled_idx],
                self.alpha,
            )
            power = float(np.mean(rejections))
        else:
            raise ValueError(f"Unknown method: {method}")
        self.cache[key] = power
        return power

    def objective(self, method: str, metric: str, lower: int, upper: int, labeled_size: float) -> float:
        labeled_int = int(np.clip(round(labeled_size), lower, upper))
        return self.power(method, metric, labeled_int) - self.target_power


def find_threshold(
    evaluator: PairPowerEvaluator,
    method: str,
    metric: str,
    lower: int,
    upper: int,
    xtol: float,
) -> ThresholdResult:
    lower_power = evaluator.power(method, metric, lower)
    upper_power = evaluator.power(method, metric, upper)
    if not np.isfinite(lower_power) or not np.isfinite(upper_power):
        return ThresholdResult(float("nan"), "invalid_power", lower_power, upper_power)
    if lower_power >= evaluator.target_power:
        return ThresholdResult(float(lower), "already_reaches_target_at_lower", lower_power, upper_power)
    if upper_power < evaluator.target_power:
        return ThresholdResult(float("nan"), "does_not_reach_target", lower_power, upper_power)
    root = brentq(
        lambda x: evaluator.objective(method, metric, lower, upper, x),
        lower,
        upper,
        xtol=xtol,
    )
    threshold = int(np.clip(math.ceil(root), lower, upper))
    while threshold > lower and evaluator.power(method, metric, threshold - 1) >= evaluator.target_power:
        threshold -= 1
    while threshold < upper and evaluator.power(method, metric, threshold) < evaluator.target_power:
        threshold += 1
    return ThresholdResult(float(threshold), "brentq", lower_power, upper_power)


def stable_pair_seed(seed: int, dataset: str, pair_index: int) -> int:
    dataset_code = sum((idx + 1) * ord(char) for idx, char in enumerate(dataset))
    seed_seq = np.random.SeedSequence([seed, dataset_code, int(pair_index)])
    return int(seed_seq.generate_state(1, dtype=np.uint32)[0])


def run_pair_task(task: PairTask) -> list[dict[str, Any]]:
    population_size = int(task.human_diff.shape[0])
    lower = int(task.min_labeled)
    upper = population_size - 2
    if lower >= upper:
        raise ValueError(f"Invalid threshold bracket for {task.dataset} pair {task.pair_index}")

    rng = np.random.default_rng(stable_pair_seed(task.seed, task.dataset, task.pair_index))
    permutations = np.vstack([rng.permutation(population_size) for _ in range(task.num_trials)])
    evaluator = PairPowerEvaluator(
        task.human_diff,
        task.metric_diffs,
        permutations,
        task.alpha,
        task.target_power,
    )
    human_threshold = find_threshold(evaluator, "human", "human", lower, upper, task.brent_xtol)
    true_effect = float(np.mean(task.human_diff))
    rows = []
    for metric in task.metric_names:
        metric_threshold = find_threshold(evaluator, "ppi", metric, lower, upper, task.brent_xtol)
        saving = (
            human_threshold.threshold - metric_threshold.threshold
            if np.isfinite(human_threshold.threshold) and np.isfinite(metric_threshold.threshold)
            else np.nan
        )
        rows.append(
            {
                "dataset": task.dataset,
                "test_set": task.test_set,
                "language_pair": task.language_pair,
                "pair_index": task.pair_index,
                "system_a": task.system_a,
                "system_b": task.system_b,
                "metric": metric,
                "population_size_M": population_size,
                "true_effect": true_effect,
                "human_threshold_L_h": human_threshold.threshold,
                "human_threshold_status": human_threshold.status,
                "human_lower_power": human_threshold.lower_power,
                "human_upper_power": human_threshold.upper_power,
                "metric_threshold_L_k": metric_threshold.threshold,
                "metric_threshold_status": metric_threshold.status,
                "metric_lower_power": metric_threshold.lower_power,
                "metric_upper_power": metric_threshold.upper_power,
                "annotations_saved": saving,
                "num_trials": task.num_trials,
                "alpha": task.alpha,
                "target_power": task.target_power,
            }
        )
    return rows


def build_tasks(
    config_path: Path,
    dataset_dir: Path,
    dataset_specs: list[str],
    max_metrics: int,
    num_trials: int,
    alpha: float,
    target_power: float,
    seed: int,
    brent_xtol: float,
    min_labeled: int,
    max_pairs_per_dataset: int = 0,
) -> tuple[list[PairTask], pd.DataFrame]:
    configs = load_dataset_configs(config_path, dataset_specs)
    tasks: list[PairTask] = []
    meta_rows = []
    for config in configs:
        human, metrics, systems, segment_ids, meta = load_dataset_matrices(config, dataset_dir, config_path, max_metrics)
        pair_iter = list(combinations(range(len(systems)), 2))
        if max_pairs_per_dataset > 0:
            pair_iter = pair_iter[:max_pairs_per_dataset]
        for pair_index, (idx_a, idx_b) in enumerate(pair_iter):
            human_diff = human[idx_a] - human[idx_b]
            true_effect = float(np.mean(human_diff))
            if not np.isfinite(true_effect) or np.isclose(true_effect, 0.0):
                continue
            system_a = systems[idx_a]
            system_b = systems[idx_b]
            sign = 1.0
            if true_effect < 0:
                sign = -1.0
                human_diff = -human_diff
                system_a, system_b = system_b, system_a
            metric_names = tuple(metrics.keys())
            metric_diffs = {
                metric: sign * (matrix[idx_a] - matrix[idx_b])
                for metric, matrix in metrics.items()
            }
            tasks.append(
                PairTask(
                    dataset=config.label,
                    test_set=config.test_set,
                    language_pair=config.language_pair,
                    pair_index=pair_index,
                    system_a=system_a,
                    system_b=system_b,
                    human_diff=human_diff.astype(np.float64, copy=False),
                    metric_diffs=metric_diffs,
                    metric_names=metric_names,
                    num_trials=num_trials,
                    alpha=alpha,
                    target_power=target_power,
                    seed=seed,
                    brent_xtol=brent_xtol,
                    min_labeled=min_labeled,
                )
            )
        meta = dict(meta)
        meta["num_tasks"] = sum(1 for task in tasks if task.dataset == config.label)
        meta["num_system_pairs_requested"] = len(pair_iter)
        meta["segment_ids_count"] = len(segment_ids)
        meta_rows.append(meta)
    return tasks, pd.DataFrame(meta_rows)


def run_tasks(tasks: list[PairTask], num_workers: int, desc: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if num_workers <= 1:
        for task in progress(tasks, desc=desc, total=len(tasks)):
            rows.extend(run_pair_task(task))
        return pd.DataFrame(rows)
    try:
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(run_pair_task, task) for task in tasks]
            for future in progress(as_completed(futures), desc=desc, total=len(futures)):
                rows.extend(future.result())
        return pd.DataFrame(rows)
    except PermissionError as exc:
        print(
            f"ProcessPoolExecutor unavailable ({exc}); falling back to serial execution.",
            flush=True,
        )
        for task in progress(tasks, desc=f"{desc} (serial fallback)", total=len(tasks)):
            rows.extend(run_pair_task(task))
        return pd.DataFrame(rows)


def estimate_runtime(
    all_tasks: list[PairTask],
    config_path: Path,
    dataset_dir: Path,
    dataset_specs: list[str],
    max_metrics: int,
    pilot_trials: int,
    pilot_pairs_per_dataset: int,
    pilot_max_metrics: int,
    alpha: float,
    target_power: float,
    seed: int,
    brent_xtol: float,
    min_labeled: int,
    num_workers: int,
) -> dict[str, Any]:
    pilot_tasks, _ = build_tasks(
        config_path=config_path,
        dataset_dir=dataset_dir,
        dataset_specs=dataset_specs,
        max_metrics=pilot_max_metrics if pilot_max_metrics > 0 else max_metrics,
        num_trials=pilot_trials,
        alpha=alpha,
        target_power=target_power,
        seed=seed,
        brent_xtol=brent_xtol,
        min_labeled=min_labeled,
        max_pairs_per_dataset=pilot_pairs_per_dataset,
    )
    start = time.perf_counter()
    pilot_df = run_tasks(pilot_tasks, min(num_workers, max(1, len(pilot_tasks))), "pilot")
    pilot_seconds = time.perf_counter() - start
    pilot_cases = max(1, len(pilot_df))
    total_cases = sum(len(task.metric_names) for task in all_tasks)
    serial_seconds = pilot_seconds / pilot_cases * total_cases * (DEFAULT_NUM_TRIALS / pilot_trials)
    requested_seconds = pilot_seconds / pilot_cases * total_cases * (all_tasks[0].num_trials / pilot_trials)
    parallel_seconds = requested_seconds / max(1, min(num_workers, len(all_tasks)))
    return {
        "pilot_seconds": pilot_seconds,
        "pilot_trials": pilot_trials,
        "pilot_tasks": len(pilot_tasks),
        "pilot_pair_metric_cases": int(pilot_cases),
        "total_tasks": len(all_tasks),
        "total_pair_metric_cases": int(total_cases),
        "num_workers": num_workers,
        "estimated_serial_seconds_for_default_1000_trials": serial_seconds,
        "estimated_serial_seconds_for_requested_trials": requested_seconds,
        "estimated_parallel_seconds_for_requested_trials": parallel_seconds,
    }


def pair_corr_sq(human_a: np.ndarray, human_b: np.ndarray, metric_a: np.ndarray, metric_b: np.ndarray) -> float:
    human_diff = human_a - human_b
    metric_diff = metric_a - metric_b
    if np.isclose(np.std(human_diff), 0.0) or np.isclose(np.std(metric_diff), 0.0):
        return np.nan
    corr = np.corrcoef(human_diff, metric_diff)[0, 1]
    return float(corr * corr) if np.isfinite(corr) else np.nan


def retained_human_pairs(pair_metric: pd.DataFrame) -> pd.DataFrame:
    pair_cols = ["dataset", "test_set", "language_pair", "pair_index", "system_a", "system_b"]
    pairs = pair_metric[pair_cols + ["human_threshold_status"]].drop_duplicates(pair_cols)
    return pairs[~pairs["human_threshold_status"].eq("does_not_reach_target")].copy()


def recompute_filtered_ppsr(
    config_path: Path,
    dataset_dir: Path,
    dataset_specs: list[str],
    pair_metric: pd.DataFrame,
    max_metrics: int,
) -> pd.DataFrame:
    configs = load_dataset_configs(config_path, dataset_specs)
    retained_pairs = retained_human_pairs(pair_metric)
    rows: list[dict[str, Any]] = []
    for config in configs:
        human, metrics, systems, _segment_ids, _meta = load_dataset_matrices(config, dataset_dir, config_path, max_metrics)
        pair_map = {
            pair_index: (idx_a, idx_b)
            for pair_index, (idx_a, idx_b) in enumerate(combinations(range(len(systems)), 2))
        }
        dataset_pairs = retained_pairs[retained_pairs["dataset"].eq(config.label)]
        retained_indices = sorted(int(value) for value in dataset_pairs["pair_index"].drop_duplicates())
        for metric, matrix in metrics.items():
            corr_values = []
            for pair_index in retained_indices:
                if pair_index not in pair_map:
                    continue
                idx_a, idx_b = pair_map[pair_index]
                corr_sq = pair_corr_sq(human[idx_a], human[idx_b], matrix[idx_a], matrix[idx_b])
                if np.isfinite(corr_sq):
                    corr_values.append(corr_sq)
            rows.append(
                {
                    "dataset": config.label,
                    "test_set": config.test_set,
                    "language_pair": config.language_pair,
                    "metric": metric,
                    "ppsr": float(np.mean(corr_values)) if corr_values else np.nan,
                    "ppsr_num_pairs": int(len(corr_values)),
                    "num_retained_pairs": int(len(retained_indices)),
                }
            )
    return pd.DataFrame(rows)


def pair_level_saved_corr_rows(
    config_path: Path,
    dataset_dir: Path,
    dataset_specs: list[str],
    pair_metric: pd.DataFrame,
    max_metrics: int,
) -> pd.DataFrame:
    configs = load_dataset_configs(config_path, dataset_specs)
    retained_pairs = retained_human_pairs(pair_metric)
    rows: list[dict[str, Any]] = []
    for config in configs:
        human, metrics, systems, _segment_ids, _meta = load_dataset_matrices(config, dataset_dir, config_path, max_metrics)
        pair_map = {
            pair_index: (idx_a, idx_b)
            for pair_index, (idx_a, idx_b) in enumerate(combinations(range(len(systems)), 2))
        }
        dataset_pairs = retained_pairs[retained_pairs["dataset"].eq(config.label)]
        for pair in dataset_pairs.itertuples(index=False):
            pair_index = int(pair.pair_index)
            if pair_index not in pair_map:
                continue
            idx_a, idx_b = pair_map[pair_index]
            pair_thresholds = pair_metric[
                pair_metric["dataset"].eq(config.label)
                & pair_metric["pair_index"].eq(pair_index)
            ]
            for metric, matrix in metrics.items():
                threshold_row = pair_thresholds[pair_thresholds["metric"].eq(metric)]
                if threshold_row.empty:
                    continue
                saved = float(threshold_row["annotations_saved"].iloc[0])
                corr_sq = pair_corr_sq(human[idx_a], human[idx_b], matrix[idx_a], matrix[idx_b])
                rows.append(
                    {
                        "dataset": config.label,
                        "test_set": config.test_set,
                        "language_pair": config.language_pair,
                        "pair_index": pair_index,
                        "system_a": pair.system_a,
                        "system_b": pair.system_b,
                        "metric": metric,
                        "annotations_saved": saved,
                        "corr_sq": corr_sq,
                        "human_threshold_status": pair.human_threshold_status,
                    }
                )
    return pd.DataFrame(rows)


def summarize_savings(
    pair_metric: pd.DataFrame,
    filtered_ppsr: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    retained_pairs = retained_human_pairs(pair_metric)
    pair_cols = ["dataset", "test_set", "language_pair", "pair_index", "system_a", "system_b"]
    filtered_pair_metric = pair_metric.merge(
        retained_pairs[pair_cols],
        on=pair_cols,
        how="inner",
    )
    valid = filtered_pair_metric[np.isfinite(filtered_pair_metric["annotations_saved"])].copy()
    savings = (
        valid.groupby(["dataset", "test_set", "language_pair", "metric"], as_index=False)
        .agg(
            total_saved_annotations=("annotations_saved", "sum"),
            mean_saved_annotations=("annotations_saved", "mean"),
            num_valid_saving_pairs=("annotations_saved", "size"),
        )
    )
    total_pairs = (
        filtered_pair_metric.groupby(["dataset", "test_set", "language_pair", "metric"], as_index=False)
        .size()
        .rename(columns={"size": "num_retained_pair_metric_rows"})
    )
    original_pairs = (
        pair_metric.groupby(["dataset", "test_set", "language_pair", "metric"], as_index=False)
        .size()
        .rename(columns={"size": "num_original_pair_metric_rows"})
    )
    savings = savings.merge(total_pairs, on=["dataset", "test_set", "language_pair", "metric"], how="right")
    savings["total_saved_annotations"] = savings["total_saved_annotations"].fillna(0.0)
    savings["mean_saved_annotations"] = savings["mean_saved_annotations"].fillna(np.nan)
    savings["num_valid_saving_pairs"] = savings["num_valid_saving_pairs"].fillna(0).astype(int)
    savings = savings.merge(original_pairs, on=["dataset", "test_set", "language_pair", "metric"], how="left")
    savings["num_excluded_human_unreachable_pairs"] = (
        savings["num_original_pair_metric_rows"] - savings["num_retained_pair_metric_rows"]
    ).astype(int)
    savings = savings.merge(
        filtered_ppsr,
        on=["dataset", "test_set", "language_pair", "metric"],
        how="left",
    )
    savings["saving_rank"] = savings.groupby("dataset")["total_saved_annotations"].rank(
        method="min", ascending=False
    )
    savings["ppsr_rank"] = savings.groupby("dataset")["ppsr"].rank(method="min", ascending=False)

    rows = []
    for dataset, group in savings.groupby("dataset", sort=False):
        mask = np.isfinite(group["total_saved_annotations"]) & np.isfinite(group["ppsr"])
        if mask.sum() >= 2:
            pearson_result = scipy_stats.pearsonr(
                group.loc[mask, "total_saved_annotations"],
                group.loc[mask, "ppsr"],
            )
            pearson_r = float(pearson_result.statistic) if np.isfinite(pearson_result.statistic) else np.nan
            pearson_p_value = float(pearson_result.pvalue) if np.isfinite(pearson_result.pvalue) else np.nan
            spearman_result = scipy_stats.spearmanr(
                group.loc[mask, "total_saved_annotations"],
                group.loc[mask, "ppsr"],
            )
            spearman_rho = (
                float(spearman_result.statistic)
                if np.isfinite(spearman_result.statistic)
                else np.nan
            )
            spearman_p_value = (
                float(spearman_result.pvalue)
                if np.isfinite(spearman_result.pvalue)
                else np.nan
            )
        else:
            pearson_r = np.nan
            pearson_p_value = np.nan
            spearman_rho = np.nan
            spearman_p_value = np.nan
        rows.append(
            {
                "dataset": dataset,
                "test_set": group["test_set"].iloc[0],
                "language_pair": group["language_pair"].iloc[0],
                "pearson_r": pearson_r,
                "pearson_p_value": pearson_p_value,
                "spearman_rho": spearman_rho,
                "spearman_p_value": spearman_p_value,
                "num_finite_metrics": int(mask.sum()),
                "num_metrics": int(group.shape[0]),
                "total_valid_pair_metric_cases": int(group["num_valid_saving_pairs"].sum()),
                "total_retained_pair_metric_cases": int(group["num_retained_pair_metric_rows"].sum()),
                "total_original_pair_metric_cases": int(group["num_original_pair_metric_rows"].sum()),
                "total_excluded_human_unreachable_cases": int(
                    group["num_excluded_human_unreachable_pairs"].sum()
                ),
            }
        )
    return savings, pd.DataFrame(rows)


def plot_outputs(metric_savings: pd.DataFrame, output_dir: Path) -> None:
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    datasets = list(metric_savings["dataset"].drop_duplicates())

    fig, axes = plt.subplots(2, 3, figsize=(12, 7.0))
    axes = axes.ravel()
    for ax, dataset in zip(axes, datasets):
        subset = metric_savings[metric_savings["dataset"].eq(dataset)]
        ax.scatter(subset["ppsr"], subset["total_saved_annotations"], s=24, color="#4C78A8", alpha=0.85)
        ax.set_title(dataset, fontsize=10)
        ax.set_xlabel("PPSR")
        ax.set_ylabel("Total annotations saved")
        ax.grid(alpha=0.25)
    for ax in axes[len(datasets):]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(figures_dir / "ppsr_vs_total_saved_all_datasets.png", dpi=220)
    fig.savefig(figures_dir / "ppsr_vs_total_saved_all_datasets.pdf")
    plt.close(fig)

    for dataset in datasets:
        subset = metric_savings[metric_savings["dataset"].eq(dataset)].copy()
        tag = dataset.replace(" ", "_").replace("-", "_")
        fig, ax = plt.subplots(figsize=(5.0, 3.8))
        ax.scatter(subset["ppsr"], subset["total_saved_annotations"], s=28, color="#4C78A8", alpha=0.85)
        ax.set_title(dataset)
        ax.set_xlabel("PPSR")
        ax.set_ylabel("Total annotations saved")
        ax.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(figures_dir / f"ppsr_vs_total_saved_{tag}.png", dpi=220)
        fig.savefig(figures_dir / f"ppsr_vs_total_saved_{tag}.pdf")
        plt.close(fig)


def write_outputs(
    output_dir: Path,
    pair_metric: pd.DataFrame,
    metric_savings: pd.DataFrame,
    dataset_correlation: pd.DataFrame,
    dataset_meta: pd.DataFrame,
    runtime_estimate: dict[str, Any],
    config: dict[str, Any],
    pair_metric_corr: pd.DataFrame | None = None,
    write_pair_metric: bool = True,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    if write_pair_metric:
        pair_metric.to_csv(output_dir / "pair_metric_thresholds.csv", index=False)
    metric_savings.to_csv(output_dir / "metric_savings.csv", index=False)
    dataset_correlation.to_csv(output_dir / "dataset_correlation.csv", index=False)
    if pair_metric_corr is not None:
        pair_metric_corr.to_csv(output_dir / "pair_metric_saved_vs_corr_sq.csv", index=False)
    dataset_meta.to_csv(output_dir / "dataset_meta.csv", index=False)
    (output_dir / "runtime_estimate.json").write_text(json.dumps(runtime_estimate, indent=2), encoding="utf-8")
    (output_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    plot_outputs(metric_savings, output_dir)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    runtime_estimate: dict[str, Any] = {}
    if args.postprocess_existing:
        threshold_path = args.output_dir / "pair_metric_thresholds.csv"
        if not threshold_path.exists():
            raise FileNotFoundError(f"Missing threshold file for postprocessing: {threshold_path}")
        pair_metric = pd.read_csv(threshold_path)
        dataset_meta_path = args.output_dir / "dataset_meta.csv"
        if dataset_meta_path.exists():
            dataset_meta = pd.read_csv(dataset_meta_path)
        else:
            _tasks, dataset_meta = build_tasks(
                config_path=args.config,
                dataset_dir=args.dataset_dir,
                dataset_specs=args.datasets,
                max_metrics=args.max_metrics,
                num_trials=args.num_trials,
                alpha=args.alpha,
                target_power=args.target_power,
                seed=args.seed,
                brent_xtol=args.brent_xtol,
                min_labeled=args.min_labeled,
                max_pairs_per_dataset=args.max_pairs_per_dataset,
            )
        filtered_ppsr = recompute_filtered_ppsr(
            args.config,
            args.dataset_dir,
            args.datasets,
            pair_metric,
            args.max_metrics,
        )
        metric_savings, dataset_correlation = summarize_savings(pair_metric, filtered_ppsr)
        pair_metric_corr = pair_level_saved_corr_rows(
            args.config,
            args.dataset_dir,
            args.datasets,
            pair_metric,
            args.max_metrics,
        )
        config_path = args.output_dir / "config.json"
        if config_path.exists():
            config = json.loads(config_path.read_text(encoding="utf-8"))
        else:
            config = {}
        config.pop("pair_level_summary", None)
        config.update(
            {
                "datasets": args.datasets,
                "config": str(args.config),
                "dataset_dir": str(args.dataset_dir),
                "max_metrics": args.max_metrics,
                "postprocess_existing": True,
                "ppsr_definition": "mean_corr_sq_over_human_reachable_pairs",
                "human_pair_filter": 'human_threshold_status != "does_not_reach_target"',
            }
        )
        runtime_path = args.output_dir / "runtime_estimate.json"
        if runtime_path.exists():
            runtime_estimate = json.loads(runtime_path.read_text(encoding="utf-8"))
        write_outputs(
            args.output_dir,
            pair_metric,
            metric_savings,
            dataset_correlation,
            dataset_meta,
            runtime_estimate,
            config,
            pair_metric_corr,
            write_pair_metric=False,
        )
        print(dataset_correlation.to_string(index=False))
        return

    tasks, dataset_meta = build_tasks(
        config_path=args.config,
        dataset_dir=args.dataset_dir,
        dataset_specs=args.datasets,
        max_metrics=args.max_metrics,
        num_trials=args.num_trials,
        alpha=args.alpha,
        target_power=args.target_power,
        seed=args.seed,
        brent_xtol=args.brent_xtol,
        min_labeled=args.min_labeled,
        max_pairs_per_dataset=args.max_pairs_per_dataset,
    )
    if not args.skip_estimate:
        runtime_estimate = estimate_runtime(
            all_tasks=tasks,
            config_path=args.config,
            dataset_dir=args.dataset_dir,
            dataset_specs=args.datasets,
            max_metrics=args.max_metrics,
            pilot_trials=args.pilot_trials,
            pilot_pairs_per_dataset=args.pilot_pairs_per_dataset,
            pilot_max_metrics=args.pilot_max_metrics,
            alpha=args.alpha,
            target_power=args.target_power,
            seed=args.seed,
            brent_xtol=args.brent_xtol,
            min_labeled=args.min_labeled,
            num_workers=args.num_workers,
        )
        (args.output_dir / "runtime_estimate.json").write_text(
            json.dumps(runtime_estimate, indent=2),
            encoding="utf-8",
        )
        print(json.dumps(runtime_estimate, indent=2), flush=True)
    if args.estimate_only:
        return

    start = time.perf_counter()
    pair_metric = run_tasks(tasks, args.num_workers, "system pairs")
    elapsed = time.perf_counter() - start
    runtime_estimate["actual_full_run_seconds"] = elapsed
    filtered_ppsr = recompute_filtered_ppsr(
        args.config,
        args.dataset_dir,
        args.datasets,
        pair_metric,
        args.max_metrics,
    )
    metric_savings, dataset_correlation = summarize_savings(pair_metric, filtered_ppsr)
    pair_metric_corr = pair_level_saved_corr_rows(
        args.config,
        args.dataset_dir,
        args.datasets,
        pair_metric,
        args.max_metrics,
    )
    write_outputs(
        args.output_dir,
        pair_metric,
        metric_savings,
        dataset_correlation,
        dataset_meta,
        runtime_estimate,
        {
            "datasets": args.datasets,
            "config": str(args.config),
            "dataset_dir": str(args.dataset_dir),
            "num_trials": args.num_trials,
            "alpha": args.alpha,
            "target_power": args.target_power,
            "seed": args.seed,
            "brent_xtol": args.brent_xtol,
            "min_labeled": args.min_labeled,
            "max_metrics": args.max_metrics,
            "max_pairs_per_dataset": args.max_pairs_per_dataset,
            "num_workers": args.num_workers,
            "ppsr_scores": str(args.ppsr_scores),
            "ppsr_definition": "mean_corr_sq_over_human_reachable_pairs",
            "human_pair_filter": 'human_threshold_status != "does_not_reach_target"',
            "human_score_columns_excluded_from_metrics": sorted(HUMAN_SEG_COLS),
            "dataset_specific_excluded_metrics": load_config(args.config).excluded_metric_columns,
        },
        pair_metric_corr,
    )
    print(dataset_correlation.to_string(index=False))
    print(f"total elapsed: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
