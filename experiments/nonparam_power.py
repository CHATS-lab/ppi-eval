#!/usr/bin/env python3
"""Power analysis for paired z-tests and paired permutation tests.

This script compares human-only and prediction-powered paired tests on the same
Monte Carlo subsamples. It is optimized for the permutation tests by batching
system pairs with identical aligned segment sets and vectorizing the B sign-flip
statistics across all pairs in each batch.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from itertools import combinations
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

from ppi_mt_eval.intervals import (
    human_perm_reject as shared_human_perm_reject,
    human_z_reject as shared_human_z_reject,
    human_z_valid,
    ppi_perm_reject as shared_ppi_perm_reject,
    ppi_z_reject as shared_ppi_z_reject,
    ppi_z_valid,
)

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover - tqdm is expected but not required.
    tqdm = None


DEFAULT_DATASETS = (
    ("wmt22", "en-de"),
    ("wmt22", "en-ru"),
    ("wmt22", "zh-en"),
    ("wmt23", "en-zh"),
    ("wmt23", "ja-en"),
    ("wmt24", "cs-uk"),
)

REPRESENTATIVE_METRIC_BY_TESTSET = {
    "wmt22": "metricx_xxl_MQM_2020-refA:seg",
    "wmt23": "MetricX-23-refA:seg",
    "wmt24": "MetricX-24-refA:seg",
}

NA_VALUES = ("", "NA", "None", "none", "nan", "NaN", "N/A", "null", "NULL")
DEFAULT_SAMPLE_SIZES = (
    (20, 800),
    (40, 800),
    (60, 800),
    (80, 800),
    (100, 800),
    (120, 800),
    (140, 800),
    (160, 800),
    (180, 800),
    (200, 800),
)
DEFAULT_NUM_TRIALS = 1000
DEFAULT_NUM_PERMUTATIONS = 1000
DEFAULT_SEED = 20260515

METHOD_LABELS = {
    "human_z": "Human Z",
    "human_perm": "Human Perm.",
    "ppi_z": "PPI Z",
    "ppi_perm": "PPI Perm.",
}
METHOD_COLORS = {
    "human_z": "#4C78A8",
    "human_perm": "#72B7B2",
    "ppi_z": "#54A24B",
    "ppi_perm": "#E45756",
}


@dataclass(frozen=True)
class DatasetConfig:
    test_set: str
    language_pair: str
    path: Path
    human_col: str
    metric_col: str
    dataset_index: int

    @property
    def label(self) -> str:
        return f"{self.test_set} {self.language_pair}"

    @property
    def tag(self) -> str:
        return f"{self.test_set}_{self.language_pair}".replace("-", "_")


@dataclass
class PairBatch:
    segment_ids: tuple
    human_diff: np.ndarray
    metric_diff: np.ndarray
    metadata: list[dict]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-summary", type=Path, default=Path("datasets/dataset_summary.csv"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/nonparam_power"),
    )
    parser.add_argument(
        "--datasets",
        nargs="*",
        default=[f"{test_set}:{lp}" for test_set, lp in DEFAULT_DATASETS],
    )
    parser.add_argument(
        "--sample-sizes",
        nargs="+",
        default=[f"{l}:{u}" for l, u in DEFAULT_SAMPLE_SIZES],
        help="Sample sizes as L:U, e.g. 80:800.",
    )
    parser.add_argument("--num-trials", type=int, default=DEFAULT_NUM_TRIALS)
    parser.add_argument("--num-permutations", type=int, default=DEFAULT_NUM_PERMUTATIONS)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="Regenerate figures from existing CSV outputs without rerunning simulations.",
    )
    return parser.parse_args()


class NullProgress:
    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def update(self, n: int = 1) -> None:
        return None

    def close(self) -> None:
        return None


def progress(iterable=None, **kwargs):
    if tqdm is None:
        return iterable if iterable is not None else NullProgress()
    if iterable is None:
        return tqdm(**kwargs)
    return tqdm(iterable, **kwargs)


def parse_sample_sizes(specs: list[str]) -> tuple[tuple[int, int], ...]:
    sample_sizes = []
    for spec in specs:
        if ":" not in spec:
            raise ValueError(f"Sample size must have form L:U, got {spec!r}")
        labeled, unlabeled = (int(part) for part in spec.split(":", 1))
        if labeled < 2 or unlabeled < 2:
            raise ValueError(f"Both L and U must be at least 2, got {spec!r}")
        sample_sizes.append((labeled, unlabeled))
    return tuple(sample_sizes)


def is_reference_system(system_name: object) -> bool:
    return str(system_name).lower().startswith("ref")


def run_batch_trials(
    batch: PairBatch,
    labeled_size: int,
    unlabeled_size: int,
    num_trials: int,
    num_permutations: int,
    alpha: float,
    rng: np.random.Generator,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    sample_size = labeled_size + unlabeled_size
    num_pairs, aligned_size = batch.human_diff.shape
    rejections = {method: np.zeros(num_pairs, dtype=np.int32) for method in METHOD_LABELS}
    valid_trials = {method: np.zeros(num_pairs, dtype=np.int32) for method in METHOD_LABELS}

    for _ in range(num_trials):
        sampled_idx = rng.choice(aligned_size, size=sample_size, replace=False)
        labeled_idx = sampled_idx[:labeled_size]
        unlabeled_idx = sampled_idx[labeled_size:]

        human_labeled = batch.human_diff[:, labeled_idx]
        metric_labeled = batch.metric_diff[:, labeled_idx]
        metric_unlabeled = batch.metric_diff[:, unlabeled_idx]

        human_z_reject = shared_human_z_reject(human_labeled, alpha)
        human_z_valid_mask = human_z_valid(human_labeled)
        ppi_z_reject = shared_ppi_z_reject(human_labeled, metric_labeled, metric_unlabeled, alpha)
        ppi_z_valid_mask = ppi_z_valid(human_labeled, metric_labeled, metric_unlabeled)

        valid_trials["human_z"] += human_z_valid_mask
        rejections["human_z"] += human_z_reject & human_z_valid_mask
        valid_trials["ppi_z"] += ppi_z_valid_mask
        rejections["ppi_z"] += ppi_z_reject & ppi_z_valid_mask

        signs = rng.choice(
            np.array([-1.0, 1.0], dtype=np.float64),
            size=(num_permutations, sample_size),
            replace=True,
        )
        signs_labeled = signs[:, :labeled_size]
        signs_unlabeled = signs[:, labeled_size:]

        human_perm_reject = shared_human_perm_reject(
            human_labeled,
            signs_labeled,
            alpha,
            strict=True,
        )
        human_perm_valid = np.isfinite(np.mean(human_labeled, axis=1))
        valid_trials["human_perm"] += human_perm_valid
        rejections["human_perm"] += human_perm_reject & human_perm_valid

        ppi_perm_reject = shared_ppi_perm_reject(
            human_labeled,
            metric_labeled,
            metric_unlabeled,
            signs_labeled,
            signs_unlabeled,
            alpha,
            strict=True,
        )
        ppi_perm_valid = ppi_z_valid(human_labeled, metric_labeled, metric_unlabeled)
        valid_trials["ppi_perm"] += ppi_perm_valid
        rejections["ppi_perm"] += ppi_perm_reject & ppi_perm_valid

    return {method: (rejections[method], valid_trials[method]) for method in METHOD_LABELS}


def load_dataset_configs(summary_path: Path, dataset_specs: list[str]) -> list[DatasetConfig]:
    summary = pd.read_csv(summary_path)
    configs: list[DatasetConfig] = []
    for dataset_index, spec in enumerate(dataset_specs):
        if ":" not in spec:
            raise ValueError(f"Dataset spec must have form test_set:language_pair, got {spec!r}")
        test_set, language_pair = spec.split(":", 1)
        match = summary[
            (summary["test_set"] == test_set) & (summary["language_pair"] == language_pair)
        ]
        if match.empty:
            raise ValueError(f"No dataset_summary row for {test_set}:{language_pair}")
        row = match.iloc[0]
        if bool(row.get("missing_human_col", False)):
            raise ValueError(f"Dataset {test_set}:{language_pair} has no usable human column")
        if test_set not in REPRESENTATIVE_METRIC_BY_TESTSET:
            raise ValueError(f"No representative metric configured for {test_set}")
        configs.append(
            DatasetConfig(
                test_set=test_set,
                language_pair=language_pair,
                path=Path(row["export_path"] if "export_path" in row.index else row["path"]),
                human_col=row["human_score_col"] if "human_score_col" in row.index else row["human_col"],
                metric_col=REPRESENTATIVE_METRIC_BY_TESTSET[test_set],
                dataset_index=dataset_index,
            )
        )
    return configs


def load_wide_tables(config: DatasetConfig) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    df = pd.read_csv(config.path, sep="\t", na_values=NA_VALUES)
    required = {"system-name", "seg-id", config.human_col, config.metric_col}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"{config.path} is missing columns: {sorted(missing)}")
    df = df[~df["system-name"].map(is_reference_system)].copy()
    human = df.pivot_table(
        index="seg-id",
        columns="system-name",
        values=config.human_col,
        aggfunc="mean",
    )
    metric = df.pivot_table(
        index="seg-id",
        columns="system-name",
        values=config.metric_col,
        aggfunc="mean",
    )
    systems = [
        system
        for system in human.columns
        if human[system].notna().any() and system in metric.columns and metric[system].notna().any()
    ]
    systems = list(human[systems].mean(axis=0).sort_values(ascending=False).index)
    return human[systems], metric[systems], systems


def build_pair_batches(
    config: DatasetConfig,
    sample_sizes: tuple[tuple[int, int], ...],
) -> tuple[list[PairBatch], dict]:
    human_table, metric_table, systems = load_wide_tables(config)
    min_required = max(labeled + unlabeled for labeled, unlabeled in sample_sizes)
    grouped: dict[tuple, list[tuple[np.ndarray, np.ndarray, dict]]] = {}
    skipped_pairs = 0

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

        human_diff = (aligned["human_a"] - aligned["human_b"]).to_numpy(dtype=np.float64)
        metric_diff = (aligned["metric_a"] - aligned["metric_b"]).to_numpy(dtype=np.float64)
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

        segment_ids = tuple(aligned.index.tolist())
        metadata = {
            "dataset": config.label,
            "test_set": config.test_set,
            "language_pair": config.language_pair,
            "human_col": config.human_col,
            "metric_col": config.metric_col,
            "system_a": ordered_a,
            "system_b": ordered_b,
            "true_effect": true_effect,
            "aligned_segments": int(aligned.shape[0]),
        }
        grouped.setdefault(segment_ids, []).append((human_diff, metric_diff, metadata))

    batches = []
    for segment_ids, entries in grouped.items():
        batches.append(
            PairBatch(
                segment_ids=segment_ids,
                human_diff=np.vstack([entry[0] for entry in entries]),
                metric_diff=np.vstack([entry[1] for entry in entries]),
                metadata=[entry[2] for entry in entries],
            )
        )

    meta = {
        "dataset": config.label,
        "test_set": config.test_set,
        "language_pair": config.language_pair,
        "human_col": config.human_col,
        "metric_col": config.metric_col,
        "num_systems": len(systems),
        "num_pair_batches": len(batches),
        "num_pairs": sum(len(batch.metadata) for batch in batches),
        "skipped_pairs": skipped_pairs,
    }
    return batches, meta


def run_dataset(
    config: DatasetConfig,
    sample_sizes: tuple[tuple[int, int], ...],
    num_trials: int,
    num_permutations: int,
    alpha: float,
    seed: int,
    show_progress: bool,
) -> tuple[list[dict], dict, list[dict]]:
    start = time.perf_counter()
    rng = np.random.default_rng(seed + config.dataset_index * 100_003)
    batches, meta = build_pair_batches(config, sample_sizes)
    rows: list[dict] = []
    runtime_rows: list[dict] = []
    progress_total = len(sample_sizes) * len(batches)

    size_iter = progress(
        sample_sizes,
        desc=f"{config.label}",
        leave=False,
        disable=not show_progress,
    )
    inner_iter = progress(
        total=progress_total,
        desc=f"{config.label} simulations",
        leave=False,
        disable=not show_progress,
    )
    with inner_iter as dataset_progress:
        for labeled_size, unlabeled_size in size_iter:
            config_start = time.perf_counter()
            accum: list[tuple[PairBatch, dict[str, tuple[np.ndarray, np.ndarray]]]] = []
            for batch in batches:
                if batch.human_diff.shape[1] < labeled_size + unlabeled_size:
                    dataset_progress.update(1)
                    continue
                trial_results = run_batch_trials(
                    batch=batch,
                    labeled_size=labeled_size,
                    unlabeled_size=unlabeled_size,
                    num_trials=num_trials,
                    num_permutations=num_permutations,
                    alpha=alpha,
                    rng=rng,
                )
                accum.append((batch, trial_results))
                dataset_progress.update(1)

            for batch, trial_results in accum:
                for pair_index, metadata in enumerate(batch.metadata):
                    for method, (num_rejections, num_valid_trials) in trial_results.items():
                        valid = int(num_valid_trials[pair_index])
                        reject = int(num_rejections[pair_index])
                        row = dict(metadata)
                        row.update(
                            {
                                "L": labeled_size,
                                "U": unlabeled_size,
                                "method": method,
                                "method_label": METHOD_LABELS[method],
                                "num_trials": num_trials,
                                "num_permutations": num_permutations
                                if "perm" in method
                                else 0,
                                "valid_trials": valid,
                                "num_rejections": reject,
                                "power": reject / valid if valid > 0 else np.nan,
                            }
                        )
                        rows.append(row)

            runtime_rows.append(
                {
                    "dataset": config.label,
                    "test_set": config.test_set,
                    "language_pair": config.language_pair,
                    "L": labeled_size,
                    "U": unlabeled_size,
                    "elapsed_seconds": time.perf_counter() - config_start,
                }
            )

    meta["elapsed_seconds"] = time.perf_counter() - start
    return rows, meta, runtime_rows


def summarize_power(pairwise: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (dataset, labeled_size, method), group in pairwise.groupby(["dataset", "L", "method"], sort=False):
        finite_power = group["power"].replace([np.inf, -np.inf], np.nan).dropna()
        rows.append(
            {
                "dataset": dataset,
                "test_set": group["test_set"].iloc[0],
                "language_pair": group["language_pair"].iloc[0],
                "L": int(labeled_size),
                "U": int(group["U"].iloc[0]),
                "method": method,
                "method_label": METHOD_LABELS[method],
                "num_pairs": int(finite_power.shape[0]),
                "mean_power": float(finite_power.mean()),
                "median_power": float(finite_power.median()),
                "min_power": float(finite_power.min()),
                "max_power": float(finite_power.max()),
            }
        )
    return pd.DataFrame(rows)


def ordered_dataset_labels(df: pd.DataFrame) -> list[str]:
    configured_order = [f"{test_set} {language_pair}" for test_set, language_pair in DEFAULT_DATASETS]
    present = set(df["dataset"].dropna().unique())
    ordered = [dataset for dataset in configured_order if dataset in present]
    extras = sorted(present.difference(ordered))
    return ordered + extras


def save_figure(fig: plt.Figure, figures_dir: Path, filename: str) -> None:
    fig.savefig(figures_dir / f"{filename}.png", dpi=200)
    fig.savefig(figures_dir / f"{filename}.pdf")


def add_shared_legend(fig: plt.Figure, axes: np.ndarray, ncol: int) -> None:
    for ax in axes:
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            fig.legend(
                handles,
                labels,
                loc="lower center",
                ncol=ncol,
                fontsize=8,
                frameon=False,
                bbox_to_anchor=(0.5, 0.01),
            )
            return


def dataset_labeled_grid(
    dataset: str,
    labeled_sizes: list[int],
    wmt_three_by_three: bool,
) -> tuple[plt.Figure, np.ndarray, list[int], tuple[float, float, float, float]]:
    if wmt_three_by_three and dataset.startswith("wmt") and len(labeled_sizes) > 9:
        fig, axes = plt.subplots(3, 3, figsize=(12.0, 8.7), sharey=True)
        return fig, axes.ravel(), labeled_sizes[:9], (0, 0.06, 1, 0.94)
    fig, axes = plt.subplots(1, len(labeled_sizes), figsize=(11.0, 3.4), sharey=True)
    if len(labeled_sizes) == 1:
        axes = np.array([axes])
    return fig, np.ravel(axes), labeled_sizes, (0, 0.12, 1, 0.9)


def plot_power_panel(
    ax: plt.Axes,
    subset_l: pd.DataFrame,
    methods: tuple[str, ...],
    title: str,
    marker_size: float,
    line_width: float,
) -> None:
    for method in methods:
        subset = subset_l[subset_l["method"] == method].sort_values("true_effect")
        ax.plot(
            subset["true_effect"],
            subset["power"],
            marker="o",
            markersize=marker_size,
            linewidth=line_width,
            label=METHOD_LABELS[method],
            color=METHOD_COLORS[method],
            alpha=0.95,
        )
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("True human effect")
    ax.set_ylim(-0.03, 1.03)
    ax.grid(alpha=0.25)


def plot_dataset_curves(pairwise: pd.DataFrame, output_dir: Path) -> None:
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    for dataset in ordered_dataset_labels(pairwise):
        dataset_df = pairwise[pairwise["dataset"] == dataset]
        labeled_sizes = sorted(dataset_df["L"].unique())
        fig, axes, plotted_sizes, layout_rect = dataset_labeled_grid(
            dataset, labeled_sizes, wmt_three_by_three=True
        )
        for ax, labeled_size in zip(axes, plotted_sizes):
            subset_l = dataset_df[dataset_df["L"] == labeled_size]
            plot_power_panel(
                ax,
                subset_l,
                tuple(METHOD_LABELS),
                f"L={labeled_size}, U={int(subset_l['U'].iloc[0])}",
                marker_size=2.1,
                line_width=1.05,
            )
        for ax in axes[len(plotted_sizes) :]:
            ax.axis("off")
        axes[0].set_ylabel("Empirical power")
        add_shared_legend(fig, axes, ncol=len(METHOD_LABELS))
        fig.suptitle(f"{dataset}: z-test vs. permutation-test power", fontsize=12)
        fig.tight_layout(rect=layout_rect)
        tag = dataset.replace(" ", "_").replace("-", "_")
        save_figure(fig, figures_dir, f"nonparam_power_curves_{tag}")
        plt.close(fig)


def plot_combined_by_l(pairwise: pd.DataFrame, output_dir: Path) -> None:
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    datasets = ordered_dataset_labels(pairwise)
    for labeled_size in sorted(pairwise["L"].unique()):
        fig, axes = plt.subplots(2, 3, figsize=(12, 6.8), sharey=True)
        axes = axes.ravel()
        for ax, dataset in zip(axes, datasets):
            subset_l = pairwise[(pairwise["dataset"] == dataset) & (pairwise["L"] == labeled_size)]
            plot_power_panel(
                ax,
                subset_l,
                tuple(METHOD_LABELS),
                dataset,
                marker_size=1.9,
                line_width=1.0,
            )
        for ax in axes[len(datasets) :]:
            ax.axis("off")
        axes[0].set_ylabel("Empirical power")
        axes[3].set_ylabel("Empirical power")
        add_shared_legend(fig, axes, ncol=len(METHOD_LABELS))
        fig.suptitle(f"Non-parametric power comparison for L={labeled_size}, U=800", fontsize=13)
        fig.tight_layout(rect=(0, 0.07, 1, 0.95))
        save_figure(fig, figures_dir, f"nonparam_power_curves_L{labeled_size}")
        plt.close(fig)


def plot_perm_only_dataset_curves(
    pairwise: pd.DataFrame,
    output_dir: Path,
) -> None:
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    methods = ("human_perm", "ppi_perm")
    for dataset in ordered_dataset_labels(pairwise):
        dataset_df = pairwise[pairwise["dataset"] == dataset]
        labeled_sizes = sorted(dataset_df["L"].unique())
        fig, axes, plotted_sizes, layout_rect = dataset_labeled_grid(
            dataset, labeled_sizes, wmt_three_by_three=True
        )
        for ax, labeled_size in zip(axes, plotted_sizes):
            subset_l = dataset_df[dataset_df["L"] == labeled_size]
            plot_power_panel(
                ax,
                subset_l,
                methods,
                f"L={labeled_size}, U={int(subset_l['U'].iloc[0])}",
                marker_size=2.2,
                line_width=1.25,
            )
        for ax in axes[len(plotted_sizes) :]:
            ax.axis("off")
        axes[0].set_ylabel("Empirical power")
        add_shared_legend(fig, axes, ncol=len(methods))
        fig.suptitle(f"{dataset}: paired permutation-test power", fontsize=12)
        fig.tight_layout(rect=layout_rect)
        tag = dataset.replace(" ", "_").replace("-", "_")
        save_figure(fig, figures_dir, f"perm_only_power_curves_{tag}")
        plt.close(fig)


def plot_perm_only_combined_by_l(
    pairwise: pd.DataFrame,
    output_dir: Path,
) -> None:
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    datasets = ordered_dataset_labels(pairwise)
    methods = ("human_perm", "ppi_perm")
    for labeled_size in sorted(pairwise["L"].unique()):
        fig, axes = plt.subplots(2, 3, figsize=(12, 6.6), sharey=True)
        axes = axes.ravel()
        for ax, dataset in zip(axes, datasets):
            subset_l = pairwise[(pairwise["dataset"] == dataset) & (pairwise["L"] == labeled_size)]
            plot_power_panel(
                ax,
                subset_l,
                methods,
                dataset,
                marker_size=2.0,
                line_width=1.15,
            )
        for ax in axes[len(datasets) :]:
            ax.axis("off")
        axes[0].set_ylabel("Empirical power")
        axes[3].set_ylabel("Empirical power")
        add_shared_legend(fig, axes, ncol=len(methods))
        fig.suptitle(f"Permutation-test power comparison for L={labeled_size}, U=800", fontsize=13)
        fig.tight_layout(rect=(0, 0.07, 1, 0.95))
        save_figure(fig, figures_dir, f"perm_only_power_curves_L{labeled_size}")
        plt.close(fig)


def plot_method_pair_dataset_curves(
    pairwise: pd.DataFrame,
    output_dir: Path,
    methods: tuple[str, str],
    prefix: str,
    title: str,
) -> None:
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    for dataset in ordered_dataset_labels(pairwise):
        dataset_df = pairwise[pairwise["dataset"] == dataset]
        labeled_sizes = sorted(dataset_df["L"].unique())
        fig, axes, plotted_sizes, layout_rect = dataset_labeled_grid(
            dataset,
            labeled_sizes,
            wmt_three_by_three=prefix in {"human_z_vs_perm", "ppi_z_vs_perm"},
        )
        for ax, labeled_size in zip(axes, plotted_sizes):
            subset_l = dataset_df[dataset_df["L"] == labeled_size]
            plot_power_panel(
                ax,
                subset_l,
                methods,
                f"L={labeled_size}, U={int(subset_l['U'].iloc[0])}",
                marker_size=2.2,
                line_width=1.25,
            )
        for ax in axes[len(plotted_sizes) :]:
            ax.axis("off")
        axes[0].set_ylabel("Empirical power")
        add_shared_legend(fig, axes, ncol=len(methods))
        fig.suptitle(f"{dataset}: {title}", fontsize=12)
        fig.tight_layout(rect=layout_rect)
        tag = dataset.replace(" ", "_").replace("-", "_")
        save_figure(fig, figures_dir, f"{prefix}_power_curves_{tag}")
        plt.close(fig)


def plot_method_pair_combined_by_l(
    pairwise: pd.DataFrame,
    output_dir: Path,
    methods: tuple[str, str],
    prefix: str,
    title: str,
) -> None:
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    datasets = ordered_dataset_labels(pairwise)
    for labeled_size in sorted(pairwise["L"].unique()):
        fig, axes = plt.subplots(2, 3, figsize=(12, 6.6), sharey=True)
        axes = axes.ravel()
        for ax, dataset in zip(axes, datasets):
            subset_l = pairwise[(pairwise["dataset"] == dataset) & (pairwise["L"] == labeled_size)]
            plot_power_panel(
                ax,
                subset_l,
                methods,
                dataset,
                marker_size=2.0,
                line_width=1.15,
            )
        for ax in axes[len(datasets) :]:
            ax.axis("off")
        axes[0].set_ylabel("Empirical power")
        axes[3].set_ylabel("Empirical power")
        add_shared_legend(fig, axes, ncol=len(methods))
        fig.suptitle(f"{title} for L={labeled_size}, U=800", fontsize=13)
        fig.tight_layout(rect=(0, 0.07, 1, 0.95))
        save_figure(fig, figures_dir, f"{prefix}_power_curves_L{labeled_size}")
        plt.close(fig)


def plot_focused_comparisons(
    pairwise: pd.DataFrame,
    output_dir: Path,
) -> None:
    comparisons = (
        (("human_perm", "ppi_perm"), "perm_only", "paired permutation-test power"),
        (("human_z", "human_perm"), "human_z_vs_perm", "human-only Z-test vs. permutation-test power"),
        (("ppi_z", "ppi_perm"), "ppi_z_vs_perm", "PPI Z-test vs. permutation-test power"),
    )
    for methods, prefix, title in comparisons:
        plot_method_pair_dataset_curves(
            pairwise=pairwise,
            output_dir=output_dir,
            methods=methods,
            prefix=prefix,
            title=title,
        )
        plot_method_pair_combined_by_l(
            pairwise=pairwise,
            output_dir=output_dir,
            methods=methods,
            prefix=prefix,
            title=title,
        )


def plot_aggregate_power_by_l(
    summary: pd.DataFrame,
    output_dir: Path,
) -> None:
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    comparisons = (
        (("human_z", "human_perm"), "Human-only", METHOD_COLORS["human_z"]),
        (("ppi_z", "ppi_perm"), "Prediction-powered", METHOD_COLORS["ppi_z"]),
    )
    datasets = ordered_dataset_labels(summary)
    fig, axes = plt.subplots(2, 3, figsize=(12, 6.7), sharex=True, sharey=True)
    axes = axes.ravel()
    for ax, dataset in zip(axes, datasets):
        subset_d = summary[summary["dataset"] == dataset]
        wide = subset_d.pivot(index="L", columns="method", values="mean_power").sort_index()
        for methods, label, color in comparisons:
            wide = subset_d.pivot(index="L", columns="method", values="mean_power").sort_index()
            gain = wide[methods[0]] - wide[methods[1]]
            ax.plot(
                gain.index,
                gain.values,
                marker="o",
                markersize=3.2,
                linewidth=1.7,
                label=label,
                color=color,
                alpha=0.95,
            )
        ax.axhline(0, color="#555555", linestyle="--", linewidth=1.0)
        ax.set_title(dataset, fontsize=10)
        ax.set_xlabel("Labeled examples (L)")
        ax.grid(alpha=0.25)
    for ax in axes[len(datasets) :]:
        ax.axis("off")
    axes[0].set_ylabel("Avg. power difference\n(Z - permutation)")
    axes[3].set_ylabel("Avg. power difference\n(Z - permutation)")
    add_shared_legend(fig, axes, ncol=len(comparisons))
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    filename = "z_vs_perm_power_gain_by_l"
    save_figure(fig, figures_dir, filename)
    plt.close(fig)

    for dataset in datasets:
        subset_d = summary[summary["dataset"] == dataset]
        wide = subset_d.pivot(index="L", columns="method", values="mean_power").sort_index()
        fig, ax = plt.subplots(figsize=(6.2, 4.0))
        for methods, label, color in comparisons:
            gain = wide[methods[0]] - wide[methods[1]]
            ax.plot(
                gain.index,
                gain.values,
                marker="o",
                markersize=3.5,
                linewidth=1.8,
                label=label,
                color=color,
                alpha=0.95,
            )
        ax.axhline(0, color="#555555", linestyle="--", linewidth=1.0)
        ax.set_title(dataset, fontsize=11)
        ax.set_xlabel("Labeled examples (L)")
        ax.set_ylabel("Avg. power difference\n(Z - permutation)")
        ax.grid(alpha=0.25)
        ax.legend(frameon=False)
        fig.tight_layout()
        tag = dataset.replace(" ", "_").replace("-", "_")
        save_figure(fig, figures_dir, f"{filename}_{tag}")
        plt.close(fig)

    single_comparisons = (
        ("human_z_vs_perm_power_gain_by_l", ("human_z", "human_perm"), "Human-only"),
        ("ppi_z_vs_perm_power_gain_by_l", ("ppi_z", "ppi_perm"), "Prediction-powered"),
    )
    for single_filename, methods, label in single_comparisons:
        fig, axes = plt.subplots(2, 3, figsize=(12, 6.7), sharex=True, sharey=True)
        axes = axes.ravel()
        for ax, dataset in zip(axes, datasets):
            subset_d = summary[summary["dataset"] == dataset]
            wide = subset_d.pivot(index="L", columns="method", values="mean_power").sort_index()
            gain = wide[methods[0]] - wide[methods[1]]
            ax.plot(
                gain.index,
                gain.values,
                marker="o",
                markersize=3.2,
                linewidth=1.7,
                label=label,
                color=METHOD_COLORS[methods[0]],
                alpha=0.95,
            )
            ax.axhline(0, color="#555555", linestyle="--", linewidth=1.0)
            ax.set_title(dataset, fontsize=10)
            ax.set_xlabel("Labeled examples (L)")
            ax.grid(alpha=0.25)
        for ax in axes[len(datasets) :]:
            ax.axis("off")
        axes[0].set_ylabel("Avg. power difference\n(Z - permutation)")
        axes[3].set_ylabel("Avg. power difference\n(Z - permutation)")
        add_shared_legend(fig, axes, ncol=1)
        fig.tight_layout(rect=(0, 0.07, 1, 1))
        save_figure(fig, figures_dir, single_filename)
        plt.close(fig)


def plot_mean_power_by_l(
    summary: pd.DataFrame,
    output_dir: Path,
    methods: tuple[str, str],
    filename: str,
    title: str,
) -> None:
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    datasets = ordered_dataset_labels(summary)
    fig, axes = plt.subplots(2, 3, figsize=(12, 6.7), sharex=True, sharey=True)
    axes = axes.ravel()
    for ax, dataset in zip(axes, datasets):
        subset_d = summary[summary["dataset"] == dataset]
        wide = subset_d.pivot(index="L", columns="method", values="mean_power").sort_index()
        for method in methods:
            ax.plot(
                wide.index,
                wide[method],
                marker="o",
                markersize=3.2,
                linewidth=1.7,
                label=METHOD_LABELS[method],
                color=METHOD_COLORS[method],
                alpha=0.95,
            )
        ax.set_title(dataset, fontsize=10)
        ax.set_xlabel("Labeled examples (L)")
        ax.set_ylim(-0.03, 1.03)
        ax.grid(alpha=0.25)
    for ax in axes[len(datasets) :]:
        ax.axis("off")
    axes[0].set_ylabel("Mean empirical power")
    axes[3].set_ylabel("Mean empirical power")
    add_shared_legend(fig, axes, ncol=len(methods))
    fig.suptitle(title, fontsize=13)
    fig.tight_layout(rect=(0, 0.07, 1, 0.95))
    save_figure(fig, figures_dir, filename)
    plt.close(fig)


def plot_mean_power_figures(
    summary: pd.DataFrame,
    output_dir: Path,
) -> None:
    plot_mean_power_by_l(
        summary=summary,
        output_dir=output_dir,
        methods=("human_z", "human_perm"),
        filename="human_z_vs_perm_mean_power_by_l",
        title="Mean human-only power by labeled sample size",
    )
    plot_mean_power_by_l(
        summary=summary,
        output_dir=output_dir,
        methods=("ppi_z", "ppi_perm"),
        filename="ppi_z_vs_perm_mean_power_by_l",
        title="Mean prediction-powered power by labeled sample size",
    )


def write_readme(output_dir: Path, summary: pd.DataFrame, runtime: pd.DataFrame) -> None:
    readme = [
        "# Non-Parametric Power Analysis",
        "",
        "This experiment compares paired z-tests and paired permutation tests for",
        "human-only and prediction-powered system comparisons. For each system pair,",
        "the full aligned paired sample is treated as the population and the system",
        "order is chosen so that the full-population human mean difference is positive.",
        "",
        "Each Monte Carlo trial samples `L + U` examples without replacement and uses",
        "the same split for all four tests.",
        "",
        "## Summary",
        "",
        summary.to_markdown(index=False),
        "",
        "## Runtime",
        "",
        runtime.to_markdown(index=False),
        "",
    ]
    (output_dir / "README.md").write_text("\n".join(readme), encoding="utf-8")


def main() -> None:
    args = parse_args()
    sample_sizes = parse_sample_sizes(args.sample_sizes)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.plot_only:
        pairwise = pd.read_csv(args.output_dir / "power_pairwise.csv")
        summary = pd.read_csv(args.output_dir / "power_summary.csv")
        plot_dataset_curves(pairwise, args.output_dir)
        plot_combined_by_l(pairwise, args.output_dir)
        plot_focused_comparisons(pairwise, args.output_dir)
        plot_aggregate_power_by_l(summary, args.output_dir)
        plot_mean_power_figures(summary, args.output_dir)
        print(f"regenerated figures from {args.output_dir / 'power_pairwise.csv'}")
        return

    configs = load_dataset_configs(args.dataset_summary, args.datasets)

    start = time.perf_counter()
    all_rows: list[dict] = []
    meta_rows: list[dict] = []
    runtime_rows: list[dict] = []

    if args.num_workers <= 1:
        for config in progress(configs, desc="datasets"):
            rows, meta, runtime = run_dataset(
                config=config,
                sample_sizes=sample_sizes,
                num_trials=args.num_trials,
                num_permutations=args.num_permutations,
                alpha=args.alpha,
                seed=args.seed,
                show_progress=True,
            )
            all_rows.extend(rows)
            meta_rows.append(meta)
            runtime_rows.extend(runtime)
    else:
        with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
            futures = {
                executor.submit(
                    run_dataset,
                    config,
                    sample_sizes,
                    args.num_trials,
                    args.num_permutations,
                    args.alpha,
                    args.seed,
                    False,
                ): config
                for config in configs
            }
            for future in progress(as_completed(futures), total=len(futures), desc="datasets"):
                config = futures[future]
                rows, meta, runtime = future.result()
                all_rows.extend(rows)
                meta_rows.append(meta)
                runtime_rows.extend(runtime)
                print(
                    f"completed {config.label}: {meta['num_pairs']} pairs in "
                    f"{meta['elapsed_seconds']:.1f}s",
                    flush=True,
                )

    pairwise = pd.DataFrame(all_rows)
    summary = summarize_power(pairwise)
    meta = pd.DataFrame(meta_rows).sort_values(["test_set", "language_pair"])
    runtime = pd.DataFrame(runtime_rows).sort_values(["test_set", "language_pair", "L"])
    total_elapsed = time.perf_counter() - start
    runtime.loc[len(runtime)] = {
        "dataset": "TOTAL",
        "test_set": "",
        "language_pair": "",
        "L": 0,
        "U": 0,
        "elapsed_seconds": total_elapsed,
    }

    pairwise.to_csv(args.output_dir / "power_pairwise.csv", index=False)
    summary.to_csv(args.output_dir / "power_summary.csv", index=False)
    meta.to_csv(args.output_dir / "dataset_meta.csv", index=False)
    runtime.to_csv(args.output_dir / "runtime_log.csv", index=False)
    (args.output_dir / "config.json").write_text(
        json.dumps(
            {
                "datasets": args.datasets,
                "representative_metric_by_testset": REPRESENTATIVE_METRIC_BY_TESTSET,
                "sample_sizes": [{"L": l, "U": u} for l, u in sample_sizes],
                "num_trials": args.num_trials,
                "num_permutations": args.num_permutations,
                "num_workers": args.num_workers,
                "alpha": args.alpha,
                "seed": args.seed,
                "methods": METHOD_LABELS,
                "ppi_permutation_variant": "metric_centered",
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    plot_dataset_curves(pairwise, args.output_dir)
    plot_combined_by_l(pairwise, args.output_dir)
    plot_focused_comparisons(pairwise, args.output_dir)
    plot_aggregate_power_by_l(summary, args.output_dir)
    plot_mean_power_figures(summary, args.output_dir)
    write_readme(args.output_dir, summary, runtime)
    print(summary.to_string(index=False))
    print(f"total elapsed: {total_elapsed:.1f}s")


if __name__ == "__main__":
    main()
