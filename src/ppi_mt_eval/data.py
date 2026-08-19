from __future__ import annotations

import subprocess
import sys
from itertools import combinations
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .config import DatasetSpec, PaperConfig


NA_VALUES = ("", "NA", "None", "none", "nan", "NaN", "N/A", "null", "NULL")
HUMAN_SEG_COLS = {
    "mqm:seg",
    "da-sqm:seg",
    "esa:seg",
    "esa-merged:seg",
    "esa-human1:seg",
    "esa-human2:seg",
    "wmt-appraise:seg",
    "wmt-appraise-z:seg",
    "wmt:seg",
    "wmt-z:seg",
}


def is_reference_system(system_name: object) -> bool:
    return str(system_name).lower().startswith("ref")


def export_scores(spec: DatasetSpec, dataset_dir: Path, overwrite: bool = False) -> Path:
    dataset_dir.mkdir(parents=True, exist_ok=True)
    out = spec.path(dataset_dir)
    if out.exists() and not overwrite:
        return out
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "mt_metrics_eval.mtme",
            "-t",
            spec.test_set,
            "-l",
            spec.language_pair,
            "--scores",
        ],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or proc.stdout.strip())
    out.write_text(proc.stdout, encoding="utf-8")
    return out


def read_scores(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t", na_values=NA_VALUES, keep_default_na=True)


def filtered_metric_columns(
    df: pd.DataFrame,
    human_col: str,
    spec: DatasetSpec,
    config: PaperConfig,
) -> list[str]:
    excluded = set(config.excluded_metric_columns.get(spec.key, ()))
    candidates = [
        col
        for col in df.columns
        if col.endswith(":seg")
        and col not in HUMAN_SEG_COLS
        and col != human_col
        and col not in excluded
        and "sentinel-" not in col
    ]
    candidate_set = set(candidates)
    kept: list[str] = []
    for col in candidates:
        if "-refB:seg" in col and col.replace("-refB:seg", "-refA:seg") in candidate_set:
            continue
        if df[col].notna().any():
            kept.append(col)
    return kept


def wide_tables(
    df: pd.DataFrame,
    human_col: str,
    metric_cols: Iterable[str],
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame], list[str]]:
    df = df.loc[~df["system-name"].map(is_reference_system)].copy()
    human = df.pivot_table(
        index="seg-id", columns="system-name", values=human_col, aggfunc="mean"
    )
    metric_tables = {
        metric: df.pivot_table(
            index="seg-id", columns="system-name", values=metric, aggfunc="mean"
        )
        for metric in metric_cols
    }
    systems = [
        system
        for system in human.columns
        if human[system].notna().any()
        and all(system in table.columns and table[system].notna().any() for table in metric_tables.values())
    ]
    systems = list(human[systems].mean(axis=0).sort_values(ascending=False).index)
    return human[systems], {m: t[systems] for m, t in metric_tables.items()}, systems


def pair_matrices(
    human_table: pd.DataFrame,
    metric_table: pd.DataFrame,
    systems: list[str],
) -> tuple[list[dict[str, object]], np.ndarray, np.ndarray]:
    common_index = human_table.dropna().index.intersection(metric_table.dropna().index)
    human_array = human_table.loc[common_index, systems].to_numpy(dtype=float).T
    metric_array = metric_table.loc[common_index, systems].to_numpy(dtype=float).T
    pair_specs: list[dict[str, object]] = []
    human_pairs: list[np.ndarray] = []
    metric_pairs: list[np.ndarray] = []
    for ia, ib in combinations(range(len(systems)), 2):
        hdiff = human_array[ia] - human_array[ib]
        fdiff = metric_array[ia] - metric_array[ib]
        pair_specs.append(
            {
                "system_a": systems[ia],
                "system_b": systems[ib],
                "true_effect": float(np.mean(hdiff)),
                "aligned_segments": int(hdiff.size),
            }
        )
        human_pairs.append(hdiff)
        metric_pairs.append(fdiff)
    return pair_specs, np.vstack(human_pairs), np.vstack(metric_pairs)


def all_system_matrix(human_table: pd.DataFrame, systems: list[str]) -> np.ndarray:
    common_index = human_table.dropna().index
    return human_table.loc[common_index, systems].to_numpy(dtype=float).T


def dataset_summary_rows(config: PaperConfig, dataset_dir: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for spec in config.datasets:
        path = spec.path(dataset_dir)
        if not path.exists():
            rows.append(
                {
                    "dataset": spec.label,
                    "test_set": spec.test_set,
                    "language_pair": spec.language_pair,
                    "human_type": spec.human_type,
                    "human_col": spec.human_col,
                    "num_inputs": "",
                    "num_systems": "",
                    "num_metrics": "",
                    "path": str(path),
                    "notes": "missing TSV",
                }
            )
            continue
        df = read_scores(path)
        metrics = filtered_metric_columns(df, spec.human_col, spec, config)
        human, _, systems = wide_tables(df, spec.human_col, [])
        aligned = int((human.dropna().shape[0]))
        rows.append(
            {
                "dataset": spec.label,
                "test_set": spec.test_set,
                "language_pair": spec.language_pair,
                "human_type": spec.human_type,
                "human_col": spec.human_col,
                "num_inputs": aligned,
                "num_systems": len(systems),
                "num_metrics": len(metrics),
                "path": str(path),
                "notes": "",
            }
        )
    return rows
