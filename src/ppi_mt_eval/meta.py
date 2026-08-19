from __future__ import annotations

from itertools import combinations

import numpy as np
from scipy import stats

from .variance import corr


def system_means(matrix: np.ndarray) -> np.ndarray:
    return np.nanmean(matrix, axis=1)


def pearson_system(y: np.ndarray, f: np.ndarray) -> float:
    return corr(system_means(y), system_means(f))


def spearman_system(y: np.ndarray, f: np.ndarray) -> float:
    val = stats.spearmanr(system_means(y), system_means(f), nan_policy="omit").statistic
    return float(val) if np.isfinite(val) else float("nan")


def kendall_system(y: np.ndarray, f: np.ndarray) -> float:
    val = stats.kendalltau(system_means(y), system_means(f), variant="b").statistic
    return float(val) if np.isfinite(val) else float("nan")


def ppsr(y: np.ndarray, f: np.ndarray) -> float:
    vals: list[float] = []
    for ia, ib in combinations(range(y.shape[0]), 2):
        r = corr(y[ia] - y[ib], f[ia] - f[ib])
        if np.isfinite(r):
            vals.append(r * r)
    return float(np.mean(vals)) if vals else float("nan")


def _finite_mean(values: list[float]) -> float:
    finite = [value for value in values if np.isfinite(value)]
    return float(np.mean(finite)) if finite else float("nan")


def group_by_item_r(y: np.ndarray, f: np.ndarray) -> float:
    vals = [corr(y[:, item], f[:, item]) for item in range(y.shape[1])]
    return _finite_mean(vals)


def no_grouping_r(y: np.ndarray, f: np.ndarray) -> float:
    return corr(y.ravel(), f.ravel())


def group_by_system_r(y: np.ndarray, f: np.ndarray) -> float:
    vals = [corr(y[system], f[system]) for system in range(y.shape[0])]
    return _finite_mean(vals)


def pdp(y: np.ndarray, f: np.ndarray) -> float:
    human_diffs = []
    metric_diffs = []
    for ia, ib in combinations(range(y.shape[0]), 2):
        human_diffs.append(y[ia] - y[ib])
        metric_diffs.append(f[ia] - f[ib])
    if not human_diffs:
        return float("nan")
    return corr(np.concatenate(human_diffs), np.concatenate(metric_diffs))


def pairwise_p_values_from_signs(scores: np.ndarray, signs: np.ndarray) -> np.ndarray:
    scores32 = scores.astype(np.float32, copy=False)
    partial = signs @ scores32.T
    sys_scores = np.sum(scores32, axis=1)
    num_systems = scores.shape[0]
    p_vals = np.full((num_systems, num_systems), np.nan, dtype=np.float64)
    for i in range(num_systems):
        for j in range(i + 1, num_systems):
            p_vals[i, j] = float(
                np.mean((partial[:, i] - partial[:, j]) >= (sys_scores[i] - sys_scores[j]))
            )
    return p_vals


def one_minus_pce(human_pvals: np.ndarray, metric_pvals: np.ndarray) -> float:
    idx = np.triu_indices(human_pvals.shape[0], 1)
    return float(1.0 - np.mean(np.abs(human_pvals[idx] - metric_pvals[idx])))


def spa_proxy(
    y: np.ndarray,
    f: np.ndarray,
    human_pvals: np.ndarray | None = None,
    signs: np.ndarray | None = None,
) -> float:
    if human_pvals is None or signs is None:
        raise ValueError("SPA requires human pairwise p-values and a sign matrix")
    return one_minus_pce(human_pvals, pairwise_p_values_from_signs(f, signs))


META_METRICS = {
    "input_r": group_by_item_r,
    "global_r": no_grouping_r,
    "system_r": group_by_system_r,
    "pdp": pdp,
    "pearson": pearson_system,
    "spearman": spearman_system,
    "kendall": kendall_system,
    "spa": spa_proxy,
    "ppsr": ppsr,
}


def dense_ranks_desc(values: dict[str, float]) -> dict[str, int]:
    finite = {k: v for k, v in values.items() if np.isfinite(v)}
    sorted_vals = sorted(set(finite.values()), reverse=True)
    return {k: sorted_vals.index(v) + 1 for k, v in finite.items()}
