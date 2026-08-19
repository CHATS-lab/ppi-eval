from __future__ import annotations

from statistics import NormalDist

import numpy as np


def zcrit_two_sided(alpha: float = 0.05) -> float:
    return NormalDist().inv_cdf(1.0 - alpha / 2.0)


def zcrit_one_sided(alpha: float = 0.05) -> float:
    return NormalDist().inv_cdf(1.0 - alpha)


def human_z_intervals(
    values: np.ndarray,
    sample_idx: np.ndarray,
    true_effects: np.ndarray,
    alpha: float = 0.05,
) -> tuple[np.ndarray, np.ndarray]:
    sampled = values[:, sample_idx]
    means = sampled.mean(axis=1)
    vars_ = sampled.var(axis=1, ddof=1)
    widths = 2.0 * zcrit_two_sided(alpha) * np.sqrt(np.maximum(vars_, 0.0) / sampled.shape[1])
    lower = means - widths / 2.0
    upper = means + widths / 2.0
    return widths, ((lower <= true_effects) & (true_effects <= upper)).astype(float)


def ppi_intervals(
    human_pairs: np.ndarray,
    metric_pairs: np.ndarray,
    labeled_idx: np.ndarray,
    unlabeled_idx: np.ndarray,
    true_effects: np.ndarray,
    alpha: float = 0.05,
) -> tuple[np.ndarray, np.ndarray]:
    y_l = human_pairs[:, labeled_idx]
    f_l = metric_pairs[:, labeled_idx]
    f_u = metric_pairs[:, unlabeled_idx]
    l_size = y_l.shape[1]
    u_size = f_u.shape[1]

    y_mean = y_l.mean(axis=1)
    f_l_mean = f_l.mean(axis=1)
    var_f_l = f_l.var(axis=1, ddof=1)
    var_f_u = f_u.var(axis=1, ddof=1)
    cov = ((y_l - y_mean[:, None]) * (f_l - f_l_mean[:, None])).sum(axis=1) / (l_size - 1)
    denom = var_f_l + (l_size / u_size) * var_f_u
    lam = np.divide(cov, denom, out=np.zeros_like(cov), where=denom > 0)

    rectifier = y_l - lam[:, None] * f_l
    imputed = lam[:, None] * f_u
    point = rectifier.mean(axis=1) + imputed.mean(axis=1)
    variance = rectifier.var(axis=1, ddof=1) / l_size + imputed.var(axis=1, ddof=1) / u_size
    widths = 2.0 * zcrit_two_sided(alpha) * np.sqrt(np.maximum(variance, 0.0))
    lower = point - widths / 2.0
    upper = point + widths / 2.0
    return widths, ((lower <= true_effects) & (true_effects <= upper)).astype(float)


def ppi_point_and_var(
    y_l: np.ndarray,
    f_l: np.ndarray,
    f_u: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    l_size = y_l.shape[1]
    u_size = f_u.shape[1]
    y_mean = y_l.mean(axis=1)
    f_l_mean = f_l.mean(axis=1)
    var_f_l = f_l.var(axis=1, ddof=1)
    var_f_u = f_u.var(axis=1, ddof=1)
    cov = ((y_l - y_mean[:, None]) * (f_l - f_l_mean[:, None])).sum(axis=1) / (l_size - 1)
    denom = var_f_l + (l_size / u_size) * var_f_u
    lam = np.divide(cov, denom, out=np.zeros_like(cov), where=denom > 0)
    rectifier = y_l - lam[:, None] * f_l
    imputed = lam[:, None] * f_u
    point = rectifier.mean(axis=1) + imputed.mean(axis=1)
    variance = rectifier.var(axis=1, ddof=1) / l_size + imputed.var(axis=1, ddof=1) / u_size
    return point, variance


def ppi_z_point_and_var(
    y_l: np.ndarray,
    f_l: np.ndarray,
    f_u: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Point estimate and variance used by the paper's paired PPI z-test.

    This intentionally matches the original `basic_power_analysis.py` and
    `nonparam_power_analysis.py` implementations. It differs from the CI
    helper above, which estimates the variance of the rectifier and imputed
    terms after plugging in lambda.
    """
    l_size = y_l.shape[1]
    u_size = f_u.shape[1]
    f_all = np.concatenate([f_l, f_u], axis=1)
    y_mean = y_l.mean(axis=1)
    f_l_mean = f_l.mean(axis=1)
    f_u_mean = f_u.mean(axis=1)
    var_y = y_l.var(axis=1, ddof=1)
    var_f = f_all.var(axis=1, ddof=1)
    cov = ((y_l - y_mean[:, None]) * (f_l - f_l_mean[:, None])).sum(axis=1) / (l_size - 1)
    valid = np.isfinite(y_mean) & np.isfinite(var_y) & np.isfinite(var_f) & np.isfinite(cov) & (var_f > 0)
    lam = np.zeros_like(y_mean)
    lam[valid] = cov[valid] / ((1.0 + l_size / u_size) * var_f[valid])
    point = y_mean + lam * (f_u_mean - f_l_mean)
    variance = var_y / l_size - u_size * (cov * cov) / (l_size * (l_size + u_size) * var_f)
    point[~valid] = np.nan
    variance[~valid] = np.nan
    return point, variance


def human_z_reject(values: np.ndarray, alpha: float = 0.05) -> np.ndarray:
    means = values.mean(axis=1)
    vars_ = values.var(axis=1, ddof=1)
    z = means / np.sqrt(np.maximum(vars_, 1e-300) / values.shape[1])
    return z > zcrit_one_sided(alpha)


def human_z_valid(values: np.ndarray) -> np.ndarray:
    means = values.mean(axis=1)
    vars_ = values.var(axis=1, ddof=1)
    return np.isfinite(means) & np.isfinite(vars_)


def auto_z_reject(values: np.ndarray, alpha: float = 0.05) -> np.ndarray:
    return human_z_reject(values, alpha)


def ppi_z_reject(y_l: np.ndarray, f_l: np.ndarray, f_u: np.ndarray, alpha: float = 0.05) -> np.ndarray:
    point, variance = ppi_z_point_and_var(y_l, f_l, f_u)
    z = point / np.sqrt(np.maximum(variance, 1e-300))
    return z > zcrit_one_sided(alpha)


def ppi_z_valid(y_l: np.ndarray, f_l: np.ndarray, f_u: np.ndarray) -> np.ndarray:
    point, variance = ppi_z_point_and_var(y_l, f_l, f_u)
    return np.isfinite(point) & np.isfinite(variance)


def _perm_reject_from_counts(counts: np.ndarray, num_permutations: int, alpha: float, strict: bool) -> np.ndarray:
    if strict:
        return (counts + 1) < alpha * (num_permutations + 1)
    return (counts + 1) / (num_permutations + 1) <= alpha


def _broadcast_signs(signs: np.ndarray, rows: int) -> np.ndarray:
    if signs.ndim == 2:
        return np.broadcast_to(signs[None, :, :], (rows, signs.shape[0], signs.shape[1]))
    return signs


def human_perm_reject(
    values: np.ndarray,
    signs: np.ndarray,
    alpha: float = 0.05,
    strict: bool = False,
) -> np.ndarray:
    signs = _broadcast_signs(signs, values.shape[0])
    observed = values.mean(axis=1)
    perm = np.einsum("tbl,tl->tb", signs, values, optimize=True) / values.shape[1]
    counts = np.sum(perm >= observed[:, None], axis=1)
    return _perm_reject_from_counts(counts, signs.shape[1], alpha, strict)


def ppi_point_estimate(y_l: np.ndarray, f_l: np.ndarray, f_u: np.ndarray) -> np.ndarray:
    point, _ = ppi_z_point_and_var(y_l, f_l, f_u)
    return point


def ppi_perm_reject(
    y_l: np.ndarray,
    f_l: np.ndarray,
    f_u: np.ndarray,
    signs_l: np.ndarray,
    signs_u: np.ndarray,
    alpha: float = 0.05,
    strict: bool = False,
    center_metric: bool = True,
) -> np.ndarray:
    signs_l = _broadcast_signs(signs_l, y_l.shape[0])
    signs_u = _broadcast_signs(signs_u, y_l.shape[0])
    if center_metric:
        # The paper uses the metric-centered PPI permutation test: centering is
        # required for sign-flip validity when metric differences have nonzero mean.
        f_all = np.concatenate([f_l, f_u], axis=1)
        f_mean = f_all.mean(axis=1, keepdims=True)
        f_l = f_l - f_mean
        f_u = f_u - f_mean
    l_size = y_l.shape[1]
    u_size = f_u.shape[1]
    total = l_size + u_size
    observed = ppi_point_estimate(y_l, f_l, f_u)
    sum_y = np.einsum("tbl,tl->tb", signs_l, y_l, optimize=True)
    sum_f_l = np.einsum("tbl,tl->tb", signs_l, f_l, optimize=True)
    sum_f_u = np.einsum("tbu,tu->tb", signs_u, f_u, optimize=True)
    sum_yf = np.sum(y_l * f_l, axis=1)
    sum_f2 = np.sum(f_l * f_l, axis=1) + np.sum(f_u * f_u, axis=1)
    mean_y = sum_y / l_size
    mean_f_l = sum_f_l / l_size
    mean_f_u = sum_f_u / u_size
    mean_f_all = (sum_f_l + sum_f_u) / total
    cov = (sum_yf[:, None] - (sum_y * sum_f_l) / l_size) / (l_size - 1)
    var_f = (sum_f2[:, None] - total * mean_f_all * mean_f_all) / (total - 1)
    with np.errstate(divide="ignore", invalid="ignore"):
        lam = cov / ((1.0 + l_size / u_size) * var_f)
        perm = mean_y + lam * (mean_f_u - mean_f_l)
    counts = np.sum(perm >= observed[:, None], axis=1)
    return _perm_reject_from_counts(counts, signs_l.shape[1], alpha, strict)
