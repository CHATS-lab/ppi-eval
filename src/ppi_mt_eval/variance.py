from __future__ import annotations

import numpy as np


def safe_var(x: np.ndarray) -> float:
    return float(np.var(x, ddof=1)) if x.size > 1 else float("nan")


def safe_cov(x: np.ndarray, y: np.ndarray) -> float:
    if x.size <= 1:
        return float("nan")
    return float(np.cov(x, y, ddof=1)[0, 1])


def human_paired_unpaired_ratio(y1: np.ndarray, y2: np.ndarray) -> float:
    paired = safe_var(y1 - y2)
    unpaired = safe_var(y1) + safe_var(y2)
    return (unpaired - paired) / paired if paired > 0 else float("nan")


def ppi_variance(y: np.ndarray, f: np.ndarray) -> float:
    """Large-unlabeled optimal-PPI variance component before dividing by L."""
    var_y = safe_var(y)
    var_f = safe_var(f)
    cov = safe_cov(y, f)
    if not np.isfinite(var_y) or not np.isfinite(var_f) or var_f <= 0:
        return float("nan")
    # Original paired-vs-unpaired PPI analysis uses U >> L, so U/(L+U) ~= 1.
    return var_y - (cov * cov / var_f)


def ppi_paired_unpaired_ratio(y1: np.ndarray, y2: np.ndarray, f1: np.ndarray, f2: np.ndarray) -> float:
    paired = ppi_variance(y1 - y2, f1 - f2)
    unpaired = ppi_variance(y1, f1) + ppi_variance(y2, f2)
    return (unpaired - paired) / paired if paired > 0 else float("nan")


def corr(x: np.ndarray, y: np.ndarray) -> float:
    if x.size <= 1 or np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])
