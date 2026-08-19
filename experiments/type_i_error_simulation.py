#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import NormalDist

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ppi_mt_eval.intervals import human_perm_reject, human_z_reject, ppi_perm_reject, ppi_z_reject
from ppi_mt_eval.plotting import ensure_dir, savefig
from ppi_mt_eval.progress import iter_progress, progress_bar


COLORS = {
    "human_z": "#4C78A8",
    "human_z_oracle": "#4C78A8",
    "human_perm": "#72B7B2",
    "ppi_z": "#54A24B",
    "ppi_z_oracle": "#54A24B",
    "ppi_perm": "#E45756",
}
LABELS = {
    "human_z": "Human Z",
    "human_z_oracle": "Human Z (oracle)",
    "human_perm": "Human Perm.",
    "ppi_z": "PPI Z",
    "ppi_z_oracle": "PPI Z (oracle)",
    "ppi_perm": "PPI Perm.",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Synthetic Type I error simulation.")
    p.add_argument("--output-dir", type=Path, default=Path("results/type_i_error"))
    p.add_argument("--rhos", nargs="+", type=float, default=[0.3, 0.7])
    p.add_argument("--nus", nargs="+", default=["3", "10", "inf"])
    p.add_argument("--labeled-sizes", nargs="+", type=int, default=list(range(20, 201, 20)))
    p.add_argument("--unlabeled-size", type=int, default=800)
    p.add_argument("--num-trials", type=int, default=10000)
    p.add_argument("--num-permutations", type=int, default=1000)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--seed", type=int, default=20260704)
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--plot-only", action="store_true")
    p.add_argument("--no-progress", action="store_true", help="Disable tqdm progress bars.")
    return p.parse_args()


def parse_nu(label: str) -> float:
    value = math.inf if label.lower() in {"inf", "infty", "infinity"} else float(label)
    if not math.isinf(value) and value <= 2:
        raise ValueError(f"nu must be greater than 2, got {label!r}")
    return value


def sample_null(rng: np.random.Generator, rows: int, size: int, rho: float, nu_label: str) -> tuple[np.ndarray, np.ndarray]:
    nu = parse_nu(nu_label)
    corr = np.array([[1.0, rho], [rho, 1.0]])
    if math.isinf(nu):
        sample = rng.multivariate_normal(np.zeros(2), corr, size=(rows, size))
    else:
        scale = corr * ((nu - 2.0) / nu)
        normal = rng.multivariate_normal(np.zeros(2), scale, size=(rows, size))
        chi2 = rng.chisquare(df=nu, size=(rows, size, 1))
        sample = normal / np.sqrt(chi2 / nu)
    return sample[:, :, 0], sample[:, :, 1]


def oracle_human(values: np.ndarray, alpha: float) -> np.ndarray:
    z = values.mean(axis=1) / math.sqrt(1.0 / values.shape[1])
    return z > NormalDist().inv_cdf(1.0 - alpha)


def oracle_ppi(y_l: np.ndarray, f_l: np.ndarray, f_u: np.ndarray, rho: float, alpha: float) -> np.ndarray:
    l_size = y_l.shape[1]
    u_size = f_u.shape[1]
    lam = rho / (1.0 + l_size / u_size)
    point = y_l.mean(axis=1) + lam * (f_u.mean(axis=1) - f_l.mean(axis=1))
    variance = 1.0 / l_size - u_size * rho * rho / (l_size * (l_size + u_size))
    z = point / math.sqrt(variance)
    return z > NormalDist().inv_cdf(1.0 - alpha)


def run_condition(args, seed: int, rho: float, nu: str, l_size: int) -> list[dict[str, object]]:
    rng = np.random.default_rng(seed)
    u_size = args.unlabeled_size
    counts = {m: 0 for m in LABELS}
    done = 0
    with progress_bar(
        enabled=not args.no_progress,
        total=args.num_trials,
        desc=f"rho={rho} nu={nu} L={l_size}",
        unit="trial",
        leave=False,
    ) as bar:
        while done < args.num_trials:
            batch = min(args.batch_size, args.num_trials - done)
            d, f = sample_null(rng, batch, l_size + u_size, rho, nu)
            y_l = d[:, :l_size]
            f_l = f[:, :l_size]
            f_u = f[:, l_size:]
            signs_l = rng.choice(np.array([-1, 1], dtype=np.int8), size=(batch, args.num_permutations, l_size))
            signs_u = rng.choice(np.array([-1, 1], dtype=np.int8), size=(batch, args.num_permutations, u_size))
            counts["human_z"] += int(human_z_reject(y_l, args.alpha).sum())
            counts["human_z_oracle"] += int(oracle_human(y_l, args.alpha).sum())
            counts["human_perm"] += int(human_perm_reject(y_l, signs_l, args.alpha).sum())
            counts["ppi_z"] += int(ppi_z_reject(y_l, f_l, f_u, args.alpha).sum())
            counts["ppi_z_oracle"] += int(oracle_ppi(y_l, f_l, f_u, rho, args.alpha).sum())
            counts["ppi_perm"] += int(ppi_perm_reject(y_l, f_l, f_u, signs_l, signs_u, args.alpha).sum())
            done += batch
            bar.update(batch)
    return [
        {
            "rho": rho,
            "nu": nu,
            "labeled_size": l_size,
            "unlabeled_size": u_size,
            "method": method,
            "method_label": LABELS[method],
            "setting": "human_only" if method.startswith("human") else "ppi",
            "rejection_count": count,
            "num_trials": args.num_trials,
            "type_i_error": count / args.num_trials,
        }
        for method, count in counts.items()
    ]


def plot(summary: pd.DataFrame, args) -> None:
    for setting, methods, filename in [
        ("human_only", ["human_z", "human_z_oracle", "human_perm"], "type_i_error_human_only"),
        ("ppi", ["ppi_z", "ppi_z_oracle", "ppi_perm"], "type_i_error_ppi"),
    ]:
        fig, axes = plt.subplots(len(args.rhos), len(args.nus), figsize=(12, 6.8), sharex=True, sharey=True)
        axes = np.asarray(axes).reshape(len(args.rhos), len(args.nus))
        for i, rho in enumerate(args.rhos):
            for j, nu in enumerate(args.nus):
                ax = axes[i, j]
                sub = summary[(summary["setting"] == setting) & np.isclose(summary["rho"], rho) & (summary["nu"].astype(str) == str(nu))]
                for method in methods:
                    line = sub[sub["method"] == method].sort_values("labeled_size")
                    ax.plot(
                        line["labeled_size"],
                        line["type_i_error"],
                        marker="o",
                        color=COLORS[method],
                        linestyle="--" if "oracle" in method else "-",
                        label=LABELS[method],
                    )
                ax.axhline(args.alpha, color="#666666", linestyle="--", linewidth=1)
                nu_display = r"\infty" if str(nu) == "inf" else str(nu)
                ax.set_title(rf"$\rho={rho}$, $\nu={nu_display}$")
                ax.set_xlabel("L")
                ax.set_ylabel("Empirical Type I error")
                ax.grid(alpha=0.25)
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False)
        fig.tight_layout(rect=(0, 0.08, 1, 1))
        savefig(fig, args.output_dir / f"{filename}.pdf")


def main() -> None:
    args = parse_args()
    ensure_dir(args.output_dir)
    if args.plot_only:
        summary = pd.read_csv(args.output_dir / "type_i_error_summary.csv")
    else:
        seed_sequence = np.random.SeedSequence(args.seed)
        conditions = [
            (rho, str(nu), l_size)
            for rho in args.rhos
            for nu in args.nus
            for l_size in args.labeled_sizes
        ]
        child_seeds = seed_sequence.spawn(len(conditions))
        rows = []
        condition_seeds = list(zip(conditions, child_seeds, strict=True))
        for (rho, nu, l_size), child_seed in iter_progress(
            condition_seeds,
            enabled=not args.no_progress,
            desc="conditions",
            unit="condition",
        ):
            rows.extend(run_condition(args, int(child_seed.generate_state(1)[0]), rho, str(nu), l_size))
            print(f"completed rho={rho} nu={nu} L={l_size}", flush=True)
        summary = pd.DataFrame(rows)
        summary.to_csv(args.output_dir / "type_i_error_summary.csv", index=False)
        (args.output_dir / "config.json").write_text(json.dumps(vars(args), default=str, indent=2), encoding="utf-8")
    plot(summary, args)
    print(summary.groupby("method")["type_i_error"].agg(["mean", "max"]).round(4).to_string())


if __name__ == "__main__":
    main()
