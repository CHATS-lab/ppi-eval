#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from ppi_mt_eval.plotting import ensure_dir


SYSTEM_META_ORDER = ("pearson", "spearman", "kendall", "spa", "ppsr")
SEGMENT_META_ORDER = ("input_r", "global_r", "system_r", "pdp", "ppsr")
META_TEX_LABELS = {
    "input_r": "Group-by-Item $r$",
    "global_r": "No-Grouping $r$",
    "system_r": "Group-by-System $r$",
    "pdp": "PDP",
    "pearson": "$r$",
    "spearman": "$\\rho$",
    "kendall": "$\\tau_b$",
    "spa": "SPA",
    "ppsr": "PPSR",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Create LaTeX metric score/rank tables.")
    p.add_argument("--scores", type=Path, default=Path("results/ppsr_discriminative_power/metric_scores.csv"))
    p.add_argument(
        "--segment-scores",
        type=Path,
        default=Path("results/segment_meta_metric_discriminative_power/metric_scores.csv"),
    )
    p.add_argument("--output-dir", type=Path, default=Path("results/ppsr_metric_rank_tables"))
    return p.parse_args()


def latex_escape(text: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(char, char) for char in text)


def fmt(score: float, rank: int | float) -> str:
    if not np.isfinite(score) or pd.isna(rank):
        return "--"
    rank_int = int(rank)
    rank_text = rf"\phantom{{0}}{rank_int}" if rank_int < 10 else str(rank_int)
    return f"{score:.3f} ({rank_text})"


def table_label(test_set: str, language_pair: str, prefix: str) -> str:
    return f"tab:{prefix}_metric_score_ranks_{test_set}_{language_pair}".replace("-", "_")


def make_table(group: pd.DataFrame, test_set: str, language_pair: str, meta_order: tuple[str, ...], prefix: str, caption_kind: str):
    dataset_label = f"{test_set.upper()} {language_pair}"
    pivot = group.pivot_table(index="metric", columns="meta_metric", values="score", aggfunc="first")
    pivot = pivot.reindex(columns=meta_order)
    ranks = pivot.rank(method="min", ascending=False, na_option="bottom").astype("Int64")
    ordered = pivot.assign(_metric=pivot.index).sort_values(["ppsr", "_metric"], ascending=[False, True], na_position="last")
    rows = []
    tex = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{3pt}",
        r"\begin{adjustbox}{max width=\textwidth}",
        r"\begin{tabular}{l" + "r" * len(meta_order) + "}",
        r"\toprule",
        "Metric & " + " & ".join(META_TEX_LABELS[key] for key in meta_order) + r" \\",
        r"\midrule",
    ]
    for metric in ordered["_metric"]:
        row = {"dataset": dataset_label, "metric": metric}
        cells = []
        for meta_metric in meta_order:
            row[f"{meta_metric}_score"] = pivot.loc[metric, meta_metric]
            row[f"{meta_metric}_rank"] = ranks.loc[metric, meta_metric]
            cells.append(fmt(float(pivot.loc[metric, meta_metric]), ranks.loc[metric, meta_metric]))
        rows.append(row)
        tex.append(latex_escape(metric.removesuffix(":seg")) + " & " + " & ".join(cells) + r" \\")
    tex.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{adjustbox}",
            (
                rf"\caption{{Scores and ranks of automatic metrics under each {caption_kind} "
                rf"meta-metric for {latex_escape(dataset_label)}. Metrics are sorted by PPSR; "
                r"tied scores receive the same rank.}"
            ),
            rf"\label{{{table_label(test_set, language_pair, prefix)}}}",
            r"\end{table*}",
            "",
        ]
    )
    return "\n".join(tex), pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    ensure_dir(args.output_dir)
    chunks = []
    frames = []
    if args.scores.exists():
        scores = pd.read_csv(args.scores)
        for (test_set, language_pair), group in scores.groupby(["test_set", "language_pair"], sort=False):
            tex, table = make_table(group, test_set, language_pair, SYSTEM_META_ORDER, "system", "system-level")
            chunks.append(tex)
            frames.append(table.assign(table_type="system"))
    if args.segment_scores.exists():
        scores = pd.read_csv(args.segment_scores)
        for (test_set, language_pair), group in scores.groupby(["test_set", "language_pair"], sort=False):
            tex, table = make_table(group, test_set, language_pair, SEGMENT_META_ORDER, "segment", "segment-level")
            chunks.append(tex)
            frames.append(table.assign(table_type="segment"))
    if not frames:
        raise FileNotFoundError(f"Neither {args.scores} nor {args.segment_scores} exists.")
    pd.concat(frames, ignore_index=True).to_csv(args.output_dir / "metric_score_rank_tables.csv", index=False)
    text = "\n".join(chunks)
    (args.output_dir / "metric_score_rank_tables.tex").write_text(text, encoding="utf-8")
    print(f"wrote {args.output_dir / 'metric_score_rank_tables.tex'}")


if __name__ == "__main__":
    main()
