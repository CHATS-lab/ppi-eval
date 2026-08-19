#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import ppsr_discriminative_power as discrim
import ppsr_ranking_stability as base

_BASE_PARSE_ARGS = base.parse_args


def parse_args():
    args = _BASE_PARSE_ARGS()
    if args.output_dir == Path("results/ppsr_ranking_stability"):
        args.output_dir = Path("results/segment_meta_metric_ranking_stability")
    args.meta_metrics = list(discrim.SEGMENT_META_METRICS)
    args.figure_prefix = "ranking_stability_segment"
    args.y_min = 0.65
    args.y_max = 1.0
    return args


def main() -> None:
    base.parse_args = parse_args
    base.main()


if __name__ == "__main__":
    main()
