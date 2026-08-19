#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import ppsr_discriminative_power as base

_BASE_PARSE_ARGS = base.parse_args


def parse_args():
    args = _BASE_PARSE_ARGS()
    if args.output_dir == Path("results/ppsr_discriminative_power"):
        args.output_dir = Path("results/segment_meta_metric_discriminative_power")
    args.meta_metrics = list(base.SEGMENT_META_METRICS)
    return args


def main() -> None:
    base.parse_args = parse_args
    base.main()


if __name__ == "__main__":
    main()
