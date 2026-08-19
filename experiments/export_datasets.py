#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from ppi_mt_eval.config import load_config
from ppi_mt_eval.data import dataset_summary_rows, export_scores


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export paper WMT datasets with mt_metrics_eval.")
    p.add_argument("--config", type=Path, default=Path("configs/paper_datasets.json"))
    p.add_argument("--dataset-dir", type=Path, default=Path("datasets"))
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--summary-path", type=Path, default=Path("datasets/dataset_summary.csv"))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    args.dataset_dir.mkdir(parents=True, exist_ok=True)
    log = []
    for spec in config.datasets:
        path = export_scores(spec, args.dataset_dir, overwrite=args.overwrite)
        log.append({"dataset": spec.label, "path": str(path)})
        print(f"ready {spec.label}: {path}", flush=True)
    rows = dataset_summary_rows(config, args.dataset_dir)
    args.summary_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.summary_path, index=False)
    (args.dataset_dir / "export_log.json").write_text(json.dumps(log, indent=2), encoding="utf-8")
    print(f"wrote {args.summary_path}")


if __name__ == "__main__":
    main()

