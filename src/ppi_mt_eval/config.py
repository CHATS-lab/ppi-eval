from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DatasetSpec:
    test_set: str
    language_pair: str
    human_col: str
    human_type: str

    @property
    def key(self) -> str:
        return f"{self.test_set}:{self.language_pair}"

    @property
    def label(self) -> str:
        return f"{self.test_set} {self.language_pair}"

    @property
    def tag(self) -> str:
        return f"{self.test_set}_{self.language_pair}".replace("-", "_")

    def path(self, dataset_dir: Path) -> Path:
        return dataset_dir / f"{self.test_set}.{self.language_pair}.tsv"


@dataclass(frozen=True)
class PaperConfig:
    datasets: tuple[DatasetSpec, ...]
    representative_metric_by_testset: dict[str, str]
    bleu_metric: str
    gemba_metrics: dict[str, str]
    excluded_metric_columns: dict[str, tuple[str, ...]]


def load_config(path: Path) -> PaperConfig:
    raw = json.loads(path.read_text(encoding="utf-8"))
    return PaperConfig(
        datasets=tuple(DatasetSpec(**item) for item in raw["datasets"]),
        representative_metric_by_testset=dict(raw["representative_metric_by_testset"]),
        bleu_metric=str(raw["bleu_metric"]),
        gemba_metrics=dict(raw["gemba_metrics"]),
        excluded_metric_columns={
            key: tuple(value) for key, value in raw.get("excluded_metric_columns", {}).items()
        },
    )


def select_specs(config: PaperConfig, dataset_keys: list[str] | None) -> list[DatasetSpec]:
    specs = list(config.datasets)
    if not dataset_keys:
        return specs
    wanted = set(dataset_keys)
    selected = [spec for spec in specs if spec.key in wanted]
    missing = wanted.difference(spec.key for spec in selected)
    if missing:
        raise ValueError(f"Unknown dataset keys: {sorted(missing)}")
    return selected

