from __future__ import annotations

from collections.abc import Iterable
from typing import Any, TypeVar

T = TypeVar("T")

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover - fallback for minimal environments.
    tqdm = None


class NullProgress:
    def __enter__(self) -> "NullProgress":
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def update(self, n: int = 1) -> None:
        return None

    def close(self) -> None:
        return None


def iter_progress(iterable: Iterable[T], *, enabled: bool = True, **kwargs: Any) -> Iterable[T]:
    if enabled and tqdm is not None:
        return tqdm(iterable, **kwargs)
    return iterable


def progress_bar(*, enabled: bool = True, **kwargs: Any) -> Any:
    if enabled and tqdm is not None:
        return tqdm(**kwargs)
    return NullProgress()
