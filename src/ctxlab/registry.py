"""Name -> constructor registries for datasets, arrangements, models, and metrics."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, TypeVar

T = TypeVar("T")

ARRANGEMENTS: dict[str, Callable[[], Any]] = {}
DATASETS: dict[str, Callable[..., list]] = {}
MODELS: dict[str, Callable[..., Any]] = {}
METRICS: dict[str, Callable[[], Any]] = {}


def register_arrangement(cls: type[T]) -> type[T]:
    ARRANGEMENTS[cls.name] = cls  # type: ignore[attr-defined]
    return cls


def register_dataset(name: str) -> Callable[[T], T]:
    def deco(fn: T) -> T:
        DATASETS[name] = fn  # type: ignore[assignment]
        return fn

    return deco


def register_model(kind: str) -> Callable[[T], T]:
    def deco(cls: T) -> T:
        MODELS[kind] = cls  # type: ignore[assignment]
        return cls

    return deco


def register_metric(cls: type[T]) -> type[T]:
    METRICS[cls.name] = cls  # type: ignore[attr-defined]
    return cls


def get_arrangement(name: str) -> Any:
    if name not in ARRANGEMENTS:
        known = ", ".join(sorted(ARRANGEMENTS)) or "(none registered)"
        raise KeyError(f"Unknown arrangement {name!r}. Known: {known}")
    return ARRANGEMENTS[name]()


def get_dataset_loader(name: str) -> Callable[..., list]:
    if name not in DATASETS:
        known = ", ".join(sorted(DATASETS)) or "(none registered)"
        raise KeyError(f"Unknown dataset {name!r}. Known: {known}")
    return DATASETS[name]


def get_model_cls(kind: str) -> Callable[..., Any]:
    if kind not in MODELS:
        known = ", ".join(sorted(MODELS)) or "(none registered)"
        raise KeyError(f"Unknown model kind {kind!r}. Known: {known}")
    return MODELS[kind]


def get_metric(name: str) -> Any:
    if name not in METRICS:
        known = ", ".join(sorted(METRICS)) or "(none registered)"
        raise KeyError(f"Unknown metric {name!r}. Known: {known}")
    return METRICS[name]()


def load_plugins() -> None:
    """Import plugin modules so their decorators populate the registries."""
    import ctxlab.arrangements.filtering  # noqa: F401
    import ctxlab.arrangements.positional  # noqa: F401
    import ctxlab.data.hotpotqa  # noqa: F401
    import ctxlab.data.toy  # noqa: F401
    import ctxlab.data.txlog  # noqa: F401
    import ctxlab.metrics.qa  # noqa: F401
    import ctxlab.metrics.txlog  # noqa: F401
    import ctxlab.models.api  # noqa: F401
    import ctxlab.models.hf_local  # noqa: F401
    import ctxlab.models.mock  # noqa: F401
