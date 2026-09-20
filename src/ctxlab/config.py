"""Experiment configuration loaded from YAML."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field

DEFAULT_SYSTEM_PROMPT = (
    "You are a question-answering assistant. Use the provided context if it is present. "
    "Reply with the short answer only, with no explanation or punctuation beyond the answer itself."
)


class DatasetConfig(BaseModel):
    name: str
    config: str | None = None
    split: str = "validation"
    n: int | None = None
    seed: int = 0
    cache_dir: str = ".cache/hf"
    hub_id: str | None = None
    extra: dict[str, Any] = Field(default_factory=dict)


class ModelConfig(BaseModel):
    name: str
    kind: Literal["mock", "api", "hf_local", "ttt"]
    model: str | None = None
    temperature: float = 0.0
    max_tokens: int = 64
    extra: dict[str, Any] = Field(default_factory=dict)


class WandbConfig(BaseModel):
    enabled: bool = False
    project: str = "ctxlab"
    entity: str | None = None


class ExperimentConfig(BaseModel):
    run_id: str
    seed: int = 0
    cache_dir: Path = Path(".cache/completions")
    runs_dir: Path = Path("runs")
    dataset: DatasetConfig
    arrangements: list[str]
    models: list[ModelConfig]
    metrics: list[str] = Field(default_factory=lambda: ["em", "f1"])
    wandb: WandbConfig = Field(default_factory=WandbConfig)
    system_prompt: str = DEFAULT_SYSTEM_PROMPT


def _deep_merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, value in overlay.items():
        if key in out and isinstance(out[key], dict) and isinstance(value, dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def load_config(path: Path | str) -> ExperimentConfig:
    path = Path(path)
    raw = yaml.safe_load(path.read_text()) or {}
    extends = raw.pop("extends", None)
    if extends:
        if Path(extends).is_absolute():
            base_path = Path(extends)
        else:
            base_path = (path.parent / extends).resolve()
        # Also allow paths relative to repo root (cwd).
        if not base_path.exists():
            base_path = Path(extends)
        base = yaml.safe_load(base_path.read_text()) or {}
        raw = _deep_merge(base, raw)
    return ExperimentConfig.model_validate(raw)


def dump_config(cfg: ExperimentConfig, path: Path) -> None:
    path.write_text(yaml.safe_dump(cfg.model_dump(mode="json"), sort_keys=False))
