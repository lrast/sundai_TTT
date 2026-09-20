"""JSONL run log (always) plus optional Weights & Biases."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ctxlab.config import WandbConfig

RECORD_FILE = "records.jsonl"


def record_key(record: dict[str, Any]) -> tuple[str, str, str]:
    return (str(record["uid"]), str(record["arrangement"]), str(record["model"]))


def load_records(run_dir: Path) -> list[dict[str, Any]]:
    path = Path(run_dir) / RECORD_FILE
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return records


class Tracker:
    def __init__(self, run_dir: Path, wandb_cfg: WandbConfig, run_name: str) -> None:
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.run_dir / RECORD_FILE
        self._fh = self.path.open("a", encoding="utf-8")
        self._wandb = None
        if wandb_cfg.enabled:
            try:
                import wandb
            except ImportError as exc:
                raise ImportError("W&B logging requires `uv sync --extra wandb`") from exc
            self._wandb = wandb.init(
                project=wandb_cfg.project,
                entity=wandb_cfg.entity,
                name=run_name,
                dir=str(self.run_dir),
                resume="allow",
            )

    def completed_keys(self) -> set[tuple[str, str, str]]:
        return {record_key(r) for r in load_records(self.run_dir)}

    def log(self, record: dict[str, Any]) -> None:
        self._fh.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
        self._fh.flush()
        if self._wandb is not None:
            metrics = {
                f"{record['arrangement']}/{k}": v for k, v in record.get("metrics", {}).items()
            }
            self._wandb.log(
                {
                    **metrics,
                    "arrangement": record["arrangement"],
                    "model": record["model"],
                    "uid": record["uid"],
                }
            )

    def close(self) -> None:
        self._fh.close()
        if self._wandb is not None:
            self._wandb.finish()
