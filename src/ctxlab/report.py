"""Aggregate JSONL runs into an arrangement x model table."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.table import Table

from ctxlab.tracking import load_records


def aggregate(records: list[dict[str, Any]]) -> dict[tuple[str, str], dict[str, float]]:
    buckets: dict[tuple[str, str], list[dict[str, float]]] = defaultdict(list)
    for rec in records:
        buckets[(rec["arrangement"], rec["model"])].append(rec.get("metrics") or {})
    out: dict[tuple[str, str], dict[str, float]] = {}
    for key, rows in buckets.items():
        metric_names = sorted({name for row in rows for name in row})
        means: dict[str, float] = {"n": float(len(rows))}
        for name in metric_names:
            vals = [row[name] for row in rows if name in row]
            means[name] = sum(vals) / len(vals) if vals else 0.0
        out[key] = means
    return out


def render_table(records: list[dict[str, Any]], *, metric: str = "em") -> Table:
    agg = aggregate(records)
    arrangements = sorted({k[0] for k in agg})
    models = sorted({k[1] for k in agg})
    table = Table(title=f"mean {metric}")
    table.add_column("arrangement", style="bold")
    for model in models:
        table.add_column(model, justify="right")
    for arr in arrangements:
        cells = [arr]
        for model in models:
            cell = agg.get((arr, model))
            if not cell:
                cells.append("—")
            else:
                cells.append(f"{cell.get(metric, 0.0):.3f} (n={int(cell['n'])})")
        table.add_row(*cells)
    return table


def report(
    run_dir: Path, *, console: Console | None = None
) -> dict[tuple[str, str], dict[str, float]]:
    run_dir = Path(run_dir)
    records = load_records(run_dir)
    console = console or Console()
    if not records:
        console.print(f"[yellow]No records in {run_dir / 'records.jsonl'}[/yellow]")
        return {}
    seen = {name for rec in records for name in (rec.get("metrics") or {})}
    first = [m for m in ("em", "f1") if m in seen]
    for metric in first + sorted(seen - set(first)):
        console.print(render_table(records, metric=metric))
    return aggregate(records)
