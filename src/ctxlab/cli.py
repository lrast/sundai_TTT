"""CLI: `ctxlab run` and `ctxlab report`."""

from __future__ import annotations

from pathlib import Path

import typer
from dotenv import load_dotenv
from rich.console import Console

from ctxlab.config import load_config
from ctxlab.report import report as render_report
from ctxlab.runner import run_experiment

load_dotenv()

app = typer.Typer(no_args_is_help=True, add_completion=False)
console = Console()


@app.command()
def run(
    config: Path = typer.Option(..., "-c", "--config", exists=True, readable=True),
) -> None:
    """Run an arrangement x model sweep from a YAML config."""
    cfg = load_config(config)
    run_dir = run_experiment(cfg)
    console.print(f"[green]wrote[/green] {run_dir / 'records.jsonl'}")
    render_report(run_dir, console=console)


@app.command()
def report(
    run_dir: Path = typer.Argument(..., exists=True, readable=True),
) -> None:
    """Print the arrangement x model table for an existing run."""
    render_report(run_dir, console=console)


if __name__ == "__main__":
    app()
