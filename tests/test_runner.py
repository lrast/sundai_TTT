from pathlib import Path

from ctxlab.config import ExperimentConfig, ModelConfig, load_config
from ctxlab.runner import run_experiment
from ctxlab.tracking import load_records


def _cfg(tmp_path: Path, run_id: str = "test") -> ExperimentConfig:
    from ctxlab.config import DatasetConfig, WandbConfig

    return ExperimentConfig(
        run_id=run_id,
        seed=0,
        cache_dir=tmp_path / "cache",
        runs_dir=tmp_path / "runs",
        dataset=DatasetConfig(name="toy", n=4),
        arrangements=["no_context", "gold_only", "gold_first"],
        models=[ModelConfig(name="mock", kind="mock", extra={"behavior": "oracle"})],
        metrics=["em", "f1"],
        wandb=WandbConfig(enabled=False),
    )


def test_runner_smoke_oracle_separates_closed_and_open(tmp_path: Path) -> None:
    run_dir = run_experiment(_cfg(tmp_path))
    records = load_records(run_dir)
    assert len(records) == 4 * 3  # examples x arrangements
    by_arr: dict[str, list[float]] = {}
    for rec in records:
        by_arr.setdefault(rec["arrangement"], []).append(rec["metrics"]["em"])
    # toy answers equal first gold title; oracle returns that iff gold is kept
    assert sum(by_arr["no_context"]) == 0
    assert sum(by_arr["gold_only"]) == 4
    assert sum(by_arr["gold_first"]) == 4
    assert (run_dir / "config.yaml").exists()


def test_runner_resumes(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path, run_id="resume")
    run_experiment(cfg)
    run_experiment(cfg)
    records = load_records(tmp_path / "runs" / "resume")
    keys = [(r["uid"], r["arrangement"], r["model"]) for r in records]
    assert len(keys) == len(set(keys)) == 12


def test_load_smoke_yaml() -> None:
    cfg = load_config(Path("configs/experiments/smoke.yaml"))
    assert cfg.run_id == "smoke"
    assert cfg.dataset.name == "toy"
    assert "gold_middle" in cfg.arrangements
    assert cfg.wandb.enabled is False
