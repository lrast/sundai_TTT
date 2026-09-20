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


def test_cache_key_separates_models_that_differ_only_in_extra() -> None:
    """Decoding knobs and TTT hyperparameters live in `extra`. If they were
    left out of the cache key, the thinking arm would silently reuse the
    in-context arm's completions whenever the two shared a model name."""
    from ctxlab.cache import prompt_cache_key
    from ctxlab.data.base import Message, Prompt
    from ctxlab.runner import _gen_params

    prompt = Prompt(system="s", messages=(Message(role="user", content="q"),))
    base = ModelConfig(name="m", kind="hf_local", model="x", extra={"top_p": 0.8})
    other = ModelConfig(name="m", kind="hf_local", model="x", extra={"top_p": 0.95})
    keys = [prompt_cache_key(prompt, c.name, _gen_params(c)) for c in (base, other)]
    assert keys[0] != keys[1]


def test_cache_key_is_unchanged_for_configs_without_extra() -> None:
    """Existing API-model runs must keep the completions they already paid for."""
    from ctxlab.cache import prompt_cache_key
    from ctxlab.data.base import Message, Prompt
    from ctxlab.runner import _gen_params

    prompt = Prompt(system="s", messages=(Message(role="user", content="q"),))
    cfg = ModelConfig(name="gpt-4o-mini", kind="api", model="gpt-4o-mini", max_tokens=32)
    legacy = {"temperature": cfg.temperature, "max_tokens": cfg.max_tokens}
    assert prompt_cache_key(prompt, cfg.name, _gen_params(cfg)) == prompt_cache_key(
        prompt, cfg.name, legacy
    )


def test_experiment_configs_parse() -> None:
    for name in ("txlog_smoke", "txlog_gate", "txlog_sweep", "txlog_replication"):
        cfg = load_config(Path(f"configs/experiments/{name}.yaml"))
        assert cfg.run_id == name
        assert cfg.dataset.name == "txlog"
        assert cfg.metrics == ["tx_joint", "tx_type", "tx_id"]
        assert all(a.startswith("tx_window_") for a in cfg.arrangements)


def test_replication_arms_stay_compute_matched() -> None:
    """The paper's claim is a FLOP-matched comparison: T_think ~= 2*N_TTT*k
    (Eq. 3.2). If someone tunes one arm without the other the table still
    renders, but it no longer says what it claims to."""
    cfg = load_config(Path("configs/experiments/txlog_replication.yaml"))
    arms = {m.name: m for m in cfg.models}
    thinking = next(m for n, m in arms.items() if n == "thinking")
    qttt = next(m for n, m in arms.items() if n.startswith("qttt"))
    assert thinking.max_tokens == 2 * qttt.extra["n_ttt"] * qttt.extra["k"]
    assert qttt.kind == "ttt"
    # Every arm must be the same weights, or the comparison is between models.
    assert len({m.model for m in cfg.models}) == 1


def test_backend_diagnostics_reach_the_run_log(tmp_path: Path) -> None:
    """`completion.raw` carries the TTT step count and loss trajectory. If the
    runner dropped it, a zero score would be indistinguishable from an adapter
    that never ran."""
    run_dir = run_experiment(_cfg(tmp_path, run_id="raw"))
    records = load_records(run_dir)
    assert all("raw" in rec for rec in records)
    assert records[0]["raw"] == {"behavior": "oracle"}
