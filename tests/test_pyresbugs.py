from __future__ import annotations

import json
from pathlib import Path

from ctxlab.config import DatasetConfig, ExperimentConfig, ModelConfig, WandbConfig, load_config
from ctxlab.data.pyresbugs import (
    buggy_file_from_patch,
    examples_from_rows,
    load_pyresbugs,
)
from ctxlab.runner import run_experiment
from ctxlab.tracking import load_records

FIXTURE = Path("tests/fixtures/pyresbugs_tiny.json")


def _rows() -> list[dict]:
    return json.loads(FIXTURE.read_text())


def test_rows_map_to_bug_localization_examples() -> None:
    rows = _rows()
    examples = examples_from_rows(rows, level="contextual", seed=0)
    assert len(examples) == 4
    by_answer = {ex.answers[0]: ex for ex in examples}
    ex = by_answer["parse_config"]
    golds = [p for p in ex.passages if p.is_gold]
    assert len(golds) == 1
    assert golds[0].title == "parse_config"
    assert "data.get('settings')" in golds[0].text  # the faulty version, not the fix
    assert ex.question == rows[0]["Contextual-Level Description"]


def test_description_level_switches_question() -> None:
    rows = _rows()
    questions = {
        level: {ex.answers[0]: ex.question for ex in examples_from_rows(rows, level=level)}
        for level in ("implementation", "contextual", "high")
    }
    assert questions["implementation"]["clamp_value"] != questions["high"]["clamp_value"]
    assert "clamp_value" in questions["implementation"]["clamp_value"]
    assert "clamp_value" not in questions["high"]["clamp_value"]


def test_distractors_are_fixed_methods_with_unique_titles() -> None:
    for ex in examples_from_rows(_rows(), level="contextual", seed=0):
        titles = [p.title for p in ex.passages]
        assert len(titles) == len(set(titles))
        for p in ex.passages:
            if not p.is_gold:
                assert p.title != ex.answers[0]


def test_no_fixed_code_leaks_into_gold_example_prompt() -> None:
    rows = _rows()
    fixed_by_method = {r["Fixed_Method"]: r["Fault Free Code"] for r in rows}
    for ex in examples_from_rows(rows, level="high", seed=0):
        gold_fix = fixed_by_method[ex.answers[0]]
        for p in ex.passages:
            assert p.text != gold_fix  # the gold row's fix never appears as a passage
        assert ex.answers[0] not in ex.question  # high-level descriptions name no method


def test_sampling_is_seeded_and_deterministic() -> None:
    a = examples_from_rows(_rows(), level="contextual", seed=7)
    b = examples_from_rows(_rows(), level="contextual", seed=7)
    assert [(ex.uid, [p.title for p in ex.passages]) for ex in a] == [
        (ex.uid, [p.title for p in ex.passages]) for ex in b
    ]


def test_mask_answer_removes_method_name_from_question() -> None:
    rows = _rows()
    masked = examples_from_rows(rows, level="implementation", seed=0, mask_answer=True)
    plain = examples_from_rows(rows, level="implementation", seed=0, mask_answer=False)
    for ex in masked:
        assert ex.answers[0].lower() not in ex.question.lower()
        assert "the affected function" in ex.question
    # masking only touches the question, never the passages
    by_uid = {ex.uid: ex for ex in plain}
    for ex in masked:
        assert [p.text for p in ex.passages] == [p.text for p in by_uid[ex.uid].passages]


def test_n_distractors_knob_and_graceful_cap() -> None:
    rows = _rows()
    small = examples_from_rows(rows, level="contextual", seed=0, n_distractors=2)
    assert all(len(ex.passages) == 3 for ex in small)  # gold + 2
    # only 3 other rows exist, so asking for 25 caps at what is available
    big = examples_from_rows(rows, level="contextual", seed=0, n_distractors=25)
    assert all(len(ex.passages) <= 4 for ex in big)
    assert all(sum(p.is_gold for p in ex.passages) == 1 for ex in big)


def test_loader_reads_extra_knobs_from_config() -> None:
    cfg = DatasetConfig(
        name="pyresbugs",
        hub_id=str(FIXTURE),
        config="implementation",
        n=4,
        seed=0,
        extra={"n_distractors": 2, "mask_answer": True},
    )
    examples = load_pyresbugs(cfg)
    assert all(len(ex.passages) == 3 for ex in examples)
    assert all(ex.answers[0].lower() not in ex.question.lower() for ex in examples)


def test_anonymous_question_has_no_description() -> None:
    from ctxlab.data.pyresbugs import ANONYMOUS_QUESTION

    examples = examples_from_rows(_rows(), question_style="anonymous", seed=0)
    assert len(examples) == 4
    for ex in examples:
        assert ex.question == ANONYMOUS_QUESTION
        assert ex.answers[0].lower() not in ex.question.lower()


def test_same_project_scope_excludes_other_projects() -> None:
    rows = _rows()
    by_project = {r["Fixed_Method"]: r["Project"] for r in rows}
    examples = examples_from_rows(rows, level="contextual", seed=0, distractor_scope="same_project")
    for ex in examples:
        gold_project = by_project[ex.answers[0]]
        for p in ex.passages:
            if not p.is_gold:
                assert by_project[p.title] == gold_project


def test_include_fixed_gold_builds_labeled_twins() -> None:
    rows = _rows()
    faulty = {r["Fixed_Method"]: r["Faulty Code"] for r in rows}
    fixed = {r["Fixed_Method"]: r["Fault Free Code"] for r in rows}
    examples = examples_from_rows(rows, question_style="anonymous", seed=0, include_fixed_gold=True)
    for ex in examples:
        name = ex.answers[0].split(" [version")[0]
        versions = [p for p in ex.passages if p.title.startswith(f"{name} [version")]
        assert len(versions) == 2
        golds = [p for p in versions if p.is_gold]
        assert len(golds) == 1
        assert golds[0].title == ex.answers[0]
        assert golds[0].text == faulty[name]
        twin = next(p for p in versions if not p.is_gold)
        assert twin.text == fixed[name]
        assert {"[version A]" in p.title for p in versions} == {True, False}


def test_mention_metric_scans_whole_reply() -> None:
    from ctxlab.metrics.bugloc import Mention

    m = Mention()
    assert m.score("parse_config", ["parse_config"]) == 1.0
    assert m.score("The bug is in\nparse_config, line 2.", ["parse_config"]) == 1.0
    assert m.score("render_row", ["parse_config"]) == 0.0
    assert m.score("", ["parse_config"]) == 0.0


def test_buggy_file_from_patch() -> None:
    rows = _rows()
    assert buggy_file_from_patch(rows[0]["Diff_patch"]) == "src/acme/config.py"
    assert buggy_file_from_patch("--- a/pkg/mod.py\n+++ b/pkg/mod.py\n") == "pkg/mod.py"
    assert buggy_file_from_patch(None) is None
    assert buggy_file_from_patch("no diff headers here") is None


def test_loader_reads_local_json_via_config() -> None:
    cfg = DatasetConfig(name="pyresbugs", hub_id=str(FIXTURE), config="implementation", n=2, seed=0)
    examples = load_pyresbugs(cfg)
    assert len(examples) == 2
    assert all(len([p for p in ex.passages if p.is_gold]) == 1 for ex in examples)


def test_runner_end_to_end_with_mock_oracle(tmp_path: Path) -> None:
    cfg = ExperimentConfig(
        run_id="bugloc-test",
        seed=0,
        cache_dir=tmp_path / "cache",
        runs_dir=tmp_path / "runs",
        dataset=DatasetConfig(name="pyresbugs", hub_id=str(FIXTURE), config="contextual", n=4),
        arrangements=["no_context", "gold_first", "gold_middle"],
        models=[ModelConfig(name="mock", kind="mock", extra={"behavior": "oracle"})],
        metrics=["em", "f1"],
        wandb=WandbConfig(enabled=False),
    )
    run_dir = run_experiment(cfg)
    records = load_records(run_dir)
    assert len(records) == 4 * 3
    by_arr: dict[str, list[float]] = {}
    for rec in records:
        by_arr.setdefault(rec["arrangement"], []).append(rec["metrics"]["em"])
    assert sum(by_arr["no_context"]) == 0
    assert sum(by_arr["gold_first"]) == 4
    assert sum(by_arr["gold_middle"]) == 4


def test_load_bugloc_smoke_yaml() -> None:
    cfg = load_config(Path("configs/experiments/bugloc_smoke.yaml"))
    assert cfg.run_id == "bugloc_smoke"
    assert cfg.dataset.name == "pyresbugs"
    assert cfg.dataset.config == "contextual"
    assert "gold_middle" in cfg.arrangements
