from __future__ import annotations

from ctxlab.config import DatasetConfig
from ctxlab.data.lca import load_lca_bugloc

FIXTURE = "tests/fixtures/lca_tiny.json"
REPOS = "tests/fixtures/lca_repo"


def _cfg(**extra) -> DatasetConfig:
    merged = {"repos_dir": REPOS, "n_files": 10, **extra}
    return DatasetConfig(name="lca_bugloc", hub_id=FIXTURE, split="dev", extra=merged)


def test_single_needle_example_from_real_files() -> None:
    examples = load_lca_bugloc(_cfg(max_changed_files=1))
    assert len(examples) == 1  # multi-file row filtered, missing-repo row skipped
    ex = examples[0]
    assert ex.answers == ["src/toolbelt/config.py"]
    golds = [p for p in ex.passages if p.is_gold]
    assert len(golds) == 1
    assert "data.get('settings')" in golds[0].text
    titles = {p.title for p in ex.passages}
    assert "src/toolbelt/net.py" in titles  # same-package sibling as distractor
    assert "Crash on startup" in ex.question
    assert len(ex.passages) <= 10


def test_multi_needle_row_kept_when_budget_allows() -> None:
    examples = load_lca_bugloc(_cfg(max_changed_files=5))
    multi = [ex for ex in examples if len(ex.answers) == 2]
    assert len(multi) == 1
    ex = multi[0]
    assert sorted(ex.answers) == ["src/toolbelt/net.py", "src/toolbelt/table.py"]
    assert sum(p.is_gold for p in ex.passages) == 2


def test_missing_repo_checkout_is_skipped() -> None:
    examples = load_lca_bugloc(_cfg(max_changed_files=5))
    assert all(not ex.uid.startswith("ghost/") for ex in examples)


def test_file_truncation_budget() -> None:
    examples = load_lca_bugloc(_cfg(max_changed_files=1, max_file_chars=20))
    assert all(len(p.text) <= 20 for ex in examples for p in ex.passages)


def test_gold_recall_metric_multi_needle() -> None:
    from ctxlab.metrics.bugloc import GoldRecall

    m = GoldRecall()
    golds = ["src/toolbelt/net.py", "src/toolbelt/table.py"]
    assert m.score("src/toolbelt/net.py", golds) == 0.5
    assert m.score("net.py: src/toolbelt/net.py and src/toolbelt/table.py", golds) == 1.0
    assert m.score("something else", golds) == 0.0


def test_changed_files_python_repr_string() -> None:
    # The HF dataset serializes changed_files as a Python repr string.
    import json as _json

    rows = _json.loads(open(FIXTURE).read())
    for r in rows:
        r["changed_files"] = repr(r["changed_files"])
    import tempfile
    from pathlib import Path as _P

    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        _json.dump(rows, f)
        tmp = f.name
    cfg = DatasetConfig(
        name="lca_bugloc",
        hub_id=tmp,
        split="dev",
        extra={"repos_dir": REPOS, "max_changed_files": 5},
    )
    examples = load_lca_bugloc(cfg)
    assert {ex.answers[0] for ex in examples} >= {"src/toolbelt/config.py"}
    multi = [ex for ex in examples if len(ex.answers) == 2]
    assert multi and sorted(multi[0].answers) == [
        "src/toolbelt/net.py",
        "src/toolbelt/table.py",
    ]
    _P(tmp).unlink()


def test_titles_only_hides_contents_keeps_truthful_meta() -> None:
    import random as _random

    from ctxlab.registry import get_arrangement, load_plugins

    load_plugins()
    cfg = DatasetConfig(name="lca_bugloc", hub_id=FIXTURE, split="dev", extra={"repos_dir": REPOS})
    ex = load_lca_bugloc(cfg)[0]
    prompt = get_arrangement("titles_only").build(ex, _random.Random(0))
    body = prompt.messages[0].content
    for p in ex.passages:
        assert p.title in body  # every path listed
        if len(p.text.strip()) > 40:
            assert p.text.strip()[:40] not in body  # no contents leak
    assert prompt.meta["n_passages"] == len(ex.passages)
    assert prompt.meta["gold_positions"] == [i for i, p in enumerate(ex.passages) if p.is_gold]


def test_anonymize_titles_breaks_the_hint_channel() -> None:
    cfg = DatasetConfig(
        name="lca_bugloc",
        hub_id=FIXTURE,
        split="dev",
        extra={"repos_dir": REPOS, "anonymize_titles": True},
    )
    examples = load_lca_bugloc(cfg)
    for ex in examples:
        titles = [p.title for p in ex.passages]
        assert all(t.startswith("file_") for t in titles)  # no real paths shown
        assert len(titles) == len(set(titles))
        golds = [p for p in ex.passages if p.is_gold]
        assert [p.title for p in golds] == ex.answers  # answer is the gold's label
        assert "toolbelt" not in " ".join(titles)  # real path never leaks via labels
    # deterministic: same seed, same labels
    again = load_lca_bugloc(cfg)
    assert [ex.answers for ex in examples] == [ex.answers for ex in again]
