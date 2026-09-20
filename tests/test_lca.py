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
