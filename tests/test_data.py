import json
from pathlib import Path

from ctxlab.data.hotpotqa import examples_from_rows
from ctxlab.metrics.judge import JudgeMetric


def test_hotpot_row_maps_supporting_titles() -> None:
    row = json.loads(Path("tests/fixtures/hotpot_tiny.json").read_text())
    ex = examples_from_rows([row])[0]
    assert ex.uid == "hp-1"
    assert ex.answers == ["Paris"]
    by_title = {p.title: p.is_gold for p in ex.passages}
    assert by_title == {"France": True, "Paris": True, "Lyon": False, "Rome": False}
    assert "Its capital is Paris." in ex.passages[0].text


def test_judge_is_stubbed() -> None:
    try:
        JudgeMetric().score("Paris", ["Paris"])
    except NotImplementedError:
        return
    raise AssertionError("judge should still be a stub")
