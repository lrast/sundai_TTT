from __future__ import annotations

import random

import pytest

from ctxlab.arrangements.filtering import GoldOnly, NoContext
from ctxlab.arrangements.positional import GoldFirst, GoldLast, GoldMiddle, Shuffled
from ctxlab.data.base import Example
from ctxlab.registry import load_plugins

load_plugins()


def _titles(prompt) -> list[str]:
    return list(prompt.meta["passage_titles"])


def test_no_context_drops_everything(toy_paris: Example) -> None:
    prompt = NoContext().build(toy_paris, random.Random(0))
    assert prompt.meta["n_passages"] == 0
    assert prompt.meta["gold_positions"] == []
    assert prompt.meta["n_gold"] == 0
    assert set(prompt.meta["dropped_titles"]) == {p.title for p in toy_paris.passages}
    assert "Context:" not in prompt.messages[0].content


def test_gold_only_keeps_only_gold(mixed_example: Example) -> None:
    prompt = GoldOnly().build(mixed_example, random.Random(0))
    assert _titles(prompt) == ["g1", "g2"]
    assert prompt.meta["gold_positions"] == [0, 1]
    assert prompt.meta["dropped_titles"] == ["d1", "d2", "d3"]
    assert prompt.meta["n_gold"] == 2


def test_gold_first_order(mixed_example: Example) -> None:
    prompt = GoldFirst().build(mixed_example, random.Random(0))
    assert _titles(prompt) == ["g1", "g2", "d1", "d2", "d3"]
    assert prompt.meta["gold_positions"] == [0, 1]
    assert prompt.meta["dropped_titles"] == []


def test_gold_last_order(mixed_example: Example) -> None:
    prompt = GoldLast().build(mixed_example, random.Random(0))
    assert _titles(prompt) == ["d1", "d2", "d3", "g1", "g2"]
    assert prompt.meta["gold_positions"] == [3, 4]


def test_gold_middle_order(mixed_example: Example) -> None:
    prompt = GoldMiddle().build(mixed_example, random.Random(0))
    # 3 distractors -> mid=1, so [d1] + golds + [d2, d3]
    assert _titles(prompt) == ["d1", "g1", "g2", "d2", "d3"]
    assert prompt.meta["gold_positions"] == [1, 2]


def test_shuffled_is_seeded(mixed_example: Example) -> None:
    a = Shuffled().build(mixed_example, random.Random(7))
    b = Shuffled().build(mixed_example, random.Random(7))
    c = Shuffled().build(mixed_example, random.Random(8))
    assert _titles(a) == _titles(b)
    assert set(_titles(a)) == {p.title for p in mixed_example.passages}
    assert _titles(a) != _titles(c)


def test_gold_positions_match_is_gold(mixed_example: Example) -> None:
    gold_titles = {p.title for p in mixed_example.passages if p.is_gold}
    for arr in (GoldFirst(), GoldLast(), GoldMiddle(), Shuffled(), GoldOnly()):
        prompt = arr.build(mixed_example, random.Random(1))
        titles = _titles(prompt)
        for i, title in enumerate(titles):
            if title in gold_titles:
                assert i in prompt.meta["gold_positions"], (arr.name, titles)
            else:
                assert i not in prompt.meta["gold_positions"], (arr.name, titles)


def test_positional_arrangements_drop_nothing(mixed_example: Example) -> None:
    n = len(mixed_example.passages)
    for arr in (GoldFirst(), GoldLast(), GoldMiddle(), Shuffled()):
        prompt = arr.build(mixed_example, random.Random(0))
        assert prompt.meta["n_passages"] == n
        assert prompt.meta["dropped_titles"] == []


def test_hotpot_gold_flags(hotpot_example: Example) -> None:
    gold = {p.title for p in hotpot_example.passages if p.is_gold}
    assert gold == {"France", "Paris"}
    prompt = GoldOnly().build(hotpot_example, random.Random(0))
    assert set(prompt.meta["passage_titles"]) == gold


@pytest.mark.parametrize(
    "name",
    ["no_context", "gold_only", "gold_first", "gold_last", "gold_middle", "shuffled"],
)
def test_registry_has_first_sweep(name: str) -> None:
    from ctxlab.registry import ARRANGEMENTS

    assert name in ARRANGEMENTS
