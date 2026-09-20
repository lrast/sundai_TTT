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


# --- transaction-log windows --------------------------------------------------


@pytest.fixture(scope="module")
def txlog_example() -> Example:
    from ctxlab.config import DatasetConfig
    from ctxlab.data.txlog import load_txlog

    return load_txlog(DatasetConfig(name="txlog", n=1, seed=0))[0]


@pytest.mark.parametrize("size", [25, 50, 95, 175, 250, 500])
def test_tx_window_keeps_exactly_size_lines(txlog_example: Example, size: int) -> None:
    from ctxlab.registry import get_arrangement

    prompt = get_arrangement(f"tx_window_{size}").build(txlog_example, random.Random(0))
    assert prompt.meta["n_passages"] == size


@pytest.mark.parametrize("size", [25, 50, 95, 175, 250, 500])
@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_tx_window_always_keeps_every_gold_line(
    txlog_example: Example, size: int, seed: int
) -> None:
    """The needle must stay in view at every length, or the sweep measures
    retrievability rather than the effect of context length."""
    from ctxlab.registry import get_arrangement

    prompt = get_arrangement(f"tx_window_{size}").build(txlog_example, random.Random(seed))
    n_gold = sum(1 for p in txlog_example.passages if p.is_gold)
    assert prompt.meta["n_gold"] == n_gold
    assert len(prompt.meta["gold_positions"]) == n_gold


@pytest.mark.parametrize("size", [25, 95, 250])
def test_tx_window_is_contiguous(txlog_example: Example, size: int) -> None:
    """Non-contiguous samples would break the balance chain and make the log
    unverifiable."""
    from ctxlab.arrangements.filtering import tx_window

    window = tx_window(txlog_example, random.Random(3), size)
    texts = [p.text for p in txlog_example.passages]
    start = texts.index(window[0].text)
    assert texts[start : start + size] == [p.text for p in window]


def test_tx_window_offset_is_seeded_but_varies(txlog_example: Example) -> None:
    from ctxlab.arrangements.filtering import tx_window

    a = tx_window(txlog_example, random.Random(7), 95)
    b = tx_window(txlog_example, random.Random(7), 95)
    c = tx_window(txlog_example, random.Random(8), 95)
    assert [p.title for p in a] == [p.title for p in b]
    assert [p.title for p in a] != [p.title for p in c]


def test_tx_window_reports_what_it_dropped(txlog_example: Example) -> None:
    from ctxlab.registry import get_arrangement

    prompt = get_arrangement("tx_window_25").build(txlog_example, random.Random(0))
    n_total = len(txlog_example.passages)
    assert len(prompt.meta["dropped_titles"]) == n_total - 25
    assert prompt.meta["n_gold_available"] == sum(1 for p in txlog_example.passages if p.is_gold)


def test_tx_window_wider_than_the_log_keeps_everything() -> None:
    import random as _random

    from ctxlab.arrangements.filtering import tx_window
    from ctxlab.data.base import Passage

    ex = Example(
        uid="short",
        question="Q",
        answers=["A"],
        passages=[Passage(f"TX{i:03d}", f"line {i}", i == 1) for i in range(5)],
    )
    assert len(tx_window(ex, _random.Random(0), 500)) == 5


def test_tx_window_rejects_an_example_with_no_gold() -> None:
    import random as _random

    from ctxlab.arrangements.filtering import tx_window
    from ctxlab.data.base import Passage

    ex = Example(
        uid="nogold",
        question="Q",
        answers=["A"],
        passages=[Passage(f"TX{i:03d}", f"line {i}", False) for i in range(50)],
    )
    with pytest.raises(ValueError, match="no gold passage"):
        tx_window(ex, _random.Random(0), 25)


@pytest.mark.parametrize("name", ["tx_window_25", "tx_window_95", "tx_window_250", "tx_window_500"])
def test_registry_has_the_txlog_windows(name: str) -> None:
    from ctxlab.registry import ARRANGEMENTS

    assert name in ARRANGEMENTS
