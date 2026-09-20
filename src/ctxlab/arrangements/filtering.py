"""Arrangements that drop passages rather than reorder them."""

from __future__ import annotations

import random

from ctxlab.arrangements.formatting import build_prompt, build_txlog_prompt
from ctxlab.data.base import Example, Passage, Prompt
from ctxlab.registry import register_arrangement


@register_arrangement
class NoContext:
    """Closed-book floor: question only."""

    name = "no_context"

    def build(self, ex: Example, rng: random.Random) -> Prompt:
        del rng
        return build_prompt(ex, [], self.name)


@register_arrangement
class GoldOnly:
    """Oracle ceiling: supporting passages only."""

    name = "gold_only"

    def build(self, ex: Example, rng: random.Random) -> Prompt:
        del rng
        golds = [p for p in ex.passages if p.is_gold]
        return build_prompt(ex, golds, self.name)


# Context lengths for the transaction-log replication. The paper's Figure 1(b)
# sweeps 25/95/250/500 transactions; the shorter extras exist so a small model
# on modest hardware can run a sweep that is not entirely on the accuracy
# floor. Registration is free -- the YAML `arrangements:` list picks the subset.
TX_WINDOWS = (25, 50, 95, 175, 250, 500)


def tx_window(ex: Example, rng: random.Random, size: int) -> list[Passage]:
    """A contiguous run of `size` transactions containing every gold line.

    Contiguity is the point: each line states both accounts' old -> new
    balances, so an unbroken run is a verifiable log, whereas a scattered
    sample is not. The needle stays in view and only the haystack grows, which
    is how the paper isolates context length from retrieval difficulty.
    """
    passages = list(ex.passages)
    if size >= len(passages):
        return passages
    gold = [i for i, p in enumerate(passages) if p.is_gold]
    if not gold:
        raise ValueError(f"{ex.uid} has no gold passage to centre a window on")
    lo, hi = gold[0], gold[-1]
    span = hi - lo + 1
    if span > size:
        raise ValueError(f"gold span of {span} does not fit in a window of {size}")
    # Seeded offset rather than a fixed centre, so gold never sits at a
    # predictable position the model could learn instead of reading the log.
    start = lo - rng.randint(0, size - span)
    start = max(0, min(start, len(passages) - size))
    return passages[start : start + size]


def _make_tx_window(size: int) -> type:
    class TxWindow:
        name = f"tx_window_{size}"
        window = size

        def build(self, ex: Example, rng: random.Random) -> Prompt:
            return build_txlog_prompt(ex, tx_window(ex, rng, self.window), self.name)

    TxWindow.__name__ = f"TxWindow{size}"
    TxWindow.__qualname__ = TxWindow.__name__
    TxWindow.__doc__ = f"Transaction-log window of {size} lines around the anomaly."
    return register_arrangement(TxWindow)


TX_WINDOW_ARRANGEMENTS = tuple(_make_tx_window(size) for size in TX_WINDOWS)
