"""Passage -> prompt text. Keep formatting orthogonal to ordering/filtering."""

from __future__ import annotations

import json
from collections.abc import Callable

from ctxlab.arrangements.base import DEFAULT_SYSTEM
from ctxlab.data.base import Example, Message, Passage, Prompt
from ctxlab.data.txlog import (
    BUG_TYPE_DESCRIPTIONS,
    RULES,
    parse_tx_line,
)

TASK_DESCRIPTION = "Analyze this banking transaction log for bugs."
ANSWER_FORMAT = (
    "Respond with only a JSON object of the form "
    '{"bug_type": "<BUG_TYPE>", "bug_location": "<TX_ID>"}.'
)


def format_passages(passages: list[Passage]) -> str:
    blocks: list[str] = []
    for i, passage in enumerate(passages, start=1):
        blocks.append(f"[{i}] {passage.title}\n{passage.text}")
    return "\n\n".join(blocks)


def make_user_message(question: str, passages: list[Passage]) -> str:
    if not passages:
        return f"Question: {question}\nAnswer:"
    return f"Context:\n{format_passages(passages)}\n\nQuestion: {question}\nAnswer:"


def passage_key(passage: Passage) -> tuple[str, str]:
    return (passage.title, passage.text)


def build_prompt(
    ex: Example,
    passages: list[Passage],
    arrangement: str,
    *,
    system: str = DEFAULT_SYSTEM,
    render: Callable[[str, list[Passage]], str] = make_user_message,
) -> Prompt:
    """Assemble a Prompt and truthful metadata for post-hoc analysis.

    `render` swaps the markup without touching the metadata contract, so every
    arrangement -- QA or transaction-log -- reports `gold_positions`,
    `kept_gold_titles` and `dropped_titles` from the same code path.
    """
    kept = {passage_key(p) for p in passages}
    dropped_titles = [p.title for p in ex.passages if passage_key(p) not in kept]
    gold_positions = [i for i, p in enumerate(passages) if p.is_gold]
    meta = {
        "arrangement": arrangement,
        "uid": ex.uid,
        "question": ex.question,
        "n_passages": len(passages),
        "n_gold": sum(1 for p in passages if p.is_gold),
        "n_gold_available": sum(1 for p in ex.passages if p.is_gold),
        "gold_positions": gold_positions,
        "passage_titles": [p.title for p in passages],
        "kept_gold_titles": [p.title for p in passages if p.is_gold],
        "dropped_titles": dropped_titles,
    }
    return Prompt(
        system=system,
        messages=(Message(role="user", content=render(ex.question, passages)),),
        meta=meta,
    )


def render_txlog(question: str, passages: list[Passage]) -> str:
    """Render a transaction-log window in the layout of the paper's Figure 7.

    The `Initial state:` header is *derived* from the first kept line's old
    balances rather than carried on the Example. That is what makes a window
    self-describing: every line prints both accounts' old -> new balances, so
    whichever line a window starts on states the balances the window opens
    with. Dropping the derivation would make a truncated log unverifiable.
    """
    if not passages:
        return f"{TASK_DESCRIPTION}\n\n(no transaction log provided)\n\n{question}"
    first = parse_tx_line(passages[0].text)
    if first is None:
        raise ValueError(f"first passage is not a transaction line: {passages[0].text!r}")
    opening = first.opening_balances()
    state = {f"account_{name}": opening[name] for name in sorted(opening)}
    state["total"] = sum(opening.values())
    rules = "\n".join(f"{i}. {rule}" for i, rule in enumerate(RULES, start=1))
    bug_types = "\n".join(
        f"- {name}: {description}" for name, description in BUG_TYPE_DESCRIPTIONS.items()
    )
    log = "\n".join(p.text for p in passages)
    return (
        f"{TASK_DESCRIPTION}\n\n"
        f"Initial state: {json.dumps(state)}\n\n"
        f"Rules:\n{rules}\n\n"
        f"Transaction logs:\n{log}\n\n"
        f"Possible bug types (choose exactly one):\n{bug_types}\n\n"
        f"{question}\n"
        f"{ANSWER_FORMAT}"
    )


def build_txlog_prompt(
    ex: Example,
    passages: list[Passage],
    arrangement: str,
    *,
    system: str = DEFAULT_SYSTEM,
) -> Prompt:
    """`build_prompt` with Figure 7 markup. Metadata contract is unchanged."""
    return build_prompt(ex, passages, arrangement, system=system, render=render_txlog)
