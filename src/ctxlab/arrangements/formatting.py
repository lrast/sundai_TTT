"""Passage -> prompt text. Keep formatting orthogonal to ordering/filtering."""

from __future__ import annotations

from ctxlab.arrangements.base import DEFAULT_SYSTEM
from ctxlab.data.base import Example, Message, Passage, Prompt


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
) -> Prompt:
    """Assemble a Prompt and truthful metadata for post-hoc analysis."""
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
        messages=(Message(role="user", content=make_user_message(ex.question, passages)),),
        meta=meta,
    )
