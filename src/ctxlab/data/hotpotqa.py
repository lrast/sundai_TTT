"""HotpotQA distractor loader.

Each validation example has 10 Wikipedia paragraphs: typically 2 gold (titles
listed in `supporting_facts`) plus 8 hard distractors. That is the knob set
context-arrangement experiments need without standing up a retriever.
"""

from __future__ import annotations

from ctxlab.config import DatasetConfig
from ctxlab.data.base import Example, Passage
from ctxlab.registry import register_dataset


def _titles_of(column: object) -> list[str]:
    if isinstance(column, dict):
        return list(column.get("title") or [])
    return list(column)  # type: ignore[arg-type]


def _row_to_example(row: dict) -> Example:
    supporting = set(_titles_of(row["supporting_facts"]))
    context = row["context"]
    titles = list(context["title"])
    sentence_lists = list(context["sentences"])
    passages: list[Passage] = []
    for title, sentences in zip(titles, sentence_lists, strict=True):
        text = " ".join(sentences)
        passages.append(Passage(title=title, text=text, is_gold=title in supporting))
    answer = row["answer"]
    answers = [answer] if isinstance(answer, str) else list(answer)
    return Example(
        uid=str(row["id"]),
        question=row["question"],
        answers=answers,
        passages=passages,
    )


@register_dataset("hotpotqa")
def load_hotpotqa(cfg: DatasetConfig) -> list[Example]:
    from datasets import load_dataset

    hub_id = cfg.hub_id or "hotpotqa/hotpot_qa"
    config_name = cfg.config or "distractor"
    ds = load_dataset(
        hub_id,
        config_name,
        split=cfg.split,
        cache_dir=cfg.cache_dir,
    )
    if cfg.n is not None:
        n = min(cfg.n, len(ds))
        ds = ds.shuffle(seed=cfg.seed).select(range(n))
    return [_row_to_example(row) for row in ds]


def examples_from_rows(rows: list[dict]) -> list[Example]:
    """Build examples from already-materialized HotpotQA-shaped dicts (tests)."""
    return [_row_to_example(row) for row in rows]
