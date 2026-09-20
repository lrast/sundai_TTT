"""Bug-localization metrics beyond exact answers.

`mention` separates "found it" from "solved it": it scores 1.0 when the gold
answer appears anywhere in the completion after SQuAD-style normalization,
so a model that names the right function inside a longer reply still counts
as a retrieval hit even when `em` scores 0.
"""

from __future__ import annotations

from ctxlab.metrics.qa import normalize_answer
from ctxlab.registry import register_metric


@register_metric
class Mention:
    name = "mention"

    def score(self, prediction: str, golds: list[str]) -> float:
        # normalize_answer keeps only the first line; mention scans the whole reply.
        pred = normalize_answer(prediction.replace("\n", " "))
        if not pred:
            return 0.0
        for gold in golds:
            norm = normalize_answer(gold)
            if norm and norm in pred:
                return 1.0
        return 0.0


@register_metric
class GoldRecall:
    """Fraction of gold answers mentioned anywhere in the completion.

    Equals `mention` for single-needle tasks; for multi-needle bug
    localization (several changed files) it is the file-set recall.
    """

    name = "gold_recall"

    def score(self, prediction: str, golds: list[str]) -> float:
        pred = normalize_answer(prediction.replace("\n", " "))
        if not pred or not golds:
            return 0.0
        hits = sum(1 for g in golds if normalize_answer(g) and normalize_answer(g) in pred)
        return hits / len(golds)
