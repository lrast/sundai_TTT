"""SQuAD-style normalized exact match and token F1."""

from __future__ import annotations

import re
import string
from collections import Counter
from typing import Protocol

from ctxlab.registry import register_metric

ARTICLES = re.compile(r"\b(a|an|the)\b", re.IGNORECASE)


class Metric(Protocol):
    name: str

    def score(self, prediction: str, golds: list[str]) -> float: ...


def normalize_answer(text: str) -> str:
    text = text.strip().splitlines()[0] if text.strip() else text
    text = text.lower()
    text = "".join(ch for ch in text if ch not in string.punctuation)
    text = ARTICLES.sub(" ", text)
    return " ".join(text.split())


def _token_f1(pred: str, gold: str) -> float:
    pred_toks = normalize_answer(pred).split()
    gold_toks = normalize_answer(gold).split()
    if not pred_toks and not gold_toks:
        return 1.0
    if not pred_toks or not gold_toks:
        return 0.0
    overlap = Counter(pred_toks) & Counter(gold_toks)
    n_same = sum(overlap.values())
    if n_same == 0:
        return 0.0
    precision = n_same / len(pred_toks)
    recall = n_same / len(gold_toks)
    return 2 * precision * recall / (precision + recall)


@register_metric
class ExactMatch:
    name = "em"

    def score(self, prediction: str, golds: list[str]) -> float:
        norm_pred = normalize_answer(prediction)
        return float(any(norm_pred == normalize_answer(g) for g in golds))


@register_metric
class TokenF1:
    name = "f1"

    def score(self, prediction: str, golds: list[str]) -> float:
        if not golds:
            return 0.0
        return max(_token_f1(prediction, g) for g in golds)
