"""Scoring for the transaction-log task: a `{bug_type, TX_ID}` pair.

SQuAD-style EM/F1 cannot score this. `metrics.qa.normalize_answer` keeps only
the first line and strips all punctuation, so the reference answer
`{"bug_type": "NEGATIVE_BAL", "bug_location": "TX004"}` collapses to a bag of
words in which a half-right answer scores partial credit and a right answer in
a different format scores zero. The paper reports a single accuracy over the
pair, which is `tx_joint` here; `tx_type` and `tx_id` are the same prediction
broken apart, which is what tells you *how* a run is failing.

Parsing is deliberately tolerant. A model that answers correctly but wraps the
pair in prose should score as correct -- otherwise the sweep measures format
compliance, which collapses with context length for its own reasons, and the
result would not be about score dilution at all.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from ctxlab.data.txlog import BUG_TYPES
from ctxlab.registry import register_metric

_BUG_TYPE_ALT = "|".join(BUG_TYPES)
_BUG_TYPE_RE = re.compile(_BUG_TYPE_ALT, re.IGNORECASE)
_TX_ID_RE = re.compile(r"TX\s*(\d+)", re.IGNORECASE)


def _keyed(text: str, key: str, value_pattern: str) -> str | None:
    """Value sitting next to `key`, whether or not the JSON is well formed.

    Figure 7 prints unquoted values (`{"bug_type": NEGATIVE_BAL, ...}`), so a
    strict `json.loads` is not usable as the primary path.
    """
    match = re.search(
        rf"[\"']?{key}[\"']?\s*[:=]\s*[\"']?({value_pattern})[\"']?",
        text,
        re.IGNORECASE,
    )
    return match.group(1) if match else None


def _normalize_tx_id(raw: str | None) -> str | None:
    """`TX004`, `tx 4` and `TX4` all name the same transaction."""
    if raw is None:
        return None
    match = _TX_ID_RE.search(raw)
    if match is None:
        return None
    return f"TX{int(match.group(1))}"


@dataclass(frozen=True)
class TxAnswer:
    bug_type: str | None
    tx_id: str | None


def parse_answer(text: str) -> TxAnswer:
    """Pull the pair out of a model completion or a reference answer.

    Prefers values labelled with their key; otherwise falls back to the *last*
    mention in the text, which is the right guess for a model that reasons
    before it concludes.
    """
    if not text or not text.strip():
        return TxAnswer(None, None)

    bug_type = _keyed(text, "bug_type", _BUG_TYPE_ALT)
    if bug_type is None:
        found = _BUG_TYPE_RE.findall(text)
        bug_type = found[-1] if found else None

    tx_id = _normalize_tx_id(_keyed(text, "bug_location", r"TX\s*\d+"))
    if tx_id is None:
        found_ids = _TX_ID_RE.findall(text)
        tx_id = f"TX{int(found_ids[-1])}" if found_ids else None

    return TxAnswer(bug_type.upper() if bug_type else None, tx_id)


def _best(prediction: str, golds: list[str], score: object) -> float:
    if not golds:
        return 0.0
    pred = parse_answer(prediction)
    return max(float(score(pred, parse_answer(g))) for g in golds)  # type: ignore[operator]


def _type_ok(pred: TxAnswer, gold: TxAnswer) -> bool:
    return pred.bug_type is not None and pred.bug_type == gold.bug_type


def _id_ok(pred: TxAnswer, gold: TxAnswer) -> bool:
    return pred.tx_id is not None and pred.tx_id == gold.tx_id


@register_metric
class TxJointAccuracy:
    """Both halves right. This is the paper's reported accuracy."""

    name = "tx_joint"

    def score(self, prediction: str, golds: list[str]) -> float:
        return _best(prediction, golds, lambda p, g: _type_ok(p, g) and _id_ok(p, g))


@register_metric
class TxTypeAccuracy:
    name = "tx_type"

    def score(self, prediction: str, golds: list[str]) -> float:
        return _best(prediction, golds, _type_ok)


@register_metric
class TxIdAccuracy:
    """Localisation alone -- the retrieval half of the task."""

    name = "tx_id"

    def score(self, prediction: str, golds: list[str]) -> float:
        return _best(prediction, golds, _id_ok)
