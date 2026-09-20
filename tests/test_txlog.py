"""Transaction-log generator, answer parsing, and scoring."""

from __future__ import annotations

import json
import os
import random

import pytest

from ctxlab.arrangements.formatting import render_txlog
from ctxlab.config import DatasetConfig
from ctxlab.data.txlog import (
    BUG_TYPES,
    canonical_answer,
    generate_log,
    load_txlog,
    parse_tx_line,
    validate_log,
)
from ctxlab.metrics.txlog import TxIdAccuracy, TxJointAccuracy, TxTypeAccuracy, parse_answer

SEEDS = range(12)


@pytest.mark.parametrize("bug_type", BUG_TYPES)
@pytest.mark.parametrize("seed", SEEDS)
def test_exactly_one_violation_of_the_labelled_type(bug_type: str, seed: int) -> None:
    """The whole task rests on this: one anomaly, of the labelled type, at the
    labelled id. `validate_log` re-derives state from the rendered lines, so it
    is an independent check on the injection code rather than a restatement."""
    rng = random.Random(f"test|{bug_type}|{seed}")
    transactions, _, gold_tx_id = generate_log(rng, bug_type)
    violations = validate_log(transactions)
    assert len(violations) == 1, violations
    assert violations[0].bug_type == bug_type
    assert violations[0].tx_id == gold_tx_id


@pytest.mark.parametrize("bug_type", BUG_TYPES)
def test_transaction_ids_never_skip(bug_type: str) -> None:
    """A hole in the id sequence would be a second, unintended tell."""
    rng = random.Random(f"ids|{bug_type}")
    transactions, _, _ = generate_log(rng, bug_type)
    numbers = sorted({int(tx.tx_id[2:]) for tx in transactions})
    assert numbers == list(range(1, len(numbers) + 1))


def test_duplicate_is_an_insertion_not_a_substitution() -> None:
    rng = random.Random("dup")
    transactions, gold_indices, gold_tx_id = generate_log(rng, "DUPLICATE_TXN")
    ids = [tx.tx_id for tx in transactions]
    assert len(ids) - len(set(ids)) == 1
    assert [ids[i] for i in gold_indices] == [gold_tx_id, gold_tx_id]
    assert len(gold_indices) == 2


@pytest.mark.parametrize("bug_type", BUG_TYPES)
def test_render_round_trips_through_parse(bug_type: str) -> None:
    rng = random.Random(f"round|{bug_type}")
    transactions, _, _ = generate_log(rng, bug_type, n_transactions=40, bug_index=20)
    for tx in transactions:
        assert parse_tx_line(tx.render()) == tx


def test_parse_tx_line_rejects_non_lines() -> None:
    assert parse_tx_line("Rules:") is None
    assert parse_tx_line("") is None
    assert parse_tx_line("[TX001]: Transfer $10: A=1 -> 2") is None


@pytest.mark.parametrize("bug_type", BUG_TYPES)
def test_rendered_window_is_solvable_on_its_own(bug_type: str) -> None:
    """A window has no access to the log's true opening state, so the header is
    derived from its first line. If that derivation were wrong the log would be
    unverifiable and the whole length sweep would measure nothing."""
    rng = random.Random(f"solve|{bug_type}")
    transactions, gold_indices, gold_tx_id = generate_log(
        rng, bug_type, n_transactions=60, bug_index=30
    )
    from ctxlab.data.base import Passage

    passages = [
        Passage(title=tx.tx_id, text=tx.render(), is_gold=i in set(gold_indices))
        for i, tx in enumerate(transactions)
    ]
    window = passages[25:40]
    body = render_txlog("Please identify the bug type and location.", window)
    opening = json.loads(body.split("Initial state: ")[1].splitlines()[0])
    first = parse_tx_line(window[0].text)
    assert first is not None
    assert opening["account_A"] + opening["account_B"] == opening["total"]
    assert opening[f"account_{first.src}"] == first.src_old
    assert opening[f"account_{first.dst}"] == first.dst_old

    recovered = [tx for tx in (parse_tx_line(ln) for ln in body.splitlines()) if tx is not None]
    found = validate_log(recovered)
    assert len(found) == 1
    assert found[0].bug_type == bug_type
    assert found[0].tx_id == gold_tx_id


def test_loader_is_deterministic_and_covers_every_bug_type() -> None:
    cfg = DatasetConfig(name="txlog", n=8, seed=0)
    first = load_txlog(cfg)
    second = load_txlog(cfg)
    assert [e.uid for e in first] == [e.uid for e in second]
    assert [e.answers for e in first] == [e.answers for e in second]
    types = {parse_answer(e.answers[0]).bug_type for e in first}
    assert types == set(BUG_TYPES)
    for example in first:
        assert any(p.is_gold for p in example.passages)


def test_loader_respects_a_bug_type_preset() -> None:
    cfg = DatasetConfig(name="txlog", n=3, seed=0, config="NEGATIVE_BAL")
    assert {parse_answer(e.answers[0]).bug_type for e in load_txlog(cfg)} == {"NEGATIVE_BAL"}
    with pytest.raises(ValueError, match="Unknown bug type"):
        load_txlog(DatasetConfig(name="txlog", n=1, seed=0, config="NOT_A_BUG"))


# --- answer parsing / scoring -------------------------------------------------

GOLD = canonical_answer("NEGATIVE_BAL", "TX004")


@pytest.mark.parametrize(
    ("prediction", "bug_type", "tx_id"),
    [
        (GOLD, "NEGATIVE_BAL", "TX4"),
        # Figure 7 prints the pair unquoted, which is not valid JSON.
        ('{"bug_type": NEGATIVE_BAL, "bug_location": TX004}', "NEGATIVE_BAL", "TX4"),
        ("negative_bal at tx 4", "NEGATIVE_BAL", "TX4"),
        # A model that reasons before concluding: the last mention wins.
        (
            "TX002 looks fine, TX003 fine. CALC_ERROR? No. NEGATIVE_BAL at TX004",
            "NEGATIVE_BAL",
            "TX4",
        ),
        ("", None, None),
        ("unknown", None, None),
        ("I could not determine the bug", None, None),
    ],
)
def test_parse_answer(prediction: str, bug_type: str | None, tx_id: str | None) -> None:
    parsed = parse_answer(prediction)
    assert parsed.bug_type == bug_type
    assert parsed.tx_id == tx_id


def test_zero_padding_does_not_change_the_transaction() -> None:
    assert parse_answer("TX004").tx_id == parse_answer("TX4").tx_id


def test_scoring_splits_the_two_halves() -> None:
    joint, kind, where = TxJointAccuracy(), TxTypeAccuracy(), TxIdAccuracy()
    assert joint.score(GOLD, [GOLD]) == 1.0
    # Right type, wrong transaction: the retrieval half is what failed.
    half = canonical_answer("NEGATIVE_BAL", "TX999")
    assert kind.score(half, [GOLD]) == 1.0
    assert where.score(half, [GOLD]) == 0.0
    assert joint.score(half, [GOLD]) == 0.0
    # Right transaction, wrong type.
    other = canonical_answer("CALC_ERROR", "TX004")
    assert kind.score(other, [GOLD]) == 0.0
    assert where.score(other, [GOLD]) == 1.0
    assert joint.score(other, [GOLD]) == 0.0


def test_scoring_handles_empty_and_missing_golds() -> None:
    joint = TxJointAccuracy()
    assert joint.score("", [GOLD]) == 0.0
    assert joint.score(GOLD, []) == 0.0


def test_metrics_are_registered() -> None:
    from ctxlab.registry import METRICS, load_plugins

    load_plugins()
    for name in ("tx_joint", "tx_type", "tx_id"):
        assert name in METRICS


@pytest.mark.skipif(
    not os.environ.get("CTXLAB_TOKENIZER"),
    reason="set CTXLAB_TOKENIZER=Qwen/Qwen3-4B to check context-length calibration",
)
def test_window_token_counts_match_the_paper() -> None:
    """The x-axis is only comparable to the paper's Table 2 if the windows land
    near its token counts. Opt-in because it needs a tokenizer download."""
    from transformers import AutoTokenizer

    from ctxlab.arrangements.filtering import tx_window

    tokenizer = AutoTokenizer.from_pretrained(os.environ["CTXLAB_TOKENIZER"])
    example = load_txlog(DatasetConfig(name="txlog", n=1, seed=0))[0]
    counts = {}
    for size in (25, 95, 250, 500):
        window = tx_window(example, random.Random(0), size)
        body = render_txlog(example.question, window)
        counts[size] = len(tokenizer(body)["input_ids"])
    print(f"\ntoken counts by window: {counts}")
    assert counts[25] < counts[95] < counts[250] < counts[500]
    assert 300 < counts[25] < 1200
    assert 6000 < counts[500] < 20000
