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
def test_rendered_lines_cost_what_the_paper_figure_costs() -> None:
    """Our line format must be as expensive as the paper's own.

    Note what this does *not* assert. Table 2 puts 25 transactions at 512
    tokens and 500 at 9,560 -- about 19 tokens per line. But the line format
    printed in Figure 7 costs 37 tokens under Qwen3's own tokenizer, so
    Table 2's token axis cannot be produced by the format Figure 7 shows.
    Rather than reverse-engineer a terser encoding to hit a number, we match
    the format in the figure and match Figure 1(b)'s x-axis, which is the
    transaction count. Measured token counts are logged per record in
    `usage.input_tokens`, so the real axis is always recoverable.

    Opt-in because it needs a tokenizer download.
    """
    from transformers import AutoTokenizer

    from ctxlab.arrangements.filtering import tx_window

    tokenizer = AutoTokenizer.from_pretrained(os.environ["CTXLAB_TOKENIZER"])
    figure_7_line = "[TX001]: Transfer $107: A=4000 \u2192 3893, B=4200 \u2192 4307"
    reference = len(tokenizer(figure_7_line)["input_ids"])

    example = load_txlog(DatasetConfig(name="txlog", n=1, seed=0))[0]
    ours = len(tokenizer(example.passages[0].text)["input_ids"])
    assert abs(ours - reference) <= 2, (ours, reference)

    counts = {}
    for size in (25, 95, 250, 500):
        window = tx_window(example, random.Random(0), size)
        counts[size] = len(tokenizer(render_txlog(example.question, window))["input_ids"])
    print(f"\nline={ours} tok (figure 7: {reference}); window token counts: {counts}")
    assert counts[25] < counts[95] < counts[250] < counts[500]
    # Per-line density holds across the sweep, so length scales with the
    # transaction count rather than with anything the generator drifts into.
    density = [(counts[b] - counts[a]) / (b - a) for a, b in ((25, 95), (95, 250), (250, 500))]
    assert all(abs(d - reference) <= 4 for d in density), density


# --- chance baseline ----------------------------------------------------------


def _random_arm_scores(size: int, n: int = 60) -> dict[str, float]:
    from ctxlab.config import ModelConfig
    from ctxlab.models.mock import MockModel
    from ctxlab.registry import get_arrangement, load_plugins

    load_plugins()
    model = MockModel(ModelConfig(name="chance", kind="mock", extra={"behavior": "txlog_random"}))
    arrangement = get_arrangement(f"tx_window_{size}")
    metrics = {"tx_id": TxIdAccuracy(), "tx_type": TxTypeAccuracy(), "tx_joint": TxJointAccuracy()}
    totals = dict.fromkeys(metrics, 0.0)
    for example in load_txlog(DatasetConfig(name="txlog", n=n, seed=1)):
        prompt = arrangement.build(example, random.Random(example.uid))
        text = model.generate(prompt).text
        for name, metric in metrics.items():
            totals[name] += metric.score(text, example.answers)
    return {name: total / n for name, total in totals.items()}


def test_random_arm_tracks_the_theoretical_chance_rate() -> None:
    """The baseline has to actually be chance, or it cannot calibrate anything.

    Guessing a transaction is right about 1/window of the time and a bug type
    about 1/4, so joint accuracy is about 1/(4*window). The point of measuring
    it is that this moves by 20x across the sweep, which means some of the
    accuracy drop the paper reports is the guess getting harder rather than
    attention diluting.
    """
    for size in (25, 95):
        scores = _random_arm_scores(size)
        assert scores["tx_id"] == pytest.approx(1 / size, abs=0.06)
        assert scores["tx_type"] == pytest.approx(0.25, abs=0.15)
        assert scores["tx_joint"] == pytest.approx(1 / (4 * size), abs=0.05)


def test_random_arm_is_deterministic() -> None:
    assert _random_arm_scores(25, n=20) == _random_arm_scores(25, n=20)
