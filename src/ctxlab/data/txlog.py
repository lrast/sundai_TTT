"""Synthetic banking transaction logs carrying exactly one injected anomaly.

Replicates the "Error in a Log of Transactions" task from *Let's (not) just put
things in Context* (arXiv 2512.13898), Section 2.1 and Figure 7.

Two accounts, deliberately. Every line prints *both* old -> new balances, so a
line is a complete state transition and any contiguous run of lines is a
self-describing log. That is what lets the `tx_window_*` arrangements slice a
window out of the middle and still hand the model something it can verify (see
`ctxlab.arrangements.formatting.build_txlog_prompt`).

Context length is **not** a knob here. The loader always emits one long log per
example; the arrangements choose how much of it the model sees. That is how the
paper sweeps length: the needle stays fixed and the haystack grows around it.
"""

from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass, replace

from ctxlab.config import DatasetConfig
from ctxlab.data.base import Example, Passage
from ctxlab.registry import register_dataset

BUG_TYPES = ("CALC_ERROR", "NEGATIVE_BAL", "LOST_UPDATE", "DUPLICATE_TXN")

# One long log per example. The widest window is 500, so the anomaly has to sit
# at least 250 lines from either end for a centred window to fit.
N_TRANSACTIONS = 600
BUG_INDEX_CENTER = 300
BUG_INDEX_JITTER = 25

MIN_AMOUNT = 50
MAX_AMOUNT = 900
# A sender never moves more than a third of its balance, so valid lines can
# never trip the non-negative rule by accident.
AMOUNT_DIVISOR = 3
MIN_SENDER_BALANCE = MIN_AMOUNT * AMOUNT_DIVISOR

TARGET_INSTRUCTION = "Please identify the bug type and location."

# The task's label space and rule set. The data module owns these because they
# define the task; `arrangements.formatting` owns only how they are rendered.
BUG_TYPE_DESCRIPTIONS = {
    "CALC_ERROR": "Mathematical calculation is incorrect",
    "NEGATIVE_BAL": "Account balance becomes negative",
    "LOST_UPDATE": "Concurrent update causes lost transaction",
    "DUPLICATE_TXN": "Same transaction processed multiple times",
}

RULES = (
    "Total money must remain constant (conservation)",
    "No account can go negative",
    "All calculations must be mathematically correct",
)

TX_LINE_RE = re.compile(
    r"^\[(?P<tx_id>TX\d+)\]: Transfer \$(?P<amount>\d+): "
    r"(?P<src>[A-Z])=(?P<src_old>-?\d+) -> (?P<src_new>-?\d+), "
    r"(?P<dst>[A-Z])=(?P<dst_old>-?\d+) -> (?P<dst_new>-?\d+)$"
)


@dataclass(frozen=True)
class Tx:
    """One transaction line. `src`/`dst` are single-letter account names."""

    tx_id: str
    amount: int
    src: str
    src_old: int
    src_new: int
    dst: str
    dst_old: int
    dst_new: int

    def render(self) -> str:
        return (
            f"[{self.tx_id}]: Transfer ${self.amount}: "
            f"{self.src}={self.src_old} -> {self.src_new}, "
            f"{self.dst}={self.dst_old} -> {self.dst_new}"
        )

    def opening_balances(self) -> dict[str, int]:
        """State the log is in immediately *before* this line."""
        return {self.src: self.src_old, self.dst: self.dst_old}


def parse_tx_line(line: str) -> Tx | None:
    """Parse a rendered line back into a `Tx`, or `None` if it is not one.

    Single source of truth: the formatter, the validator and the tests all go
    through here rather than re-deriving the format.
    """
    match = TX_LINE_RE.match(line.strip())
    if match is None:
        return None
    g = match.groupdict()
    return Tx(
        tx_id=g["tx_id"],
        amount=int(g["amount"]),
        src=g["src"],
        src_old=int(g["src_old"]),
        src_new=int(g["src_new"]),
        dst=g["dst"],
        dst_old=int(g["dst_old"]),
        dst_new=int(g["dst_new"]),
    )


def canonical_answer(bug_type: str, tx_id: str) -> str:
    """The reference answer string, matching Figure 7's model output."""
    return json.dumps({"bug_type": bug_type, "bug_location": tx_id})


def _valid_tx(rng: random.Random, tx_id: str, balances: dict[str, int]) -> Tx:
    """A transfer that violates none of the three rules.

    When one account is below the floor -- which happens for exactly one line,
    right after a NEGATIVE_BAL injection -- this pulls it back above zero in a
    single move, the way Figure 7's TX005 does. Climbing back over several
    lines would leave the account negative in between and manufacture a second
    rule violation, breaking the one-anomaly-per-log guarantee.
    """
    accounts = sorted(balances)
    rich, poor = sorted(accounts, key=lambda a: -balances[a])
    if balances[poor] < MIN_SENDER_BALANCE:
        src, dst = rich, poor
        floor = MIN_SENDER_BALANCE - balances[dst]
        ceiling = balances[src] - MIN_SENDER_BALANCE
        if ceiling < floor:
            raise ValueError(f"cannot restore {dst} from {balances}: total too low")
        amount = min(ceiling, floor + rng.randint(200, 900))
    else:
        src = rng.choice(accounts)
        dst = next(a for a in accounts if a != src)
        ceiling = max(MIN_AMOUNT, min(balances[src] // AMOUNT_DIVISOR, MAX_AMOUNT))
        amount = rng.randint(MIN_AMOUNT, ceiling)
    return Tx(
        tx_id=tx_id,
        amount=amount,
        src=src,
        src_old=balances[src],
        src_new=balances[src] - amount,
        dst=dst,
        dst_old=balances[dst],
        dst_new=balances[dst] + amount,
    )


def _apply(balances: dict[str, int], tx: Tx) -> None:
    """Advance state to the balances the line *claims*, bug or not.

    Figure 7's TX005 continues from the anomalous `A=-16`, so the log stays
    internally consistent after the injection point.
    """
    balances[tx.src] = tx.src_new
    balances[tx.dst] = tx.dst_new


def _inject_calc_error(rng: random.Random, tx: Tx) -> Tx:
    """Receiver's arithmetic is wrong: dst_new != dst_old + amount."""
    delta = rng.choice((-500, -250, -100, 100, 250, 500))
    return replace(tx, dst_new=tx.dst_new + delta)


def _inject_negative_bal(rng: random.Random, tx: Tx, balances: dict[str, int]) -> Tx:
    """Sender overdraws. Arithmetic stays correct; the balance goes negative."""
    overdraft = rng.randint(50, 900)
    amount = balances[tx.src] + overdraft
    return Tx(
        tx_id=tx.tx_id,
        amount=amount,
        src=tx.src,
        src_old=balances[tx.src],
        src_new=-overdraft,
        dst=tx.dst,
        dst_old=balances[tx.dst],
        dst_new=balances[tx.dst] + amount,
    )


def _inject_lost_update(tx: Tx, stale_src_balance: int) -> Tx:
    """Stale write: `src_old` is the balance from two lines back, so the
    intervening commit is silently overwritten."""
    amount = min(tx.amount, max(MIN_AMOUNT, stale_src_balance // AMOUNT_DIVISOR))
    return Tx(
        tx_id=tx.tx_id,
        amount=amount,
        src=tx.src,
        src_old=stale_src_balance,
        src_new=stale_src_balance - amount,
        dst=tx.dst,
        dst_old=tx.dst_old,
        dst_new=tx.dst_old + amount,
    )


def _inject_duplicate(original: Tx, balances: dict[str, int]) -> Tx:
    """The same transaction id is committed a second time.

    Balances chain correctly, so the only evidence is the repeated TX id --
    which is exactly "same transaction processed multiple times".
    """
    return Tx(
        tx_id=original.tx_id,
        amount=original.amount,
        src=original.src,
        src_old=balances[original.src],
        src_new=balances[original.src] - original.amount,
        dst=original.dst,
        dst_old=balances[original.dst],
        dst_new=balances[original.dst] + original.amount,
    )


def generate_log(
    rng: random.Random,
    bug_type: str,
    *,
    n_transactions: int = N_TRANSACTIONS,
    bug_index: int | None = None,
) -> tuple[list[Tx], list[int], str]:
    """Build one log.

    Returns `(transactions, gold_indices, gold_tx_id)`. `gold_indices` holds two
    entries for DUPLICATE_TXN (the original commit and its replay) and one
    otherwise.

    The replayed commit is *inserted*, not substituted: a real system that
    processes a transaction twice still assigns fresh ids to everything after
    it. Substituting would leave a hole in the id sequence, which is a second,
    unintended tell -- the model could find the gap instead of the duplicate.
    """
    if bug_type not in BUG_TYPES:
        raise ValueError(f"Unknown bug type {bug_type!r}. Known: {', '.join(BUG_TYPES)}")
    if bug_index is None:
        bug_index = BUG_INDEX_CENTER + rng.randint(-BUG_INDEX_JITTER, BUG_INDEX_JITTER)

    balances = {"A": rng.randrange(3000, 6000, 100), "B": rng.randrange(3000, 6000, 100)}
    transactions: list[Tx] = []
    # Balance each account held one line earlier, for the stale LOST_UPDATE write.
    previous_balances = dict(balances)
    gold_indices: list[int] = []
    gold_tx_id = ""

    def commit(tx: Tx) -> None:
        nonlocal previous_balances
        previous_balances = dict(balances)
        _apply(balances, tx)
        transactions.append(tx)

    for i in range(n_transactions):
        tx_id = f"TX{i + 1:03d}"
        if i == bug_index and bug_type == "DUPLICATE_TXN":
            original_index = len(transactions) - rng.randint(1, 3)
            duplicate = _inject_duplicate(transactions[original_index], balances)
            gold_indices.extend((original_index, len(transactions)))
            gold_tx_id = duplicate.tx_id
            commit(duplicate)
        tx = _valid_tx(rng, tx_id, balances)
        if i == bug_index and bug_type != "DUPLICATE_TXN":
            stale = dict(previous_balances)
            if bug_type == "CALC_ERROR":
                tx = _inject_calc_error(rng, tx)
            elif bug_type == "NEGATIVE_BAL":
                tx = _inject_negative_bal(rng, tx, balances)
            else:
                tx = _inject_lost_update(tx, stale[tx.src])
            gold_indices.append(len(transactions))
            gold_tx_id = tx.tx_id
        commit(tx)

    return transactions, sorted(gold_indices), gold_tx_id


@register_dataset("txlog")
def load_txlog(cfg: DatasetConfig) -> list[Example]:
    """Generate `cfg.n` transaction-log examples.

    Only existing `DatasetConfig` fields are used -- `n`, `seed` and `config`
    (a bug-type preset). Unknown YAML keys under `dataset:` are dropped
    silently by pydantic, so generator knobs must not be invented there.
    """
    n = cfg.n if cfg.n is not None else 100
    bug_types = _bug_types_for(cfg.config)
    examples: list[Example] = []
    for i in range(n):
        rng = random.Random(f"{cfg.seed}|txlog|{i}")
        bug_type = bug_types[i % len(bug_types)]
        transactions, gold_indices, gold_tx_id = generate_log(rng, bug_type)
        gold = set(gold_indices)
        passages = [
            Passage(title=tx.tx_id, text=tx.render(), is_gold=j in gold)
            for j, tx in enumerate(transactions)
        ]
        examples.append(
            Example(
                uid=f"txlog-{bug_type.lower()}-{i:04d}",
                question=TARGET_INSTRUCTION,
                answers=[canonical_answer(bug_type, gold_tx_id)],
                passages=passages,
            )
        )
    return examples


def _bug_types_for(preset: str | None) -> tuple[str, ...]:
    if preset in (None, "all", "all_bugs"):
        return BUG_TYPES
    requested = tuple(part.strip().upper() for part in preset.split(",") if part.strip())
    unknown = [b for b in requested if b not in BUG_TYPES]
    if unknown:
        raise ValueError(f"Unknown bug type(s) {unknown}. Known: {', '.join(BUG_TYPES)}")
    return requested


@dataclass(frozen=True)
class Violation:
    index: int
    tx_id: str
    bug_type: str


def validate_log(transactions: list[Tx]) -> list[Violation]:
    """Replay a log and report every rule violation it contains.

    The generator promises exactly one. This is what checks that promise, and
    it is deliberately independent of the injection code: it re-derives state
    from the lines themselves, so a generator bug shows up as an extra (or
    missing, or misclassified) violation rather than passing silently.

    The checks are ordered so each injected anomaly trips exactly one of them:
    a stale write is reported as LOST_UPDATE rather than as the arithmetic
    error it also superficially resembles.
    """
    if not transactions:
        return []
    balances = transactions[0].opening_balances()
    seen: set[str] = set()
    violations: list[Violation] = []
    for i, tx in enumerate(transactions):
        if tx.tx_id in seen:
            violations.append(Violation(i, tx.tx_id, "DUPLICATE_TXN"))
        seen.add(tx.tx_id)
        if balances.get(tx.src) != tx.src_old or balances.get(tx.dst) != tx.dst_old:
            violations.append(Violation(i, tx.tx_id, "LOST_UPDATE"))
        elif tx.src_new != tx.src_old - tx.amount or tx.dst_new != tx.dst_old + tx.amount:
            violations.append(Violation(i, tx.tx_id, "CALC_ERROR"))
        elif tx.src_new < 0 or tx.dst_new < 0:
            violations.append(Violation(i, tx.tx_id, "NEGATIVE_BAL"))
        _apply(balances, tx)
    return violations
