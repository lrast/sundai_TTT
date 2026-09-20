"""Deterministic mock model so tests and CI never touch the network."""

from __future__ import annotations

import hashlib
import random
from typing import Any

from ctxlab.config import ModelConfig
from ctxlab.data.base import Completion, Prompt
from ctxlab.data.txlog import BUG_TYPES, canonical_answer, parse_tx_line, validate_log
from ctxlab.registry import register_model


@register_model("mock")
class MockModel:
    """Behaviors:

    - `oracle` (default): if the prompt still contains gold passages, return the
      first kept gold title; otherwise `unknown`. Toy answers match those titles
      so a smoke run already shows closed-book < open-book.
    - `unknown`: always `unknown`.
    - `echo`: last 80 characters of the user message.
    - `txlog_solver`: re-derives the answer from the rendered transaction log.
      It reads the prompt *text*, not `meta`, so a smoke run fails if a window
      is not self-describing -- that is, if the renderer or the windowing ever
      stops handing the model enough information to solve the task.
    - `txlog_random`: uniform guess over the transactions actually shown, and
      over the four bug types. This is the chance baseline, and a sweep wants
      it because chance is *not* constant across the x-axis: picking a
      transaction at random is right 1-in-25 of the time in the shortest
      window and 1-in-500 in the longest. Without it, "accuracy falls as
      context grows" is partly just the guess getting harder, which is not
      what score dilution claims. Costs nothing to run.
    """

    def __init__(self, cfg: ModelConfig) -> None:
        self.name = cfg.name
        self.behavior = str(cfg.extra.get("behavior", "oracle"))

    def generate(self, prompt: Prompt, **kwargs: Any) -> Completion:
        del kwargs
        if self.behavior == "unknown":
            text = "unknown"
        elif self.behavior == "txlog_solver":
            text = self._solve_txlog(prompt)
        elif self.behavior == "txlog_random":
            text = self._guess_txlog(prompt)
        elif self.behavior == "echo":
            last = prompt.messages[-1].content if prompt.messages else ""
            text = last[-80:]
        else:
            kept = prompt.meta.get("kept_gold_titles") or []
            text = kept[0] if kept else "unknown"
        return Completion(text=text, raw={"behavior": self.behavior}, usage=None)

    @staticmethod
    def _txlog_ids(prompt: Prompt) -> tuple[str, list[str]]:
        body = prompt.messages[-1].content if prompt.messages else ""
        ids = [tx.tx_id for tx in (parse_tx_line(ln) for ln in body.splitlines()) if tx]
        return body, ids

    @classmethod
    def _guess_txlog(cls, prompt: Prompt) -> str:
        body, tx_ids = cls._txlog_ids(prompt)
        if not tx_ids:
            return "unknown"
        # Seeded by the prompt, because the completion cache assumes generate
        # is a function of its input.
        rng = random.Random(hashlib.sha256(body.encode()).hexdigest())
        return canonical_answer(rng.choice(BUG_TYPES), rng.choice(tx_ids))

    @staticmethod
    def _solve_txlog(prompt: Prompt) -> str:
        body = prompt.messages[-1].content if prompt.messages else ""
        transactions = [
            tx for tx in (parse_tx_line(line) for line in body.splitlines()) if tx is not None
        ]
        violations = validate_log(transactions)
        if len(violations) != 1:
            return f"unknown ({len(violations)} violations found)"
        found = violations[0]
        return canonical_answer(found.bug_type, found.tx_id)
