"""Arrangement protocol and shared prompt-construction helpers."""

from __future__ import annotations

import random
from typing import Protocol

from ctxlab.config import DEFAULT_SYSTEM_PROMPT
from ctxlab.data.base import Example, Prompt


class Arrangement(Protocol):
    name: str

    def build(self, ex: Example, rng: random.Random) -> Prompt: ...


DEFAULT_SYSTEM = DEFAULT_SYSTEM_PROMPT
