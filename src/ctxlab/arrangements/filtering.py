"""Arrangements that drop passages rather than reorder them."""

from __future__ import annotations

import random

from ctxlab.arrangements.formatting import build_prompt
from ctxlab.data.base import Example, Prompt
from ctxlab.registry import register_arrangement


@register_arrangement
class NoContext:
    """Closed-book floor: question only."""

    name = "no_context"

    def build(self, ex: Example, rng: random.Random) -> Prompt:
        del rng
        return build_prompt(ex, [], self.name)


@register_arrangement
class GoldOnly:
    """Oracle ceiling: supporting passages only."""

    name = "gold_only"

    def build(self, ex: Example, rng: random.Random) -> Prompt:
        del rng
        golds = [p for p in ex.passages if p.is_gold]
        return build_prompt(ex, golds, self.name)
