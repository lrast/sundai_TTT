"""Arrangements that keep every passage and only change order."""

from __future__ import annotations

import random

from ctxlab.arrangements.formatting import build_prompt
from ctxlab.data.base import Example, Passage, Prompt
from ctxlab.registry import register_arrangement


def _split_gold(ex: Example) -> tuple[list[Passage], list[Passage]]:
    golds = [p for p in ex.passages if p.is_gold]
    distractors = [p for p in ex.passages if not p.is_gold]
    return golds, distractors


@register_arrangement
class GoldFirst:
    name = "gold_first"

    def build(self, ex: Example, rng: random.Random) -> Prompt:
        del rng
        golds, distractors = _split_gold(ex)
        return build_prompt(ex, golds + distractors, self.name)


@register_arrangement
class GoldLast:
    name = "gold_last"

    def build(self, ex: Example, rng: random.Random) -> Prompt:
        del rng
        golds, distractors = _split_gold(ex)
        return build_prompt(ex, distractors + golds, self.name)


@register_arrangement
class GoldMiddle:
    """Lost-in-the-middle probe: gold block in the center of distractors."""

    name = "gold_middle"

    def build(self, ex: Example, rng: random.Random) -> Prompt:
        del rng
        golds, distractors = _split_gold(ex)
        mid = len(distractors) // 2
        ordered = distractors[:mid] + golds + distractors[mid:]
        return build_prompt(ex, ordered, self.name)


@register_arrangement
class Shuffled:
    """Seeded shuffle of the dataset-native passage list."""

    name = "shuffled"

    def build(self, ex: Example, rng: random.Random) -> Prompt:
        passages = list(ex.passages)
        rng.shuffle(passages)
        return build_prompt(ex, passages, self.name)
