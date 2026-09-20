"""LLM-as-judge metric. Stubbed for a later experiment.

Do not wire this into default experiment configs until a judge prompt and
aggregation policy are agreed on in the research log.
"""

from __future__ import annotations


class JudgeMetric:
    name = "judge"

    def score(self, prediction: str, golds: list[str]) -> float:
        del prediction, golds
        raise NotImplementedError(
            "LLM-as-judge is not implemented in this scaffold. "
            "Use em/f1, or implement metrics/judge.py and register it."
        )
