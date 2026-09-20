"""Deterministic mock model so tests and CI never touch the network."""

from __future__ import annotations

from typing import Any

from ctxlab.config import ModelConfig
from ctxlab.data.base import Completion, Prompt
from ctxlab.registry import register_model


@register_model("mock")
class MockModel:
    """Behaviors:

    - `oracle` (default): if the prompt still contains gold passages, return the
      first kept gold title; otherwise `unknown`. Toy answers match those titles
      so a smoke run already shows closed-book < open-book.
    - `unknown`: always `unknown`.
    - `echo`: last 80 characters of the user message.
    """

    def __init__(self, cfg: ModelConfig) -> None:
        self.name = cfg.name
        self.behavior = str(cfg.extra.get("behavior", "oracle"))

    def generate(self, prompt: Prompt, **kwargs: Any) -> Completion:
        del kwargs
        if self.behavior == "unknown":
            text = "unknown"
        elif self.behavior == "echo":
            last = prompt.messages[-1].content if prompt.messages else ""
            text = last[-80:]
        else:
            kept = prompt.meta.get("kept_gold_titles") or []
            text = kept[0] if kept else "unknown"
        return Completion(text=text, raw={"behavior": self.behavior}, usage=None)
