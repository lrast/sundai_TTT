"""Language-model protocol. Implementations live in mock/api/hf_local."""

from __future__ import annotations

from typing import Any, Protocol

from ctxlab.data.base import Completion, Prompt


class LanguageModel(Protocol):
    """Anything the runner can call.

    HuggingFaceLocalModel also exposes `.model` and `.tokenizer` for the
    test-time-training phase. No other code should reach into weights.
    """

    name: str

    def generate(self, prompt: Prompt, **kwargs: Any) -> Completion: ...
