"""Test-time training seam.

Nothing here runs in phase 1. The contract is: an Adapter is fitted on the
test-time context (the same `Prompt` an arrangement just built), then used
to generate. It is the only module allowed to touch
`HuggingFaceLocalModel.model` and `.tokenizer`.

Hosted API models cannot participate; TTT needs local weights and gradients.
"""

from __future__ import annotations

from typing import Any, Protocol

from ctxlab.data.base import Completion, Prompt


class Adapter(Protocol):
    """Fit on a test-time prompt, then generate.

    A typical implementation:

    1. Take a `HuggingFaceLocalModel` (for `.model` / `.tokenizer`).
    2. `fit(prompt)` runs a few gradient steps on a self-supervised loss
       over the context tokens (or a LoRA adapter on those weights).
    3. `generate(prompt)` decodes the answer with the adapted weights.
    4. Optionally restore the base weights so examples stay independent.
    """

    name: str

    def fit(self, prompt: Prompt, **kwargs: Any) -> None: ...

    def generate(self, prompt: Prompt, **kwargs: Any) -> Completion: ...


class NotYetImplementedAdapter:
    name = "ttt_stub"

    def fit(self, prompt: Prompt, **kwargs: Any) -> None:
        del prompt, kwargs
        raise NotImplementedError("Test-time training is phase 2. See src/ctxlab/ttt/README.md.")

    def generate(self, prompt: Prompt, **kwargs: Any) -> Completion:
        del prompt, kwargs
        raise NotImplementedError("Test-time training is phase 2. See src/ctxlab/ttt/README.md.")
