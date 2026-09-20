# Test-time training (phase 2)

This package is a **landing spot**, not a working trainer.

## Why it sits here

Phase 1 measures how *prompt geometry* (order, filtering, formatting) changes
QA accuracy for a frozen model. Phase 2 asks whether *updating weights on the
test context* beats, complements, or interacts with those geometries.

The rest of `ctxlab` must not import `transformers` except through
`HuggingFaceLocalModel`. TTT is the exception: adapters may use
`HuggingFaceLocalModel.model` and `.tokenizer` directly.

## Intended shape

```python
from ctxlab.models.hf_local import HuggingFaceLocalModel
from ctxlab.ttt.base import Adapter


class LanguageModelingTTT:
    name = "lm_ttt"

    def __init__(self, lm: HuggingFaceLocalModel, steps: int = 4, lr: float = 1e-5):
        self.lm = lm  # .model, .tokenizer
        self.steps = steps
        self.lr = lr

    def fit(self, prompt):
        # self-supervised loss on context tokens in prompt.messages
        ...

    def generate(self, prompt):
        return self.lm.generate(prompt)
```

Wire it as another `LanguageModel` kind (`ttt`) so the runner's arrangement
x model cross product just works: each arrangement's prompt is the TTT
context.

## Non-goals for this stub

- No optimizer, no LoRA, no inner-loop hyperparameters.
- No API-model TTT (impossible without weights).
- Do not log gold answers into the TTT loss.
