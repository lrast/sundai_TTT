# Contributing

The interesting surface is **arrangements**. Everything else is shared infrastructure; change it only when the protocol needs to move.

## Adding an arrangement

1. Pick a module:
   - drop passages → [`src/ctxlab/arrangements/filtering.py`](src/ctxlab/arrangements/filtering.py)
   - reorder passages → [`src/ctxlab/arrangements/positional.py`](src/ctxlab/arrangements/positional.py)
   - change markup only → [`src/ctxlab/arrangements/formatting.py`](src/ctxlab/arrangements/formatting.py)
2. Register a class. `build` must be a pure function of `(example, rng)`:

```python
from ctxlab.arrangements.formatting import build_prompt
from ctxlab.data.base import Example, Prompt
from ctxlab.registry import register_arrangement
import random

@register_arrangement
class GoldFirstTwoDistractors:
    name = "gold_plus_2"

    def build(self, ex: Example, rng: random.Random) -> Prompt:
        golds = [p for p in ex.passages if p.is_gold]
        distractors = [p for p in ex.passages if not p.is_gold]
        rng.shuffle(distractors)
        return build_prompt(ex, golds + distractors[:2], self.name)
```

3. Use `build_prompt(...)` so `meta` stays truthful (`gold_positions`, `kept_gold_titles`, `dropped_titles`). Tests assert this.
4. Add the name to a YAML config under `arrangements:`.
5. Add a unit test in `tests/test_arrangements.py` for the invariant you care about (what was dropped, where gold landed).
6. Run `make smoke` and open a PR.

Do **not** put gold answers into the prompt text or into fields the model sees. `Prompt.meta` is logged but never sent to the model or the cache key.

## Branch / PR conventions

- Branch: `arr/<name>` for a new arrangement, `exp/<name>` for a sweep config, `fix/<name>` otherwise.
- One arrangement (or one tightly related family) per PR.
- PR description: hypothesis, config path, n examples, model. Paste the `ctxlab report` table.
- Do not commit `runs/`, `.cache/`, or `.env`.
- If you learned something, add a dated note to `docs/research-log.md` in the same PR.

## Running experiments

```bash
uv run ctxlab run -c configs/experiments/your_sweep.yaml
uv run ctxlab report runs/your_sweep
```

Hosted models go through LiteLLM (`kind: api`, `model: gpt-4o-mini` / `claude-3-5-haiku-latest` / `openrouter/...`). Local weights: `kind: hf_local` (install with `uv sync --extra local`).

## Code style

`ruff check` + `ruff format`. Pre-commit is optional but CI runs both plus pytest on the mock model.
