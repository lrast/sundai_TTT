# Research log

Dated notes, not code. One entry per experiment or dead end. Link the config and the `runs/<id>` directory (local to whoever ran it).

## Template

```
## YYYY-MM-DD — short title
- Hypothesis:
- Config:
- n / model:
- Result (EM by arrangement):
- What we think it means:
- Follow-up:
```

## 2026-09-20 — scaffold + first position sweep

Repo bootstrapped. First real number from `configs/experiments/position_sweep.yaml`
on HotpotQA distractor validation, n=20, model `gpt-4o-mini` (temperature 0).

Mean EM:

- `no_context` 0.150 (closed-book floor)
- `gold_only` 0.700 (oracle ceiling)
- `gold_first` 0.550
- `gold_last` / `gold_middle` / `shuffled` 0.500

Mean F1 follows the same order, with `gold_only` at 0.852. n=20 is too small
to claim a lost-in-the-middle effect; the floor/ceiling gap is the takeaway.
Follow-up: larger n, and a model with a longer context window.

## 2026-09-20 — PyResBugs bug localization: arrangement doesn't move the needle (yet)

- Hypothesis: gold-function position in a 10-function haystack changes localization accuracy.
- Config: `configs/experiments/bugloc_{impl,ctx,high}.yaml` — PyResBugs, n=50,
  6 arrangements, gpt-4o-mini + gpt-5.6-luna.
- Result (mean EM): all position arrangements statistically tied (0.76–0.86, n=50,
  SE ≈ ±0.06); `no_context` 0.72–0.84; `gold_only` a flat 0.86 ceiling across both
  models and all three description levels.
- What we think it means: for these OpenAI models, this open-source bug data produces
  no variation in retrieval by context arrangement. Candidate explanations:
  (1) a 10-function context is tiny relative to the model window, so nothing dilutes;
  (2) public OSS bugs are plausibly in pretraining data — models localize ~75% of bugs
  with **no code shown**, so descriptions plus parametric memory carry the task;
  (3) the task as posed is below what these models can do. Practical corollary: repo-QA
  benchmarks built from public code need a no-context floor reported, or "reading the
  repo" gets over-credited.
- Follow-up: `bugloc_curve{10,25,50}` — mask the function name out of the description
  and grow the haystack, to separate "robust retrieval" from "never had to retrieve".
