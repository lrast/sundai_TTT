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
