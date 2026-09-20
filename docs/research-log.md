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

## 2026-09-20 — transaction-log task: generator and window arrangements

No model numbers yet; this entry records design decisions that are hard to
recover from the diff. Replicating the "Error in a Log of Transactions" task
from arXiv 2512.13898 (Figure 7, Table 2).

- **Length is an arrangement, not a dataset knob.** The paper holds the needle
  fixed and grows the haystack, which is what a ctxlab arrangement already
  does. One long log per example; `tx_window_{25,95,250,500}` choose how much
  the model sees. The sweep then falls out of `ctxlab report` unchanged —
  rows are context lengths, columns will be the three arms.
- **Two accounts, on purpose.** Every line prints both accounts' old → new
  balances, so a line is a complete state transition and any contiguous window
  is self-describing: the renderer derives the `Initial state:` header from the
  first kept line. With three or more accounts a window would open in an
  unknown state and the log would be unverifiable. Two accounts also happens to
  match the paper's token calibration (~19 tokens/line, 500 lines ≈ 9.5k).
- **The duplicated commit is inserted, not substituted.** Substituting left a
  hole in the TX id sequence — a second, unintended tell that a model could
  find instead of the duplicate. Caught by looking at generator output, not by
  a test; `test_transaction_ids_never_skip` now pins it.
- **EM/F1 cannot score this task.** `normalize_answer` keeps only the first
  line and strips punctuation, so the reference pair collapses to a bag of
  words. Added `tx_joint` (the paper's accuracy) plus `tx_type` / `tx_id`,
  which say *how* a run fails — localisation is the half that should collapse
  with context length.
- Follow-up: in-context and thinking arms, then qTTT. Before spending GPU
  hours, gate on `tx_window_25` in-context accuracy — if the shortest context
  is already near the floor the sweep cannot show a crossover and the model is
  too small.
