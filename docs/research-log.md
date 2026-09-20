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

## 2026-09-20 — query-only TTT, and a token-density discrepancy with the paper

Adapter implemented (`src/ctxlab/ttt/qttt.py`, model kind `ttt`). Still no model
numbers; this records two things worth knowing before the sweep runs.

**The paper's token axis cannot come from the format in its own Figure 7.**
Table 2 puts 25 transactions at 512 tokens and 500 at 9,560 — about 19 tokens
per line. But the line printed in Figure 7,

```
[TX001]: Transfer $107: A=4000 → 3893, B=4200 → 4307
```

costs **37 tokens** under Qwen3's own tokenizer (Qwen splits digits
individually, so four 4-digit balances alone are 16 tokens). Our generated
lines cost 37 too, so the format is faithful to the figure; the table's token
counts are not reproducible from it. Measured window sizes:

| transactions | ours (measured) | paper Table 2 |
|---|---|---|
| 25 | 1,109 | 512 |
| 95 | 3,763 | 2,536 |
| 250 | 9,644 | 5,120 |
| 500 | 19,110 | 9,560 |

We match Figure 1(b)'s x-axis (transaction count) with the paper's own line
format rather than reverse-engineering a terser encoding to hit Table 2's
numbers. Consequence: at equal transaction counts our contexts are ~2x longer
in tokens, so our curve should sit *below* the paper's — more dilution, not
less. The claim being replicated is the shape and the crossover, not the
absolute values. `usage.input_tokens` is logged per record, so the real axis is
always recoverable. Pinned by `test_rendered_lines_cost_what_the_paper_figure_costs`.

Practical fallout: 500 transactions is ~19k tokens, roughly double what the
paper's budget assumed, so peak memory for Qwen3-4B lands near 21GB (weights 8,
AdamW states 4.5, KV cache 2.8, span-backward activations 5.6). Comfortable on
an A100 40GB; an L4 22GB needs `optimizer: sgd`.

**Two mechanism facts worth trusting the implementation on**, both pinned by
tests on a tiny model:

- Forwarding a span against a *sliced* prefill cache is bit-identical to a full
  forward (max abs diff 1.5e-7), and decoding against the reused cache
  reproduces an ordinary `generate` exactly. Had either been subtly wrong,
  every qTTT number would have been wrong in a way no accuracy metric could
  distinguish from the method simply not working.
- The stored K/V tensors are unchanged by the update steps, which is the
  property the whole method rests on (K and V do not depend on W_Q).

One bug found by testing rather than by reading: when the context is too short
for k=128 the adapter shrinks the span, but it was still *reporting* k=128. The
arms are FLOP-matched through T_think ≈ 2·N_TTT·k, so that would have
overstated qTTT's budget. Records now carry `ttt_span_requested` and
`ttt_span_used` separately.

**End-to-end on real weights** (Qwen3-0.6B, CPU, n=1, `tx_window_25`): both
arms ran clean. The model emits exactly the requested JSON, wrapped in a
```json fence that `strip_fences` handles. qTTT recorded `ttt_steps: 4`,
`ttt_span_used: 64`, and a descending loss. Both arms produced byte-identical
completions -- verified as genuine (distinct cache keys, `cached: false`), not
a cache collision: four steps at lr 1e-5 on a 0.6B model does not move a greedy
argmax. Scores were `tx_type` 1.0 / `tx_id` 0.0, which is the split those
sub-metrics exist for: at 0.6B the model names the bug correctly and fails
*localisation*, the retrieval half the paper says collapses with length.

Two corrections that run forced:

- `completion.raw` was never persisted by the runner, so every TTT diagnostic
  was being thrown away. Now logged, with `raw_text` bounded to its tail so a
  thinking run does not bloat the JSONL.
- `ttt_first_loss` / `ttt_last_loss` were misleading: each step samples a
  *different* span, so the two are unrelated text and the drop between them is
  mostly noise. Records now carry the full `ttt_losses` trajectory, and the
  test that adaptation works measures one fixed probe span before and after.

Follow-up: run `txlog_gate.yaml` before the sweep. If in-context accuracy at
`tx_window_25` is near zero the model is too small and the crossover cannot
appear.

## 2026-09-20 — local runs, and two cache/memory bugs they exposed

Running the harness against real Qwen3 weights on a laptop, before spending
anything on a GPU. Two bugs surfaced that would both have corrupted the Colab
sweep silently.

**The completion cache ignored which model produced the completion.** A
Qwen3-1.7B gate run returned, to three decimals, the numbers from a Qwen3-0.6B
run: `tx_id` 0.200, `tx_type` 0.320, `tx_joint` 0.040. Every completion was a
cache hit. Only the arm's *name* was in the key, and both configs called the
arm `in_context`, so `ModelConfig.model` — the weights — never entered the
hash. Nothing in the report table showed it; the recorded `raw.model_id` is
what gave it away, which is the second time that diagnostic has paid for
itself. `_gen_params` now covers model id and `extra` with no special cases:
if it changes the output, it is in the key. This invalidates `.cache/`, which
is local and gitignored — a stale cache costs a re-run, a wrong one costs a
wrong result.

Corrected gate numbers, n=25, `tx_window_25`:

| model | tx_id | tx_type | tx_joint |
|---|---|---|---|
| Qwen3-0.6B | 0.200 | 0.320 | 0.040 |
| Qwen3-1.7B | 0.240 | 0.520 | 0.160 |

Qwen3-1.7B clears the gate: 16% joint at the shortest window is ~16x chance,
with room to fall. 0.6B is floored on the joint metric.

**Three arms meant three copies of the same checkpoint.** The runner builds
every model in a config up front, so a three-arm sweep over one checkpoint held
three sets of weights. This OOM'd MPS at 20GB on a 1.7B model in fp32, and
would have OOM'd an A100 on `txlog_replication.yaml`: 3 x 8GB for Qwen3-4B in
bf16, before a single activation. Arms sharing a checkpoint now share the
loaded weights. Safe because arms differ only in decoding parameters; the TTT
arm mutates query projections but restores them in a `finally`, which
`test_restore_puts_the_base_weights_back` is there to keep true. An earlier
note in this log put peak usage at ~21GB — that figure is only correct *with*
this fix.

**Apple Silicon needs a plain `.to(device)`.** `device_map="mps"` segfaults the
interpreter (exit 139) rather than raising. Added a `device` option that loads
without accelerate's dispatch and then moves.

**Chance is not constant across the x-axis, and the paper does not say so.**
Guessing a transaction at random is right 1-in-25 of the time in the shortest
window and 1-in-500 in the longest — a 20x drop in the baseline across the
sweep. So part of any measured decline is the guess getting harder, which is
not what score dilution claims. Added a `txlog_random` mock arm as an empirical
baseline; it costs nothing to run and tracks theory closely (4.5% vs 4.0% at
window 25, 0.5% vs 0.2% at 500). Every sweep should carry it.

**Thinking has a budget floor, not a gentle slope.** At a compute-matched local
budget of T_think=1024, Qwen3-1.7B never closes `</think>`: all six timing
records hit exactly 1024 output tokens mid-calculation, extraction correctly
returns nothing, and the arm scores 0.00 at every window. That is budget
starvation, not dilution.

This is worth more than a note about laptops. The FLOP-matching argument in
§3.3 treats thinking and qTTT as two ways to spend one budget, which assumes
thinking degrades smoothly as the budget shrinks. It does not — below some
floor it emits no answer at all, while qTTT at the same budget still produces
one. A matched comparison at small budgets therefore flatters qTTT for a reason
that has nothing to do with attention. Worth measuring where that floor sits
before reading too much into any FLOP-matched number, ours or the paper's.
