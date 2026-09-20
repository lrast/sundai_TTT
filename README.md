# ctxlab

Collaborative lab for **context arrangement** experiments (and, later, test-time training).

The unit of research is an **arrangement**: a pure function from a retrieval example to a prompt. Datasets, models, metrics, and logging stay fixed so a new arrangement is one class and a YAML line.

```
example  ->  arrangement.build()  ->  prompt  ->  model.generate()  ->  EM / F1
                                              ^
                                       disk cache
```

## Quick start

```bash
uv sync --group dev
cp .env.example .env   # only needed for hosted models
make test
make smoke
```

`make smoke` runs the full pipeline on the in-repo toy dataset and a deterministic mock model. No network, no keys.

A 20-example HotpotQA distractor position sweep against a cheap API model
(`hotpotqa/hotpot_qa`, not the retired `hotpot_qa` script dataset):

```bash
# requires OPENAI_API_KEY (or another LiteLLM-compatible key)
make sweep-small
```

Re-runs are cheap: completions are hashed to `.cache/completions/`, and already-written `(uid, arrangement, model)` rows in `runs/<id>/records.jsonl` are skipped.

## Layout

| Path | Role |
|---|---|
| `src/ctxlab/arrangements/` | What you probably want to edit. Filtering + positional arrangements live here. |
| `src/ctxlab/data/` | `Example` / `Passage` types; toy + HotpotQA distractor loaders |
| `src/ctxlab/models/` | `mock`, `api` (LiteLLM), `hf_local` (weights + tokenizer for TTT) |
| `src/ctxlab/metrics/` | SQuAD-style EM / token-F1; judge stub |
| `src/ctxlab/runner.py` | Cross product, cache, resume |
| `src/ctxlab/ttt/` | Phase-2 landing spot. Do not put arrangement code here. |
| `configs/experiments/` | YAML sweeps |
| `runs/` | Per-run JSONL + copied config (gitignored) |
| `docs/research-log.md` | Dated notes, not code |

## Adding an arrangement

See [CONTRIBUTING.md](CONTRIBUTING.md). Short version: subclass, set `name`, implement `build`, export via the `@register_arrangement` decorator, add the name to a YAML config.

## Results

JSONL under `runs/<run_id>/records.jsonl` is the source of truth. W&B is optional and off by default (`wandb.enabled: true` in YAML, plus `uv sync --extra wandb`).

```bash
uv run ctxlab report runs/smoke
```

`notebooks/analysis.ipynb` reads the same JSONL.

## Test-time training

Not in this scaffold. `HuggingFaceLocalModel` exposes `.model` and `.tokenizer` so a later adapter can fit on the test-time prompt. Read `src/ctxlab/ttt/README.md`.
