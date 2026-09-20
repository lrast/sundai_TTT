"""Run the arrangement x model x example cross product."""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from typing import Any

from ctxlab.cache import CompletionCache
from ctxlab.config import ExperimentConfig, ModelConfig, dump_config
from ctxlab.registry import (
    get_arrangement,
    get_dataset_loader,
    get_metric,
    get_model_cls,
    load_plugins,
)
from ctxlab.tracking import Tracker


def _stable_seed(root: int, *parts: str) -> int:
    h = hashlib.sha256(f"{root}|{'|'.join(parts)}".encode()).hexdigest()
    return int(h[:8], 16)


def _gen_params(cfg: ModelConfig) -> dict[str, Any]:
    """Generation parameters, which also form part of the completion cache key.

    `extra` is included because that is where per-arm decoding knobs and
    test-time-training hyperparameters live; without it two models whose only
    difference is `top_p` (or a TTT learning rate) would share cached
    completions. It is omitted when empty so configs that never used `extra`
    keep the cache keys they already have.
    """
    params: dict[str, Any] = {"temperature": cfg.temperature, "max_tokens": cfg.max_tokens}
    if cfg.extra:
        params["extra"] = dict(cfg.extra)
    return params


def run_experiment(cfg: ExperimentConfig) -> Path:
    load_plugins()
    run_dir = Path(cfg.runs_dir) / cfg.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    dump_config(cfg, run_dir / "config.yaml")

    examples = get_dataset_loader(cfg.dataset.name)(cfg.dataset)
    arrangements = [get_arrangement(name) for name in cfg.arrangements]
    model_cfgs = {m.name: m for m in cfg.models}
    models = [get_model_cls(m.kind)(m) for m in cfg.models]
    metrics = [get_metric(name) for name in cfg.metrics]
    cache = CompletionCache(Path(cfg.cache_dir))
    tracker = Tracker(run_dir, cfg.wandb, cfg.run_id)
    done = tracker.completed_keys()

    n_skip = 0
    n_run = 0
    try:
        for example in examples:
            for arrangement in arrangements:
                for model in models:
                    key = (example.uid, arrangement.name, model.name)
                    if key in done:
                        n_skip += 1
                        continue
                    rng = random.Random(_stable_seed(cfg.seed, example.uid, arrangement.name))
                    prompt = arrangement.build(example, rng)
                    if cfg.system_prompt and prompt.system != cfg.system_prompt:
                        prompt = type(prompt)(
                            system=cfg.system_prompt,
                            messages=prompt.messages,
                            meta=prompt.meta,
                        )
                    mcfg = model_cfgs[model.name]
                    completion, cached = cache.get_or_generate(model, prompt, _gen_params(mcfg))
                    scores = {
                        metric.name: metric.score(completion.text, example.answers)
                        for metric in metrics
                    }
                    tracker.log(
                        {
                            "uid": example.uid,
                            "question": example.question,
                            "answers": example.answers,
                            "arrangement": arrangement.name,
                            "model": model.name,
                            "prompt": prompt.to_record(),
                            "completion": completion.text,
                            "usage": completion.usage,
                            "metrics": scores,
                            "cached": cached,
                        }
                    )
                    n_run += 1
                    done.add(key)
    finally:
        tracker.close()

    summary = {
        "run_id": cfg.run_id,
        "n_examples": len(examples),
        "n_arrangements": len(arrangements),
        "n_models": len(models),
        "n_written": n_run,
        "n_skipped": n_skip,
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return run_dir
