"""Query-only test-time training (Algorithm 1 of arXiv 2512.13898).

One prefill caches {K, V} for the whole context. A few gradient steps then
adapt *only* the query projections on short spans sampled from that context,
reusing the cache. Keys and values do not depend on W_Q, so the cache stays
valid across the updates -- which is the entire reason the method is cheap
enough to run at inference.

The point is not to learn the answer. It is to raise the target-distractor
logit margin for this particular context (Proposition 3.1), counteracting the
score dilution that makes long-context retrieval fail.

Torch is imported lazily, as in `models.hf_local`, so `registry.load_plugins()`
still works without the `local` extra installed.
"""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass, field
from typing import Any

from ctxlab.config import ModelConfig
from ctxlab.data.base import Completion, Prompt
from ctxlab.models.decoding import apply_chat, extract_answer, tail
from ctxlab.models.hf_local import HuggingFaceLocalModel
from ctxlab.registry import register_model

# Appendix D: (k, N_TTT) = (128, 32), lr swept over {3e-4 ... 3e-7} with the
# plateau between 1e-5 and 1e-6. Compute-matched to T_think via T_think ~= 2*N*k.
DEFAULT_SPAN = 128
DEFAULT_STEPS = 32
DEFAULT_LR = 1e-5
DEFAULT_WEIGHT_DECAY = 0.01
DEFAULT_GRAD_CLIP = 1.0

QUERY_SUFFIX = "q_proj.weight"


@dataclass
class Prefill:
    """A single forward pass over the context, kept for the whole example.

    `kv` covers every token but the last, so `generate` can reuse it by
    forwarding only that final token.
    """

    input_ids: Any
    kv: list[tuple[Any, Any]]
    losses: list[float] = field(default_factory=list)
    steps_run: int = 0
    # Span actually used, which is not the requested k when the context is too
    # short to sample one. Recorded because the compute match to the thinking
    # arm is T_think ~= 2 * N_TTT * k -- reporting the requested k while having
    # run a shorter one would overstate the budget qTTT actually spent.
    span_used: int = 0

    @property
    def length(self) -> int:
        return int(self.kv[0][0].shape[-2])


class QueryOnlyTTT:
    """Fit on the test-time prompt, then generate. Satisfies `ttt.base.Adapter`.

    The only module that reaches into `HuggingFaceLocalModel.model` /
    `.tokenizer`, as `ttt/README.md` requires.
    """

    name = "qttt"

    def __init__(
        self,
        lm: HuggingFaceLocalModel,
        *,
        span: int = DEFAULT_SPAN,
        steps: int = DEFAULT_STEPS,
        lr: float = DEFAULT_LR,
        weight_decay: float = DEFAULT_WEIGHT_DECAY,
        grad_clip: float = DEFAULT_GRAD_CLIP,
        optimizer: str = "adamw",
        seed: int = 0,
    ) -> None:
        import torch

        self._torch = torch
        self.lm = lm
        self.span = span
        self.steps = steps
        self.lr = lr
        self.weight_decay = weight_decay
        self.grad_clip = grad_clip
        self.optimizer = optimizer
        self.seed = seed
        self._backup: dict[str, Any] | None = None
        self._prefill: Prefill | None = None

    # -- parameters ------------------------------------------------------

    def query_parameters(self) -> list[tuple[str, Any]]:
        return [
            (name, param)
            for name, param in self.lm.model.named_parameters()
            if name.endswith(QUERY_SUFFIX)
        ]

    # -- Adapter protocol ------------------------------------------------

    def fit(self, prompt: Prompt, **kwargs: Any) -> None:
        del kwargs
        torch = self._torch
        model = self.lm.model
        text = apply_chat(
            self.lm.tokenizer, prompt.to_chat(), enable_thinking=self.lm.enable_thinking
        )
        input_ids = self.lm.tokenizer(text, return_tensors="pt").input_ids.to(model.device)
        with torch.no_grad():
            cache = model(input_ids=input_ids[:, :-1], use_cache=True).past_key_values
        # Detached, so no gradient can reach W_K / W_V even by accident.
        kv = [(layer.keys.detach(), layer.values.detach()) for layer in cache.layers]
        self._prefill = Prefill(input_ids=input_ids, kv=kv)

        span = min(self.span, self._prefill.length - 1)
        self._prefill.span_used = max(span, 0) if self.steps > 0 else 0
        if self.steps <= 0 or span < 1:
            # Context too short to sample a span from. Decoding unadapted is
            # the honest outcome; it is also what the shortest window in the
            # sweep will do on a small model, and silently skipping it would
            # misreport that row as having been adapted.
            return
        self._adapt(self._prefill, span)

    def generate(self, prompt: Prompt, **kwargs: Any) -> Completion:
        del prompt
        from transformers import DynamicCache

        torch = self._torch
        state = self._prefill
        if state is None:
            raise RuntimeError("call fit() before generate()")
        model, tokenizer = self.lm.model, self.lm.tokenizer
        max_tokens = kwargs.get("max_tokens", self.lm.max_tokens)
        temperature = kwargs.get("temperature", self.lm.temperature)
        do_sample = temperature is not None and temperature > 0
        sampling: dict[str, Any] = {}
        if do_sample:
            sampling["temperature"] = temperature
            if self.lm.top_p is not None:
                sampling["top_p"] = self.lm.top_p
            if self.lm.top_k is not None:
                sampling["top_k"] = self.lm.top_k

        # Rebuilt from the stored tensors rather than reused in place, because
        # generation appends to whatever cache it is handed.
        cache = DynamicCache([(k, v) for k, v in state.kv], config=model.config)
        with torch.no_grad():
            out = model.generate(
                input_ids=state.input_ids,
                past_key_values=cache,
                max_new_tokens=max_tokens,
                do_sample=do_sample,
                pad_token_id=tokenizer.pad_token_id,
                **sampling,
            )
        new_tokens = out[0, state.input_ids.shape[-1] :]
        raw_text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        n_new = int(new_tokens.shape[-1])
        mean_loss = sum(state.losses) / len(state.losses) if state.losses else None
        return Completion(
            text=extract_answer(raw_text),
            raw={
                "model_id": self.lm.model_id,
                "n_new_tokens": n_new,
                "raw_text": tail(raw_text),
                "truncated": n_new >= max_tokens,
                "ttt_steps": state.steps_run,
                "ttt_span_requested": self.span,
                "ttt_span_used": state.span_used,
                "ttt_lr": self.lr,
                # The whole trajectory, not its endpoints. Each step samples a
                # different span, so first-vs-last compares two unrelated
                # spans and would invite reading noise as progress.
                "ttt_losses": list(state.losses),
                "ttt_mean_loss": mean_loss,
            },
            usage={"input_tokens": int(state.input_ids.shape[-1]), "output_tokens": n_new},
        )

    def restore(self) -> None:
        """Put the base query projections back so examples stay independent."""
        if self._backup is None:
            return
        with self._torch.no_grad():
            for name, param in self.query_parameters():
                param.copy_(self._backup[name])
        self._backup = None
        self._prefill = None

    # -- internals -------------------------------------------------------

    def _adapt(self, state: Prefill, span: int) -> None:
        torch = self._torch
        state.span_used = span
        model = self.lm.model
        named = self.query_parameters()
        if not named:
            raise ValueError(
                f"no parameters ending in {QUERY_SUFFIX!r}; query-only TTT needs a "
                "model whose attention exposes separate query projections"
            )
        self._backup = {name: param.detach().clone() for name, param in named}
        was_trainable = {n: p.requires_grad for n, p in model.named_parameters()}
        for param in model.parameters():
            param.requires_grad_(False)
        params = []
        for _, param in named:
            param.requires_grad_(True)
            params.append(param)

        optimizer = self._make_optimizer(params)
        rng = random.Random(self._span_seed(state.input_ids))
        try:
            for _ in range(self.steps):
                start = rng.randint(1, state.length - span)
                loss = self._span_loss(state, start, span)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, self.grad_clip)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                state.losses.append(float(loss.detach()))
                state.steps_run += 1
        finally:
            for name, param in model.named_parameters():
                param.requires_grad_(was_trainable[name])

    def _span_loss(self, state: Prefill, start: int, span: int) -> Any:
        from transformers import DynamicCache

        torch = self._torch
        model = self.lm.model
        device = state.input_ids.device
        # Sliced views of the frozen prefill. A fresh cache each step, because
        # cropping the shared one in place would destroy it for later steps.
        cache = DynamicCache(
            [(k[:, :, :start], v[:, :, :start]) for k, v in state.kv], config=model.config
        )
        positions = torch.arange(start, start + span, device=device)
        out = model(
            input_ids=state.input_ids[:, start : start + span],
            past_key_values=cache,
            position_ids=positions.unsqueeze(0),
            cache_position=positions,
            attention_mask=torch.ones(1, start + span, dtype=torch.long, device=device),
            use_cache=True,
        )
        targets = state.input_ids[:, start + 1 : start + span + 1]
        return torch.nn.functional.cross_entropy(
            out.logits.reshape(-1, out.logits.shape[-1]).float(), targets.reshape(-1)
        )

    def _make_optimizer(self, params: list[Any]) -> Any:
        torch = self._torch
        if self.optimizer == "sgd":
            # Escape hatch for GPUs that cannot hold AdamW's two extra states
            # for every query projection (see the L4 tier in the Colab notes).
            return torch.optim.SGD(params, lr=self.lr, weight_decay=self.weight_decay)
        if self.optimizer != "adamw":
            raise ValueError(f"Unknown optimizer {self.optimizer!r}; use 'adamw' or 'sgd'")
        return torch.optim.AdamW(params, lr=self.lr, weight_decay=self.weight_decay)

    def _span_seed(self, input_ids: Any) -> int:
        """Spans depend only on the prompt, so a rerun reproduces a run.

        The completion cache assumes `generate` is a function of the prompt;
        sampling spans from global randomness would break that quietly.
        """
        digest = hashlib.sha256(
            f"{self.seed}|".encode() + input_ids.detach().cpu().numpy().tobytes()
        ).hexdigest()
        return int(digest[:8], 16)


@register_model("ttt")
class TestTimeTrainingModel:
    """`LanguageModel` wrapper: fits a fresh adapter for every example.

    Hyperparameters ride in `ModelConfig.extra`, which the runner folds into
    the completion cache key -- so two learning rates do not share cached
    completions.
    """

    def __init__(self, cfg: ModelConfig) -> None:
        extra = dict(cfg.extra)
        span = int(extra.pop("k", DEFAULT_SPAN))
        steps = int(extra.pop("n_ttt", DEFAULT_STEPS))
        lr = float(extra.pop("lr", DEFAULT_LR))
        weight_decay = float(extra.pop("weight_decay", DEFAULT_WEIGHT_DECAY))
        grad_clip = float(extra.pop("grad_clip", DEFAULT_GRAD_CLIP))
        optimizer = str(extra.pop("optimizer", "adamw"))
        seed = int(extra.pop("ttt_seed", 0))
        self.name = cfg.name
        self.lm = HuggingFaceLocalModel(cfg.model_copy(update={"extra": extra}))
        self.adapter = QueryOnlyTTT(
            self.lm,
            span=span,
            steps=steps,
            lr=lr,
            weight_decay=weight_decay,
            grad_clip=grad_clip,
            optimizer=optimizer,
            seed=seed,
        )

    def generate(self, prompt: Prompt, **kwargs: Any) -> Completion:
        self.adapter.fit(prompt, **kwargs)
        try:
            return self.adapter.generate(prompt, **kwargs)
        finally:
            self.adapter.restore()
