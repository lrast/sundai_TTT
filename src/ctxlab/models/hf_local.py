"""Local Hugging Face causal LM.

Exposes `.model` and `.tokenizer` as public attributes. That is the only
seam the test-time-training phase should use to reach weights; the rest of
ctxlab talks to `generate()` only.
"""

from __future__ import annotations

from typing import Any

from ctxlab.config import ModelConfig
from ctxlab.data.base import Completion, Prompt
from ctxlab.models.decoding import apply_chat, extract_answer, tail
from ctxlab.registry import register_model

# Weights are shared between arms that ask for the same checkpoint. The
# runner builds every model in the config up front, so a three-arm sweep over
# one checkpoint would otherwise hold three copies: 24GB for Qwen3-4B in bf16,
# before a single activation. That OOMs an A100 on the paper's own setup.
#
# Safe because arms differ only in decoding parameters. The exception is the
# TTT arm, which mutates query projections -- it restores them in a `finally`,
# and `tests/test_qttt.py::test_restore_puts_the_base_weights_back` is what
# keeps that true.
_WEIGHTS: dict[tuple[str, str, str, str], tuple[Any, Any]] = {}


def clear_weight_cache() -> None:
    _WEIGHTS.clear()


def _dtype_kwarg() -> str:
    """`torch_dtype` was renamed to `dtype` in transformers 5."""
    from transformers import __version__ as version

    return "dtype" if int(version.split(".")[0]) >= 5 else "torch_dtype"


@register_model("hf_local")
class HuggingFaceLocalModel:
    def __init__(self, cfg: ModelConfig) -> None:
        if not cfg.model:
            raise ValueError(f"hf_local model {cfg.name!r} requires `model` (Hub id or path)")
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise ImportError(
                "hf_local requires the `local` extra: `uv sync --extra local`"
            ) from exc

        self.name = cfg.name
        self.model_id = cfg.model
        self.temperature = cfg.temperature
        self.max_tokens = cfg.max_tokens
        self.extra = dict(cfg.extra)
        self._torch = torch

        # Decoding knobs the paper sets per arm (Appendix D). They ride in
        # `extra` because ModelConfig has no field for them, and the runner
        # folds `extra` into the completion cache key so two arms that differ
        # only here do not collide in `.cache/`. Set before any early return:
        # arms sharing weights still need their own decoding parameters.
        self.enable_thinking = self.extra.pop("enable_thinking", None)
        self.top_p = self.extra.pop("top_p", None)
        self.top_k = self.extra.pop("top_k", None)

        # `device` loads plainly and then moves, bypassing accelerate's
        # dispatch. That is the only path that works on Apple Silicon:
        # `device_map="mps"` segfaults the interpreter.
        device = self.extra.pop("device", None)
        device_map = self.extra.pop("device_map", None if device else "auto")
        torch_dtype = self.extra.pop("torch_dtype", None)
        dtype = getattr(torch, torch_dtype) if isinstance(torch_dtype, str) else torch_dtype

        cache_key = (self.model_id, str(dtype), str(device), str(device_map))
        if cache_key in _WEIGHTS:
            self.model, self.tokenizer = _WEIGHTS[cache_key]
            return

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        load_kwargs: dict[str, Any] = {"device_map": device_map}
        if dtype is not None:
            load_kwargs[_dtype_kwarg()] = dtype
        self.model = AutoModelForCausalLM.from_pretrained(self.model_id, **load_kwargs)
        if device:
            self.model = self.model.to(device)
        self.model.eval()
        _WEIGHTS[cache_key] = (self.model, self.tokenizer)

    def generate(self, prompt: Prompt, **kwargs: Any) -> Completion:
        max_tokens = kwargs.get("max_tokens", self.max_tokens)
        temperature = kwargs.get("temperature", self.temperature)
        text = apply_chat(self.tokenizer, prompt.to_chat(), enable_thinking=self.enable_thinking)
        encoded = self.tokenizer(text, return_tensors="pt")
        encoded = {k: v.to(self.model.device) for k, v in encoded.items()}
        do_sample = temperature is not None and temperature > 0
        sampling: dict[str, Any] = {}
        if do_sample:
            sampling["temperature"] = temperature
            if self.top_p is not None:
                sampling["top_p"] = self.top_p
            if self.top_k is not None:
                sampling["top_k"] = self.top_k
        with self._torch.no_grad():
            out = self.model.generate(
                **encoded,
                max_new_tokens=max_tokens,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.pad_token_id,
                **sampling,
            )
        new_tokens = out[0, encoded["input_ids"].shape[-1] :]
        raw_text = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        answer = extract_answer(raw_text)
        n_new = int(new_tokens.shape[-1])
        return Completion(
            text=answer,
            raw={
                "model_id": self.model_id,
                "n_new_tokens": n_new,
                "enable_thinking": self.enable_thinking,
                # Kept so a zero score can be told apart from a parse failure.
                "raw_text": tail(raw_text),
                "truncated": n_new >= max_tokens,
            },
            usage={"input_tokens": int(encoded["input_ids"].shape[-1]), "output_tokens": n_new},
        )
