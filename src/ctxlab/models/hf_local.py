"""Local Hugging Face causal LM.

Exposes `.model` and `.tokenizer` as public attributes. That is the only
seam the test-time-training phase should use to reach weights; the rest of
ctxlab talks to `generate()` only.
"""

from __future__ import annotations

from typing import Any

from ctxlab.config import ModelConfig
from ctxlab.data.base import Completion, Prompt
from ctxlab.registry import register_model


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
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        device_map = self.extra.pop("device_map", "auto")
        torch_dtype = self.extra.pop("torch_dtype", None)
        dtype = getattr(torch, torch_dtype) if isinstance(torch_dtype, str) else torch_dtype
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            device_map=device_map,
            torch_dtype=dtype,
        )
        self.model.eval()
        self._torch = torch

    def generate(self, prompt: Prompt, **kwargs: Any) -> Completion:
        max_tokens = kwargs.get("max_tokens", self.max_tokens)
        temperature = kwargs.get("temperature", self.temperature)
        messages = prompt.to_chat()
        if hasattr(self.tokenizer, "apply_chat_template") and self.tokenizer.chat_template:
            text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        else:
            text = "\n".join(f"{m['role']}: {m['content']}" for m in messages) + "\nassistant:"
        encoded = self.tokenizer(text, return_tensors="pt")
        encoded = {k: v.to(self.model.device) for k, v in encoded.items()}
        do_sample = temperature is not None and temperature > 0
        with self._torch.no_grad():
            out = self.model.generate(
                **encoded,
                max_new_tokens=max_tokens,
                do_sample=do_sample,
                temperature=temperature if do_sample else None,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        new_tokens = out[0, encoded["input_ids"].shape[-1] :]
        decoded = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        return Completion(
            text=decoded,
            raw={"model_id": self.model_id, "n_new_tokens": int(new_tokens.shape[-1])},
            usage={"output_tokens": int(new_tokens.shape[-1])},
        )
