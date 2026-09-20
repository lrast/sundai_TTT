"""Hosted models via LiteLLM (OpenAI, Anthropic, OpenRouter, ...)."""

from __future__ import annotations

from typing import Any

from ctxlab.config import ModelConfig
from ctxlab.data.base import Completion, Prompt
from ctxlab.registry import register_model


@register_model("api")
class APIModel:
    def __init__(self, cfg: ModelConfig) -> None:
        if not cfg.model:
            raise ValueError(f"API model {cfg.name!r} requires `model` (LiteLLM model string)")
        self.name = cfg.name
        self.model = cfg.model
        self.temperature = cfg.temperature
        self.max_tokens = cfg.max_tokens
        self.extra = dict(cfg.extra)

    def generate(self, prompt: Prompt, **kwargs: Any) -> Completion:
        import litellm

        temperature = kwargs.get("temperature", self.temperature)
        max_tokens = kwargs.get("max_tokens", self.max_tokens)
        response = litellm.completion(
            model=self.model,
            messages=prompt.to_chat(),
            temperature=temperature,
            max_tokens=max_tokens,
            **self.extra,
        )
        choice = response.choices[0].message
        text = (choice.content or "").strip()
        usage = None
        if getattr(response, "usage", None) is not None:
            if hasattr(response.usage, "model_dump"):
                usage = response.usage.model_dump()
            else:
                usage = dict(response.usage)
        raw = {"id": getattr(response, "id", None), "model": getattr(response, "model", self.model)}
        return Completion(text=text, raw=raw, usage=usage)
