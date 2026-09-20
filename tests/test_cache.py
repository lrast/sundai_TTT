from pathlib import Path

from ctxlab.cache import CompletionCache, prompt_cache_key
from ctxlab.config import ModelConfig
from ctxlab.data.base import Message, Prompt
from ctxlab.models.mock import MockModel


def test_cache_roundtrip(tmp_path: Path) -> None:
    cache = CompletionCache(tmp_path)
    prompt = Prompt(system="s", messages=(Message("user", "hello"),), meta={"n_gold": 0})
    model = MockModel(ModelConfig(name="mock", kind="mock", extra={"behavior": "unknown"}))
    first, hit1 = cache.get_or_generate(model, prompt, {"temperature": 0.0, "max_tokens": 16})
    second, hit2 = cache.get_or_generate(model, prompt, {"temperature": 0.0, "max_tokens": 16})
    assert hit1 is False
    assert hit2 is True
    assert first.text == second.text == "unknown"


def test_meta_is_not_part_of_cache_key() -> None:
    a = Prompt(system="s", messages=(Message("user", "q"),), meta={"n_gold": 0})
    b = Prompt(system="s", messages=(Message("user", "q"),), meta={"n_gold": 9})
    params = {"temperature": 0.0, "max_tokens": 8}
    assert prompt_cache_key(a, "mock", params) == prompt_cache_key(b, "mock", params)
