"""Content-addressed disk cache for model completions."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from ctxlab.data.base import Completion, Prompt
from ctxlab.models.base import LanguageModel


def prompt_cache_key(
    prompt: Prompt,
    model_name: str,
    gen_params: dict[str, Any],
) -> str:
    payload = {
        "system": prompt.system,
        "messages": [m.to_dict() for m in prompt.messages],
        "model": model_name,
        "params": {k: gen_params[k] for k in sorted(gen_params)},
    }
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


class CompletionCache:
    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        return self.root / key[:2] / f"{key}.json"

    def get(self, key: str) -> Completion | None:
        path = self._path(key)
        if not path.exists():
            return None
        data = json.loads(path.read_text())
        return Completion(text=data["text"], raw=data.get("raw"), usage=data.get("usage"))

    def put(self, key: str, completion: Completion) -> None:
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {"text": completion.text, "raw": completion.raw, "usage": completion.usage},
                ensure_ascii=False,
            )
        )

    def get_or_generate(
        self,
        model: LanguageModel,
        prompt: Prompt,
        gen_params: dict[str, Any],
    ) -> tuple[Completion, bool]:
        key = prompt_cache_key(prompt, model.name, gen_params)
        hit = self.get(key)
        if hit is not None:
            return hit, True
        completion = model.generate(prompt, **gen_params)
        self.put(key, completion)
        return completion, False
