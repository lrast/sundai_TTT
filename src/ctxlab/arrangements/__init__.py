"""Context arrangements: pure functions from Example -> Prompt."""

from ctxlab.arrangements.base import DEFAULT_SYSTEM, Arrangement
from ctxlab.arrangements.formatting import build_prompt, format_passages

__all__ = ["Arrangement", "DEFAULT_SYSTEM", "build_prompt", "format_passages"]
