"""Chat templating and answer extraction, kept free of torch.

Deciding what counts as "the model's answer" is the step most likely to be
silently wrong, and a wrong answer-extractor produces an accuracy curve that
collapses with context length for reasons that have nothing to do with the
thing being measured. So it lives here, as ordinary functions, and is tested
without weights.
"""

from __future__ import annotations

from typing import Any

THINK_OPEN = "<think>"
THINK_CLOSE = "</think>"
FINAL_MARKER = "Final:"
_FENCES = ("```json", "```JSON", "```")


def apply_chat(
    tokenizer: Any,
    messages: list[dict[str, str]],
    *,
    enable_thinking: bool | None = None,
) -> str:
    """Render chat messages to a prompt string.

    Raises when `enable_thinking` is requested but the tokenizer does not
    support it. Falling back silently would leave the "thinking" arm decoding
    exactly like the in-context arm, and the sweep would report a difference
    between two identical configurations as a finding.
    """
    template = getattr(tokenizer, "chat_template", None)
    if not (hasattr(tokenizer, "apply_chat_template") and template):
        if enable_thinking is not None:
            raise ValueError("enable_thinking was requested but the tokenizer has no chat template")
        return "\n".join(f"{m['role']}: {m['content']}" for m in messages) + "\nassistant:"

    if enable_thinking is None:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )
    except TypeError as exc:
        raise ValueError(
            "This tokenizer's chat template does not accept `enable_thinking`. "
            "The paper's thinking arm uses Qwen3's /think switch; pick a Qwen3 "
            "model or drop `enable_thinking` from the model's `extra`."
        ) from exc


def tail(text: str, limit: int = 2000) -> str:
    """Keep the end of a long generation, for the record.

    A thinking run can emit thousands of tokens per example, which would bloat
    `records.jsonl` past the point of being useful. The tail is the part worth
    keeping: it is where `Final:` lives, so it is what distinguishes a genuine
    wrong answer from an extraction failure.
    """
    if len(text) <= limit:
        return text
    return f"...[{len(text) - limit} chars elided]..." + text[-limit:]


def strip_fences(text: str) -> str:
    text = text.strip()
    for fence in _FENCES:
        if text.startswith(fence):
            text = text[len(fence) :]
            break
    if text.endswith("```"):
        text = text[: -len("```")]
    return text.strip()


def extract_answer(text: str) -> str:
    """The model's final answer, with any reasoning removed.

    An unterminated `<think>` means the generation budget ran out mid-thought,
    so there is no answer. Returning the reasoning instead would let the
    tolerant metric parser pick a transaction id out of the model's
    deliberation and score a non-answer as correct.
    """
    if THINK_CLOSE in text:
        text = text.rsplit(THINK_CLOSE, 1)[1]
    elif THINK_OPEN in text:
        return ""
    if FINAL_MARKER in text:
        text = text.rsplit(FINAL_MARKER, 1)[1]
    return strip_fences(text).strip().strip('"').strip()
