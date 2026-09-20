"""Chat templating and answer extraction, tested without weights."""

from __future__ import annotations

import pytest

from ctxlab.metrics.txlog import TxJointAccuracy, parse_answer
from ctxlab.models.decoding import apply_chat, extract_answer

GOLD = '{"bug_type": "NEGATIVE_BAL", "bug_location": "TX004"}'


class FakeTokenizer:
    """Stands in for a Qwen3 tokenizer's templating behaviour."""

    chat_template = "{{ messages }}"

    def __init__(self, *, supports_thinking: bool = True) -> None:
        self.supports_thinking = supports_thinking
        self.seen: dict[str, object] = {}

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt, **kwargs):
        if "enable_thinking" in kwargs and not self.supports_thinking:
            raise TypeError("unexpected keyword argument 'enable_thinking'")
        self.seen = dict(kwargs)
        return "".join(m["content"] for m in messages)


class TemplatelessTokenizer:
    chat_template = None


# --- extract_answer -----------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (GOLD, GOLD),
        (f"<think>checking TX002...</think>\nFinal: {GOLD}", GOLD),
        (f"Final: {GOLD}", GOLD),
        (f"reasoning here\nFinal: nope\nFinal: {GOLD}", GOLD),
        (f"```json\n{GOLD}\n```", GOLD),
        (f'"{GOLD}"', GOLD),
        (f"<think>done</think>{GOLD}", GOLD),
        ("", ""),
    ],
)
def test_extract_answer(raw: str, expected: str) -> None:
    assert extract_answer(raw) == expected


def test_unterminated_thinking_yields_no_answer() -> None:
    """A thinking run that exhausts its budget mid-thought has not answered.

    The metric parser is deliberately tolerant, so returning the reasoning
    would let it pick a transaction id out of the model's deliberation and
    score a non-answer as correct -- inflating exactly the arm the paper says
    should degrade.
    """
    truncated = "<think>Let me check TX004. The balance goes negative, so NEGATIVE_BAL"
    assert extract_answer(truncated) == ""
    assert parse_answer(extract_answer(truncated)).tx_id is None
    assert TxJointAccuracy().score(extract_answer(truncated), [GOLD]) == 0.0
    # Without the guard the same text would have scored a false positive.
    assert TxJointAccuracy().score(truncated, [GOLD]) == 1.0


# --- apply_chat ---------------------------------------------------------------


def test_apply_chat_passes_the_thinking_switch() -> None:
    tokenizer = FakeTokenizer()
    apply_chat(tokenizer, [{"role": "user", "content": "hi"}], enable_thinking=True)
    assert tokenizer.seen == {"enable_thinking": True}
    apply_chat(tokenizer, [{"role": "user", "content": "hi"}], enable_thinking=False)
    assert tokenizer.seen == {"enable_thinking": False}


def test_apply_chat_omits_the_switch_when_unset() -> None:
    tokenizer = FakeTokenizer()
    apply_chat(tokenizer, [{"role": "user", "content": "hi"}])
    assert tokenizer.seen == {}


def test_apply_chat_refuses_to_silently_drop_thinking() -> None:
    """A silent fallback would leave the thinking arm decoding identically to
    the in-context arm, and the sweep would report the difference between two
    identical configurations as a finding."""
    with pytest.raises(ValueError, match="enable_thinking"):
        apply_chat(
            FakeTokenizer(supports_thinking=False),
            [{"role": "user", "content": "hi"}],
            enable_thinking=True,
        )
    with pytest.raises(ValueError, match="no chat template"):
        apply_chat(
            TemplatelessTokenizer(),
            [{"role": "user", "content": "hi"}],
            enable_thinking=True,
        )


def test_apply_chat_falls_back_without_a_template() -> None:
    text = apply_chat(TemplatelessTokenizer(), [{"role": "user", "content": "hi"}])
    assert text == "user: hi\nassistant:"
