"""Query-only test-time training, on a tiny randomly-initialised model.

Skipped unless the `local` extra is installed, which CI does not install.
These are mechanism tests: they check that the adapter updates what the paper
says it updates and nothing else, that the KV cache genuinely survives the
updates, and that decoding against the reused cache is exact.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from ctxlab.data.base import Message, Prompt  # noqa: E402
from ctxlab.ttt.qttt import QUERY_SUFFIX, QueryOnlyTTT  # noqa: E402


class TinyTokenizer:
    """Byte-level stand-in; `apply_chat` takes its templateless fallback."""

    chat_template = None
    pad_token_id = 0

    def __call__(self, text: str, return_tensors: str | None = None):
        ids = torch.tensor([[(ord(c) % 100) + 1 for c in text]])
        return SimpleNamespace(input_ids=ids)

    def decode(self, tokens, skip_special_tokens: bool = True) -> str:
        return "".join(chr(int(t) % 100 + 32) for t in tokens)


def _tiny_lm():
    from transformers import AutoConfig, AutoModelForCausalLM

    torch.manual_seed(0)
    config = AutoConfig.for_model(
        "qwen3",
        hidden_size=64,
        num_hidden_layers=3,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        intermediate_size=128,
        vocab_size=128,
        max_position_embeddings=1024,
    )
    model = AutoModelForCausalLM.from_config(config)
    model.eval()
    return SimpleNamespace(
        model=model,
        tokenizer=TinyTokenizer(),
        model_id="tiny-qwen3",
        enable_thinking=None,
        top_p=None,
        top_k=None,
        max_tokens=6,
        temperature=0.0,
    )


# A repeating pattern, so a few gradient steps have something to latch onto.
PROMPT = Prompt(system=None, messages=(Message(role="user", content="abcdefgh" * 40),))


@pytest.fixture
def adapter():
    return QueryOnlyTTT(_tiny_lm(), span=16, steps=8, lr=1e-3, seed=0)


def _snapshot(model) -> dict[str, object]:
    return {name: p.detach().clone() for name, p in model.named_parameters()}


def test_gradients_reach_only_query_projections(adapter: QueryOnlyTTT) -> None:
    adapter.steps = 0
    adapter.fit(PROMPT)
    model = adapter.lm.model
    for param in model.parameters():
        param.requires_grad_(False)
    for _, param in adapter.query_parameters():
        param.requires_grad_(True)

    adapter._span_loss(adapter._prefill, 1, 16).backward()

    got = {name for name, p in model.named_parameters() if p.grad is not None}
    assert got == {name for name, _ in adapter.query_parameters()}
    assert got and all(name.endswith(QUERY_SUFFIX) for name in got)


def test_only_query_projections_change(adapter: QueryOnlyTTT) -> None:
    """The observable form of the claim: everything but W_Q is untouched."""
    before = _snapshot(adapter.lm.model)
    adapter.fit(PROMPT)
    after = _snapshot(adapter.lm.model)
    changed = {n for n in before if not torch.equal(before[n], after[n])}
    assert changed == {name for name, _ in adapter.query_parameters()}


def test_kv_cache_survives_the_updates(adapter: QueryOnlyTTT) -> None:
    """The whole reason qTTT is cheap: K and V do not depend on W_Q, so one
    prefill serves every step. If a step mutated the cache, later steps would
    silently train against corrupted evidence."""
    adapter.steps = 0
    adapter.fit(PROMPT)
    state = adapter._prefill
    assert state is not None
    original = [(k.clone(), v.clone()) for k, v in state.kv]

    adapter.steps = 8
    adapter._adapt(state, span=16)

    assert state.steps_run == 8
    for (k0, v0), (k1, v1) in zip(original, state.kv, strict=True):
        assert torch.equal(k0, k1)
        assert torch.equal(v0, v1)


def test_restore_puts_the_base_weights_back(adapter: QueryOnlyTTT) -> None:
    """Examples must stay independent; a leaked update would let one example's
    adaptation help the next and quietly inflate the whole arm."""
    before = _snapshot(adapter.lm.model)
    adapter.fit(PROMPT)
    assert any(not torch.equal(before[n], p.detach()) for n, p in adapter.query_parameters()), (
        "fit should have changed the query projections"
    )
    adapter.restore()
    after = _snapshot(adapter.lm.model)
    for name in before:
        assert torch.equal(before[name], after[name]), name


def test_adaptation_reduces_the_loss_on_a_held_out_span(adapter: QueryOnlyTTT) -> None:
    """Measured on one *fixed* span before and after.

    The training losses cannot show this: every step samples a different span,
    so comparing the first to the last compares two unrelated pieces of text.
    A fixed probe is the only way to tell adaptation from sampling noise.
    """
    adapter.steps = 0
    adapter.fit(PROMPT)
    state = adapter._prefill
    probe_start, probe_span = 40, 16
    with torch.no_grad():
        before = float(adapter._span_loss(state, probe_start, probe_span))

    adapter.steps = 24
    adapter._adapt(state, span=16)
    with torch.no_grad():
        after = float(adapter._span_loss(state, probe_start, probe_span))

    assert len(state.losses) == 24
    assert after < before, (before, after)


def test_span_sampling_is_deterministic_for_a_prompt(adapter: QueryOnlyTTT) -> None:
    """The completion cache assumes generate is a function of the prompt."""
    adapter.fit(PROMPT)
    first = list(adapter._prefill.losses)
    adapter.restore()
    second_adapter = QueryOnlyTTT(_tiny_lm(), span=16, steps=8, lr=1e-3, seed=0)
    second_adapter.fit(PROMPT)
    assert first == pytest.approx(second_adapter._prefill.losses)


def test_a_different_seed_samples_different_spans() -> None:
    a = QueryOnlyTTT(_tiny_lm(), span=16, steps=8, lr=1e-3, seed=0)
    b = QueryOnlyTTT(_tiny_lm(), span=16, steps=8, lr=1e-3, seed=1)
    a.fit(PROMPT)
    b.fit(PROMPT)
    assert a._prefill.losses != b._prefill.losses


def test_decoding_against_the_reused_cache_is_exact() -> None:
    """With zero steps the adapter must reproduce an ordinary forward pass. If
    the cache reuse were subtly wrong, every qTTT number would be wrong in a
    way no accuracy metric could distinguish from the method not working."""
    adapter = QueryOnlyTTT(_tiny_lm(), span=16, steps=0, lr=1e-3, seed=0)
    model, tokenizer = adapter.lm.model, adapter.lm.tokenizer
    ids = tokenizer(PROMPT.messages[0].content + "\nassistant:").input_ids
    with torch.no_grad():
        plain = model.generate(input_ids=ids, max_new_tokens=6, do_sample=False, pad_token_id=0)[
            0, ids.shape[-1] :
        ]

    adapter.fit(PROMPT)
    completion = adapter.generate(PROMPT, max_tokens=6, temperature=0.0)
    assert completion.raw["raw_text"] == tokenizer.decode(plain).strip()
    assert completion.raw["ttt_steps"] == 0


def test_a_short_context_shrinks_the_span_and_says_so() -> None:
    """k=128 does not fit a tiny context. Shrinking is better than crashing,
    but the record has to show the span actually used: the compute match to
    the thinking arm is T_think ~= 2 * N_TTT * k, so reporting the requested
    k after running a shorter one would overstate qTTT's budget."""
    adapter = QueryOnlyTTT(_tiny_lm(), span=128, steps=4, lr=1e-3, seed=0)
    short = Prompt(system=None, messages=(Message(role="user", content="hi" * 6),))
    adapter.fit(short)
    state = adapter._prefill
    assert state is not None
    assert 0 < state.span_used < 128
    completion = adapter.generate(short, max_tokens=2, temperature=0.0)
    assert completion.raw["ttt_span_requested"] == 128
    assert completion.raw["ttt_span_used"] == state.span_used


def test_a_context_too_short_for_any_span_skips_adaptation() -> None:
    adapter = QueryOnlyTTT(_tiny_lm(), span=128, steps=4, lr=1e-3, seed=0)

    class OneToken(TinyTokenizer):
        def __call__(self, text, return_tensors=None):
            return SimpleNamespace(input_ids=torch.tensor([[5, 6]]))

    adapter.lm.tokenizer = OneToken()
    adapter.fit(Prompt(system=None, messages=(Message(role="user", content="x"),)))
    assert adapter._prefill is not None
    assert adapter._prefill.steps_run == 0
    assert adapter._prefill.span_used == 0
    assert adapter._backup is None


def test_generate_before_fit_is_an_error() -> None:
    adapter = QueryOnlyTTT(_tiny_lm(), span=16, steps=1, lr=1e-3, seed=0)
    with pytest.raises(RuntimeError, match="fit"):
        adapter.generate(PROMPT)
