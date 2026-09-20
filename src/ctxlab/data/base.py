"""Shared data types for datasets, arrangements, and models."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Passage:
    title: str
    text: str
    is_gold: bool
    score: float | None = None


@dataclass(frozen=True)
class Example:
    uid: str
    question: str
    answers: list[str]
    passages: list[Passage]


@dataclass(frozen=True)
class Message:
    role: str
    content: str

    def to_dict(self) -> dict[str, str]:
        return {"role": self.role, "content": self.content}


@dataclass(frozen=True)
class Prompt:
    """Model-ready prompt plus analysis metadata.

    `meta` is never sent to the model. Cache keys hash `system` + `messages`
    only, together with model identity and generation params.
    """

    system: str | None
    messages: tuple[Message, ...]
    meta: dict[str, Any] = field(default_factory=dict)

    def to_chat(self) -> list[dict[str, str]]:
        msgs: list[dict[str, str]] = []
        if self.system:
            msgs.append({"role": "system", "content": self.system})
        msgs.extend(m.to_dict() for m in self.messages)
        return msgs

    def to_record(self) -> dict[str, Any]:
        return {
            "system": self.system,
            "messages": [m.to_dict() for m in self.messages],
            "meta": self.meta,
        }


@dataclass(frozen=True)
class Completion:
    text: str
    raw: dict[str, Any] | None = None
    usage: dict[str, Any] | None = None
