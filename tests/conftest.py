"""Shared fixtures."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ctxlab.data.base import Example, Passage
from ctxlab.data.hotpotqa import examples_from_rows
from ctxlab.data.toy import TOY_EXAMPLES

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def toy_examples() -> list[Example]:
    return list(TOY_EXAMPLES)


@pytest.fixture
def toy_paris() -> Example:
    return TOY_EXAMPLES[0]


@pytest.fixture
def hotpot_example() -> Example:
    row = json.loads((FIXTURES / "hotpot_tiny.json").read_text())
    return examples_from_rows([row])[0]


@pytest.fixture
def mixed_example() -> Example:
    return Example(
        uid="mix",
        question="Q?",
        answers=["A"],
        passages=[
            Passage("d1", "d1 text", False),
            Passage("g1", "g1 text", True),
            Passage("d2", "d2 text", False),
            Passage("g2", "g2 text", True),
            Passage("d3", "d3 text", False),
        ],
    )
