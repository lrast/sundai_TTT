"""Tiny in-repo dataset so smoke tests never touch the network."""

from __future__ import annotations

from ctxlab.config import DatasetConfig
from ctxlab.data.base import Example, Passage
from ctxlab.registry import register_dataset

TOY_EXAMPLES = [
    Example(
        uid="toy-paris",
        question="What is the capital of France?",
        answers=["Paris"],
        passages=[
            Passage("Paris", "Paris is the capital and largest city of France.", True),
            Passage("Lyon", "Lyon is a city in east-central France.", False),
            Passage("Bordeaux", "Bordeaux is known for wine in southwestern France.", False),
        ],
    ),
    Example(
        uid="toy-curie",
        question="Who discovered radium with Pierre Curie?",
        answers=["Marie Curie"],
        passages=[
            Passage(
                "Marie Curie",
                "Marie Curie jointly discovered radium with Pierre Curie.",
                True,
            ),
            Passage("Einstein", "Albert Einstein developed the theory of relativity.", False),
            Passage("Newton", "Isaac Newton formulated the laws of motion.", False),
        ],
    ),
    Example(
        uid="toy-python",
        question="Who created the Python programming language?",
        answers=["Guido van Rossum"],
        passages=[
            Passage(
                "Guido van Rossum",
                "Python was created by Guido van Rossum and first released in 1991.",
                True,
            ),
            Passage("Linux", "Linux is a kernel created by Linus Torvalds.", False),
            Passage("Ruby", "Ruby is a programming language created by Yukihiro Matsumoto.", False),
        ],
    ),
    Example(
        uid="toy-multihop",
        question="What is the capital of the country where the Eiffel Tower stands?",
        answers=["Paris"],
        passages=[
            Passage(
                "Paris",
                "Paris is the capital of France, the country where the Eiffel Tower stands.",
                True,
            ),
            Passage("Eiffel Tower", "The Eiffel Tower stands in Paris, France.", True),
            Passage("Rome", "Rome is the capital of Italy.", False),
            Passage("Madrid", "Madrid is the capital of Spain.", False),
        ],
    ),
]


@register_dataset("toy")
def load_toy(cfg: DatasetConfig) -> list[Example]:
    examples = list(TOY_EXAMPLES)
    if cfg.n is not None:
        examples = examples[: cfg.n]
    return examples
