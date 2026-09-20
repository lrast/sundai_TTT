"""PyResBugs bug-localization loader.

Each row of OSS-forge/PyResBugs (github.com/dessertlab/PyResBugs) is one
residual Python bug: a faulty method, its fixed version, project/commit
provenance, and three natural-language descriptions of the fault. We frame
bug localization as retrieval QA over code:

    question  = one description (`dataset.config`: implementation | contextual | high)
    passages  = the faulty method (gold) hidden among fixed methods sampled
                from other rows, same project preferred (self-contained haystack)
    answers   = the faulty method's name

`dataset.hub_id` may be a Hub id (default `OSS-forge/PyResBugs`) or a path to
a local `.json` file with a list of PyResBugs-shaped rows (tests, smoke runs).

Full-repo haystacks (clone `Project` at the parent of `Commit_sha` and sample
real neighboring code) are a follow-up; `buggy_file_from_patch` already
recovers the faulty file path from `Diff_patch` for that phase.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
from pathlib import Path

from ctxlab.config import DatasetConfig
from ctxlab.data.base import Example, Passage
from ctxlab.registry import register_dataset

# Keep examples HotpotQA-shaped: 1-2 gold among ~10 passages.
N_DISTRACTORS = 9

DESCRIPTION_LEVELS = {
    "implementation": "implementation_level_description",
    "contextual": "contextual_level_description",
    "high": "high_level_description",
}

_DIFF_HEADER = re.compile(r"^diff --git a/(\S+) b/", re.MULTILINE)
_MINUS_HEADER = re.compile(r"^--- a/(\S+)", re.MULTILINE)


def _norm_key(key: str) -> str:
    return re.sub(r"[\s_\-]+", "_", key.strip().lower())


def _norm_row(row: dict) -> dict:
    """Index a row by normalized key so xlsx/HF spelling variants all work."""
    return {_norm_key(k): v for k, v in row.items() if isinstance(k, str)}


def buggy_file_from_patch(patch: str | None) -> str | None:
    """Recover the faulty file path from a `Diff_patch` blob."""
    if not patch:
        return None
    match = _DIFF_HEADER.search(patch) or _MINUS_HEADER.search(patch)
    return match.group(1) if match else None


def mask_answer_in_text(text: str, name: str) -> str:
    """Replace mentions of the buggy method's name so the description
    cannot leak the answer (with or without a trailing call parenthesis)."""
    pattern = re.compile(re.escape(name) + r"(\(\))?", re.IGNORECASE)
    return pattern.sub("the affected function", text)


def _description(row: dict, level: str) -> str | None:
    field = DESCRIPTION_LEVELS.get(level)
    if field is None:
        known = ", ".join(sorted(DESCRIPTION_LEVELS))
        raise KeyError(f"Unknown description level {level!r}. Known: {known}")
    value = row.get(field)
    return str(value).strip() if value else None


def _usable(row: dict, level: str) -> bool:
    return bool(row.get("fixed_method") and row.get("faulty_code") and _description(row, level))


def _uid(row: dict) -> str:
    digest = hashlib.sha256(str(row.get("faulty_code")).encode()).hexdigest()[:8]
    sha = str(row.get("commit_sha") or "")[:10]
    return f"{row.get('project')}@{sha}:{row.get('fixed_method')}:{digest}"


def _stable_seed(root: int, uid: str) -> int:
    return int(hashlib.sha256(f"{root}|{uid}".encode()).hexdigest()[:8], 16)


ANONYMOUS_QUESTION = "One of these functions contains a residual bug. Which one?"


def _row_to_example(
    row: dict,
    others: list[dict],
    level: str,
    seed: int,
    n_distractors: int,
    mask_answer: bool,
    question_style: str,
    distractor_scope: str,
    include_fixed_gold: bool,
) -> Example:
    uid = _uid(row)
    name = str(row["fixed_method"])
    rng = random.Random(_stable_seed(seed, uid))

    fixed_text = str(row.get("fault_free_code") or "")
    twin = include_fixed_gold and bool(fixed_text)
    if twin:
        # Hard negative: the gold's own fixed version. Both share a name, so
        # they get version labels and the buggy one is assigned by seed.
        buggy_label, clean_label = (
            (f"{name} [version A]", f"{name} [version B]")
            if rng.random() < 0.5
            else (f"{name} [version B]", f"{name} [version A]")
        )
        gold_title = buggy_label
        gold = Passage(title=buggy_label, text=str(row["faulty_code"]), is_gold=True)
        distractors = [Passage(title=clean_label, text=fixed_text, is_gold=False)]
    else:
        gold_title = name
        gold = Passage(title=name, text=str(row["faulty_code"]), is_gold=True)
        distractors = []

    same_project = [o for o in others if o.get("project") == row.get("project")]
    other_project = [o for o in others if o.get("project") != row.get("project")]
    rng.shuffle(same_project)
    rng.shuffle(other_project)
    pool = same_project if distractor_scope == "same_project" else same_project + other_project

    seen_titles = {name, gold_title} | {p.title for p in distractors}
    for candidate in pool:
        if len(distractors) >= n_distractors:
            break
        title = str(candidate.get("fixed_method") or "")
        # Distractors are *fixed* methods: clean code that never contains the bug.
        text = str(candidate.get("fault_free_code") or "")
        if not title or not text or title in seen_titles:
            continue
        seen_titles.add(title)
        distractors.append(Passage(title=title, text=text, is_gold=False))

    passages = [gold] + distractors
    rng.shuffle(passages)
    if question_style == "anonymous":
        question = ANONYMOUS_QUESTION
    else:
        question = _description(row, level) or ""
        if mask_answer:
            question = mask_answer_in_text(question, name)
    return Example(
        uid=uid,
        question=question,
        answers=[gold_title],
        passages=passages,
    )


def examples_from_rows(
    rows: list[dict],
    *,
    level: str = "contextual",
    n: int | None = None,
    seed: int = 0,
    n_distractors: int = N_DISTRACTORS,
    mask_answer: bool = False,
    question_style: str = "description",
    distractor_scope: str = "any",
    include_fixed_gold: bool = False,
) -> list[Example]:
    """Build bug-localization examples from PyResBugs-shaped dicts."""
    normed = [_norm_row(r) for r in rows]
    need_description = question_style != "anonymous"
    usable = [
        r
        for r in normed
        if r.get("fixed_method")
        and r.get("faulty_code")
        and (not need_description or _description(r, level))
    ]
    order = list(range(len(usable)))
    random.Random(seed).shuffle(order)
    if n is not None:
        order = order[: min(n, len(order))]
    examples = []
    for i in order:
        others = usable[:i] + usable[i + 1 :]
        examples.append(
            _row_to_example(
                usable[i],
                others,
                level,
                seed,
                n_distractors,
                mask_answer,
                question_style,
                distractor_scope,
                include_fixed_gold,
            )
        )
    return examples


@register_dataset("pyresbugs")
def load_pyresbugs(cfg: DatasetConfig) -> list[Example]:
    source = cfg.hub_id or "OSS-forge/PyResBugs"
    if source.endswith(".json"):
        rows = json.loads(Path(source).read_text())
    else:
        from datasets import load_dataset

        ds = load_dataset(source, split=cfg.split, cache_dir=cfg.cache_dir)
        rows = [dict(r) for r in ds]
    level = cfg.config or "contextual"
    return examples_from_rows(
        rows,
        level=level,
        n=cfg.n,
        seed=cfg.seed,
        n_distractors=int(cfg.extra.get("n_distractors", N_DISTRACTORS)),
        mask_answer=bool(cfg.extra.get("mask_answer", False)),
        question_style=str(cfg.extra.get("question_style", "description")),
        distractor_scope=str(cfg.extra.get("distractor_scope", "any")),
        include_fixed_gold=bool(cfg.extra.get("include_fixed_gold", False)),
    )
