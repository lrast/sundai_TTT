"""Long Code Arena bug-localization loader (JetBrains-Research/lca-bug-localization).

Each row is a real GitHub issue plus the repository snapshot (`base_sha`)
where the bug reproduces and the list of files the fix changed. Framed for
ctxlab:

    question  = issue title + body (user-written symptom, truncated)
    passages  = the gold file(s) plus same-package .py neighbors from the
                repo snapshot, each truncated to a character budget
    answers   = repo-relative paths of the changed files

Repo snapshots are read from local checkouts, not the dataset's tarballs:
`dataset.extra.repos_dir` names a directory containing one checkout per repo
at its `base_sha`, laid out as `<repos_dir>/<owner>__<name>`. Clone them with:

    git clone https://github.com/<owner>/<name> <repos_dir>/<owner>__<name>
    git -C <repos_dir>/<owner>__<name> checkout <base_sha>

Rows whose snapshot directory is missing are skipped (and counted), so a
handful of cloned repos is enough for a pilot.

`dataset.hub_id` may be the HF id (default) or a local `.json` file of
LCA-shaped rows (tests, smoke runs).

extra knobs: repos_dir (required for real rows), n_files (haystack size,
default 10), max_file_chars (per-passage truncation, default 6000),
max_issue_chars (default 4000), max_changed_files (default 1: single-needle).
"""

from __future__ import annotations

import ast
import json
import re
from pathlib import Path

from ctxlab.config import DatasetConfig
from ctxlab.data.base import Example, Passage
from ctxlab.registry import register_dataset

N_FILES = 10
MAX_FILE_CHARS = 6000
MAX_ISSUE_CHARS = 4000


def _norm_key(key: str) -> str:
    return re.sub(r"[\s_\-]+", "_", key.strip().lower())


def _norm_row(row: dict) -> dict:
    return {_norm_key(k): v for k, v in row.items() if isinstance(k, str)}


def _changed_files(row: dict) -> list[str]:
    raw = row.get("changed_files")
    if raw is None:
        return []
    if isinstance(raw, str):
        # The HF dataset stores the list as its Python repr: "['a.py', 'b.py']".
        try:
            raw = ast.literal_eval(raw)
        except (ValueError, SyntaxError):
            try:
                raw = json.loads(raw)
            except json.JSONDecodeError:
                raw = [p.strip(" '\"[]") for p in re.split(r"[\n,]+", raw)]
    return [str(p).strip().strip("'\"") for p in raw if str(p).strip()]


def _issue_text(row: dict, max_chars: int) -> str:
    title = str(row.get("issue_title") or "").strip()
    body = str(row.get("issue_body") or "").strip()
    text = f"{title}\n\n{body}".strip()
    return text[:max_chars]


def _repo_dir(row: dict, repos_dir: Path) -> Path:
    owner = str(row.get("repo_owner") or "")
    name = str(row.get("repo_name") or "")
    return repos_dir / f"{owner}__{name}"


def _read_file(path: Path, max_chars: int) -> str | None:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    return text[:max_chars]


def _sibling_files(repo: Path, gold_paths: list[str], budget: int) -> list[str]:
    """Same-directory .py files first, then the rest of the package tree."""
    golds = set(gold_paths)
    out: list[str] = []
    seen: set[str] = set(golds)
    dirs = [Path(g).parent for g in gold_paths]
    for d in dirs:
        for f in sorted((repo / d).glob("*.py")):
            rel = str(f.relative_to(repo))
            if rel not in seen:
                seen.add(rel)
                out.append(rel)
            if len(out) >= budget:
                return out
    for d in dirs:
        parent = repo / d.parent if str(d) != "." else repo
        for f in sorted(parent.rglob("*.py")):
            rel = str(f.relative_to(repo))
            if rel not in seen:
                seen.add(rel)
                out.append(rel)
            if len(out) >= budget:
                return out
    return out


def _stable_seed(uid: str, seed: int) -> int:
    import hashlib

    return int(hashlib.sha256(f"{seed}|{uid}".encode()).hexdigest()[:8], 16)


def _anonymize(passages: list[Passage], uid: str, seed: int) -> tuple[list[Passage], list[str]]:
    """Replace real paths with neutral labels so issue hints cannot name the answer.

    Labels are assigned by a per-example seeded shuffle, so the gold's label
    carries no information. Returns the relabeled passages and the gold labels.
    """
    import random as _random

    labels = [f"file_{i:02d}.py" for i in range(1, len(passages) + 1)]
    _random.Random(_stable_seed(uid, seed)).shuffle(labels)
    out = [
        Passage(title=label, text=p.text, is_gold=p.is_gold)
        for label, p in zip(labels, passages, strict=True)
    ]
    return out, [p.title for p in out if p.is_gold]


def _row_to_example(
    row: dict,
    repos_dir: Path,
    n_files: int,
    max_file_chars: int,
    max_issue_chars: int,
) -> Example | None:
    changed = [p for p in _changed_files(row) if p.endswith(".py")]
    if not changed:
        return None
    repo = _repo_dir(row, repos_dir)
    if not repo.is_dir():
        return None
    golds = []
    for rel in changed:
        text = _read_file(repo / rel, max_file_chars)
        if text:
            golds.append(Passage(title=rel, text=text, is_gold=True))
    if not golds:
        return None
    gold_paths = [p.title for p in golds]
    distractors = []
    for rel in _sibling_files(repo, gold_paths, max(n_files - len(golds), 0)):
        text = _read_file(repo / rel, max_file_chars)
        if text:
            distractors.append(Passage(title=rel, text=text, is_gold=False))
    uid = f"{row.get('repo_owner')}/{row.get('repo_name')}@{str(row.get('base_sha'))[:10]}"
    return Example(
        uid=uid,
        question=_issue_text(row, max_issue_chars),
        answers=gold_paths,
        passages=golds + distractors,
    )


@register_dataset("lca_bugloc")
def load_lca_bugloc(cfg: DatasetConfig) -> list[Example]:
    source = cfg.hub_id or "JetBrains-Research/lca-bug-localization"
    if source.endswith(".json"):
        rows = json.loads(Path(source).read_text())
    else:
        from datasets import load_dataset

        ds = load_dataset(source, cfg.config or "py", split=cfg.split, cache_dir=cfg.cache_dir)
        rows = [dict(r) for r in ds]
    repos_dir = Path(str(cfg.extra.get("repos_dir", ".cache/lca-repos")))
    anonymize = bool(cfg.extra.get("anonymize_titles", False))
    n_files = int(cfg.extra.get("n_files", N_FILES))
    max_file_chars = int(cfg.extra.get("max_file_chars", MAX_FILE_CHARS))
    max_issue_chars = int(cfg.extra.get("max_issue_chars", MAX_ISSUE_CHARS))
    max_changed = int(cfg.extra.get("max_changed_files", 1))

    examples: list[Example] = []
    n_skipped = 0
    for raw in rows:
        row = _norm_row(raw)
        changed = [p for p in _changed_files(row) if p.endswith(".py")]
        if not changed or len(changed) > max_changed:
            continue
        ex = _row_to_example(row, repos_dir, n_files, max_file_chars, max_issue_chars)
        if ex is None:
            n_skipped += 1
            continue
        if anonymize:
            passages, gold_labels = _anonymize(list(ex.passages), ex.uid, cfg.seed)
            ex = Example(uid=ex.uid, question=ex.question, answers=gold_labels, passages=passages)
        examples.append(ex)
        if cfg.n is not None and len(examples) >= cfg.n:
            break
    if not examples:
        raise RuntimeError(
            f"No usable LCA examples. Skipped {n_skipped} rows without a local repo "
            f"checkout under {repos_dir} (see ctxlab/data/lca.py docstring for cloning)."
        )
    return examples
