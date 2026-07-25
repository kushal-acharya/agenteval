"""Golden dataset schema + loading.

A dataset is a JSONL file of Case objects. Version the *file* (transit_v1,
transit_v2, ...) — scores are meaningless if the set silently changes.
"""

from __future__ import annotations

import json
from enum import Enum
from pathlib import Path

from pydantic import BaseModel, Field


class GradingMode(str, Enum):
    EXACT = "exact"                # normalized string equality
    KEYWORD = "keyword"            # all `keywords` must appear (case-insensitive)
    NUMERIC = "numeric"            # a number in the answer within `tolerance` of `expected`
    EXECUTION = "execution"        # S2: run gold SQL vs agent SQL, compare RESULT SETS
    JUDGE = "judge"                # S3: rubric-based LLM-as-judge (must be calibrated)
    UNANSWERABLE = "unanswerable"  # agent must decline, not fabricate


class Case(BaseModel):
    id: str
    question: str
    grading_mode: GradingMode
    # Gold answer (exact/numeric), gold SQL (execution), or rubric text (judge).
    # None for unanswerable cases.
    expected: str | float | None = None
    keywords: list[str] = Field(default_factory=list)  # for KEYWORD mode
    tolerance: float = 0.0                             # for NUMERIC mode
    tags: list[str] = Field(default_factory=list)      # e.g. sql, rag, safety
    difficulty: str = "medium"                         # easy | medium | hard
    # Which database this question is *about* (EXECUTION mode). It lives on the
    # case, not the run, because that's what it is: "which state had the most
    # fatal crashes" is inherently a question about fars.db. One dataset can
    # therefore mix databases, and the adapter and the scorer read the same
    # field — they cannot disagree about what was being queried.
    # Path is resolved relative to the current working directory.
    db: str | None = None


def load_dataset(path: str | Path) -> list[Case]:
    path = Path(path)
    cases: list[Case] = []
    with path.open() as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                cases.append(Case.model_validate(json.loads(line)))
            except Exception as e:  # fail loudly with location
                raise ValueError(f"{path}:{line_no}: invalid case: {e}") from e

    ids = [c.id for c in cases]
    dupes = {i for i in ids if ids.count(i) > 1}
    if dupes:
        raise ValueError(f"{path}: duplicate case ids: {sorted(dupes)}")
    return cases
