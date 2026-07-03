"""Layered scorers — cheapest first.

Deterministic scorers live here (S1). EXECUTION (S2) and JUDGE (S3) are
routed but return skipped=True until implemented, so they never silently
inflate or deflate the pass rate.
"""

from __future__ import annotations

import re

from pydantic import BaseModel

from .adapter import AgentResult
from .cases import Case, GradingMode

DECLINE_MARKERS = [
    "cannot", "can't", "unable to", "don't know", "do not know",
    "don't have", "do not have", "not available", "no data",
    "insufficient", "not enough information",
]

_NUM_RE = re.compile(r"-?\d[\d,]*\.?\d*")


def _normalize(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip().strip(".").lower()


class Score(BaseModel):
    passed: bool
    skipped: bool = False
    scorer: str
    reason: str = ""


def score_exact(case: Case, result: AgentResult) -> Score:
    expected = _normalize(str(case.expected))
    got = _normalize(result.answer)
    ok = got == expected
    return Score(passed=ok, scorer="exact",
                 reason="" if ok else f"expected {expected!r}, got {got!r}")


def score_keyword(case: Case, result: AgentResult) -> Score:
    answer = result.answer.lower()
    missing = [k for k in case.keywords if k.lower() not in answer]
    return Score(passed=not missing, scorer="keyword",
                 reason="" if not missing else f"missing keywords: {missing}")


def score_numeric(case: Case, result: AgentResult) -> Score:
    expected = float(case.expected)  # type: ignore[arg-type]
    found = [float(m.replace(",", "")) for m in _NUM_RE.findall(result.answer)]
    ok = any(abs(n - expected) <= case.tolerance for n in found)
    return Score(passed=ok, scorer="numeric",
                 reason="" if ok else f"expected {expected}±{case.tolerance}, numbers in answer: {found}")


def score_unanswerable(case: Case, result: AgentResult) -> Score:
    answer = result.answer.lower()
    declined = any(m in answer for m in DECLINE_MARKERS)
    return Score(passed=declined, scorer="unanswerable",
                 reason="" if declined else "agent answered instead of declining — possible fabrication")


def score_case(case: Case, result: AgentResult) -> Score:
    """Route a case to its scorer."""
    if result.error:
        return Score(passed=False, scorer="none", reason=f"agent error: {result.error}")

    mode = case.grading_mode
    if mode == GradingMode.EXACT:
        return score_exact(case, result)
    if mode == GradingMode.KEYWORD:
        return score_keyword(case, result)
    if mode == GradingMode.NUMERIC:
        return score_numeric(case, result)
    if mode == GradingMode.UNANSWERABLE:
        return score_unanswerable(case, result)
    if mode == GradingMode.EXECUTION:
        # TODO(S2): run gold SQL and agent SQL against the synthetic DB,
        # compare RESULT SETS (not query strings).
        return Score(passed=False, skipped=True, scorer="execution",
                     reason="execution grading lands in S2")
    if mode == GradingMode.JUDGE:
        # TODO(S3): rubric-based LLM-as-judge. Calibrate against ~30
        # hand-labeled cases and report judge-human agreement.
        return Score(passed=False, skipped=True, scorer="judge",
                     reason="judge grading lands in S3")
    return Score(passed=False, scorer="none", reason=f"unknown grading mode {mode}")
