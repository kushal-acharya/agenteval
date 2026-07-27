"""Layered scorers — cheapest first.

Deterministic scorers live here (S1) plus EXECUTION (S2). JUDGE (S3) is
routed but returns skipped=True until implemented, so it never silently
inflates or deflates the pass rate.
"""

from __future__ import annotations

import re

from pydantic import BaseModel

from .adapter import AgentResult
from .cases import Case, GradingMode
from .execution import SQL_TOOL, SQLError, compare, last_sql, order_matters, run_query

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


def score_execution(case: Case, result: AgentResult, *, tool: str = SQL_TOOL) -> Score:
    """Grade the SQL the agent actually ran, by result set.

    Failure taxonomy, kept distinct on purpose — "wrong rows" and "never
    queried at all" are different bugs and a single FAIL would hide that:
      - no db configured          -> skipped (harness gap, not the agent)
      - gold SQL broken           -> skipped (dataset bug, not the agent)
      - agent never queried       -> FAIL (answered from thin air)
      - every query errored       -> FAIL, with the last SQLite error
      - query ran, rows differ    -> FAIL, with the first divergence
    """
    if not case.db:
        return Score(passed=False, skipped=True, scorer="execution",
                     reason="case has no `db` — nothing to execute against")

    calls = [c for c in result.trajectory if c.name == tool]
    sql, call_error = last_sql(result.trajectory, tool)
    if not calls:
        return Score(passed=False, scorer="execution",
                     reason=f"agent never called {tool!r} — answered without querying")
    if call_error:
        # The agent's final database action failed. Report *that*, not a verdict
        # on some earlier exploratory query it never offered as its answer.
        return Score(passed=False, scorer="execution",
                     reason=f"agent's final query errored: {call_error}")
    if sql is None:
        return Score(passed=False, scorer="execution",
                     reason=f"final {tool!r} call carried no SQL: {calls[-1].arguments}")

    # Gold runs first. A broken gold query is a dataset defect, and letting it
    # count as an agent failure would quietly corrupt the pass rate — skip it
    # loudly instead. (The runner never crashes, so we must not raise here.)
    try:
        gold_rows, gold_truncated = run_query(case.db, str(case.expected))
    except SQLError as e:
        return Score(passed=False, skipped=True, scorer="execution",
                     reason=f"GOLD SQL FAILED — dataset bug, not the agent: {e}")
    if gold_truncated:
        return Score(passed=False, skipped=True, scorer="execution",
                     reason="gold query exceeds the row cap — tighten the gold SQL")

    try:
        got_rows, got_truncated = run_query(case.db, sql)
    except SQLError as e:
        return Score(passed=False, scorer="execution", reason=f"agent SQL failed: {e}")
    if got_truncated:
        return Score(passed=False, scorer="execution",
                     reason="agent query exceeded the row cap (missing aggregate or LIMIT?)")

    ok, why = compare(gold_rows, got_rows, ordered=order_matters(str(case.expected)))
    if ok and not result.answer.strip():
        # Found in a real Haiku run: f-08 repeat 2 returned the correct result
        # set and *no prose at all*, and scored a clean pass. Execution grading
        # validates the query; it must not also certify an answer that was never
        # given. Right query, nothing said, is not a pass.
        return Score(passed=False, scorer="execution",
                     reason="correct result set but the agent returned no answer")
    return Score(passed=ok, scorer="execution", reason="" if ok else why)


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
        return score_execution(case, result)
    if mode == GradingMode.JUDGE:
        # TODO(S3): rubric-based LLM-as-judge. Calibrate against ~30
        # hand-labeled cases and report judge-human agreement.
        return Score(passed=False, skipped=True, scorer="judge",
                     reason="judge grading lands in S3")
    return Score(passed=False, scorer="none", reason=f"unknown grading mode {mode}")
