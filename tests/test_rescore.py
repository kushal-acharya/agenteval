"""Rescore tests.

The load-bearing property is that rescoring re-derives *judgments* while
leaving *observations* untouched — and that it refuses to invent a verdict it
cannot honestly derive.
"""

from __future__ import annotations

import sqlite3

import pytest

from agenteval.adapter import ToolCall
from agenteval.cases import Case
from agenteval.rescore import rescore_run
from agenteval.runner import RepeatResult, RunRecord


@pytest.fixture
def db(tmp_path):
    path = tmp_path / "t.db"
    con = sqlite3.connect(path)
    con.executescript("""
        CREATE TABLE accidents (st_case INTEGER, statename TEXT, fatals INTEGER);
        INSERT INTO accidents VALUES (1,'Texas',2), (2,'Texas',1), (3,'Ohio',1);
    """)
    con.commit()
    con.close()
    return str(path)


def _case(db, gold):
    return Case.model_validate({"id": "c1", "question": "q",
                                "grading_mode": "execution",
                                "expected": gold, "db": db})


def _run(results, dataset="d.jsonl"):
    return RunRecord(run_id="orig", created_at="2026-01-01T00:00:00",
                     adapter="a", dataset=dataset, repeats=1, results=results)


def _result(**kw):
    base = dict(case_id="c1", repeat=0, answer="Texas.", passed=False,
                skipped=False, scorer="execution", reason="stale verdict",
                latency_s=9.1, tokens_in=1200, tokens_out=90,
                tool_calls=0, tool_errors=0, trajectory=[])
    base.update(kw)
    return RepeatResult(**base)


def _sql(sql):
    return [ToolCall(name="run_sql", arguments={"sql": sql})]


# --- the point: stale judgments are re-derived --------------------------------

def test_stale_fail_becomes_pass_under_the_current_scorer(db):
    """The bug that motivated this: a correct answer saved as FAIL."""
    gold = "SELECT statename FROM accidents GROUP BY statename ORDER BY COUNT(*) DESC LIMIT 1"
    agent = ("SELECT statename, COUNT(*) AS n FROM accidents "
             "GROUP BY statename ORDER BY n DESC LIMIT 1")
    run = _run([_result(passed=False, tool_calls=1, trajectory=_sql(agent))])

    new, flips = rescore_run(run, [_case(db, gold)])

    assert new.results[0].passed
    assert [(f.was, f.now) for f in flips] == [("FAIL", "PASS")]


def test_observations_are_carried_through_untouched(db):
    """Latency and tokens belong to the original call and must not be reinvented."""
    gold = "SELECT COUNT(*) FROM accidents"
    run = _run([_result(tool_calls=1, trajectory=_sql(gold))])

    new, _ = rescore_run(run, [_case(db, gold)])
    before, after = run.results[0], new.results[0]

    for field in ("latency_s", "tokens_in", "tokens_out", "answer",
                  "trajectory", "tool_calls", "case_id", "repeat"):
        assert getattr(after, field) == getattr(before, field), field


def test_original_run_is_not_mutated(db):
    """The old verdicts are the 'before' half of the story."""
    gold = "SELECT COUNT(*) FROM accidents"
    run = _run([_result(passed=False, reason="stale verdict",
                        tool_calls=1, trajectory=_sql(gold))])

    rescore_run(run, [_case(db, gold)])

    assert run.results[0].passed is False
    assert run.results[0].reason == "stale verdict"


# --- refusing to invent a verdict ---------------------------------------------

def test_lost_trajectory_is_skipped_not_failed(db):
    """`tool_calls: 3` with an empty trajectory means we failed to record it —
    the agent did query. Scoring that as FAIL would corrupt the pass rate."""
    run = _run([_result(passed=True, tool_calls=3, trajectory=[])])

    new, flips = rescore_run(run, [_case(db, "SELECT COUNT(*) FROM accidents")])

    assert new.results[0].skipped and not new.results[0].passed
    assert "no trajectory recorded" in new.results[0].reason
    assert flips[0].now == "SKIP"


def test_genuinely_absent_trajectory_still_fails(db):
    """Zero tool calls is not data loss — it is the fabrication trap firing."""
    run = _run([_result(passed=False, tool_calls=0, trajectory=[])])

    new, _ = rescore_run(run, [_case(db, "SELECT COUNT(*) FROM accidents")])

    assert not new.results[0].passed and not new.results[0].skipped
    assert "never called" in new.results[0].reason


def test_case_removed_from_dataset_is_skipped(db):
    run = _run([_result(case_id="gone", tool_calls=1, trajectory=_sql("SELECT 1"))])

    new, _ = rescore_run(run, [_case(db, "SELECT COUNT(*) FROM accidents")])

    assert new.results[0].skipped
    assert "not in the dataset" in new.results[0].reason


# --- provenance ----------------------------------------------------------------

def test_provenance_is_recorded(db, tmp_path):
    """A reader must be able to tell a rescored run from a real one, and see
    whether the golden set moved underneath it."""
    ds = tmp_path / "d.jsonl"
    ds.write_text('{"id":"c1","question":"q","grading_mode":"exact","expected":"x"}\n')
    gold = "SELECT COUNT(*) FROM accidents"
    run = _run([_result(tool_calls=1, trajectory=_sql(gold))], dataset=str(ds))

    new, _ = rescore_run(run, [_case(db, gold)], dataset_path=ds)

    assert new.rescored_from == "orig"
    assert new.run_id != "orig"
    assert new.rescored_at
    assert new.dataset_sha and new.dataset_sha != "missing"


def test_no_change_reports_no_flips(db):
    gold = "SELECT COUNT(*) FROM accidents"
    run = _run([_result(passed=True, reason="", tool_calls=1, trajectory=_sql(gold))])

    _, flips = rescore_run(run, [_case(db, gold)])

    assert flips == []
