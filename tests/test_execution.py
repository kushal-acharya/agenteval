"""EXECUTION grading tests.

Uses a tiny fixture DB, not fars.db — CI must not download 50 MB, and the
grading logic has nothing to do with dataset size.
"""

from __future__ import annotations

import sqlite3

import pytest

from agenteval.adapter import AgentResult, ToolCall
from agenteval.cases import Case
from agenteval.scorers import score_case


@pytest.fixture
def db(tmp_path):
    path = tmp_path / "t.db"
    con = sqlite3.connect(path)
    con.executescript("""
        CREATE TABLE accidents (st_case INTEGER, statename TEXT, fatals INTEGER, hour INTEGER);
        INSERT INTO accidents VALUES
            (1, 'Texas', 2, 3), (2, 'Texas', 1, 14), (3, 'Ohio', 1, 22),
            (4, 'Ohio',  3, 7), (5, 'Maine', 1, 9);
    """)
    con.commit()
    con.close()
    return str(path)


def _case(db, gold, **kw):
    return Case.model_validate(
        {"id": "x", "question": "q", "grading_mode": "execution",
         "expected": gold, "db": db, **kw})


def _ran(sql, *, tool="run_sql", error=None, answer="Texas."):
    """An AgentResult whose trajectory contains one run_sql call."""
    return AgentResult(answer=answer,
                       trajectory=[ToolCall(name=tool, arguments={"sql": sql}, error=error)])


# --- the whole point of execution grading -----------------------------------

def test_equivalent_but_differently_written_query_passes(db):
    """Different alias, ordinal GROUP BY, different whitespace — same data."""
    gold = "SELECT statename, SUM(fatals) FROM accidents GROUP BY statename"
    agent = "select   statename,\n  sum(fatals) as total_deaths\nfrom accidents\ngroup by 1"
    assert score_case(_case(db, gold), _ran(agent)).passed


def test_wrong_result_fails_with_diagnostic(db):
    gold = "SELECT statename, SUM(fatals) FROM accidents GROUP BY statename"
    agent = "SELECT statename, COUNT(*) FROM accidents GROUP BY statename"  # counts crashes, not deaths
    score = score_case(_case(db, gold), _ran(agent))
    assert not score.passed and not score.skipped
    assert "row" in score.reason.lower()


def test_row_order_ignored_unless_gold_says_otherwise(db):
    gold = "SELECT statename FROM accidents GROUP BY statename"
    agent = "SELECT statename FROM accidents GROUP BY statename ORDER BY statename DESC"
    assert score_case(_case(db, gold), _ran(agent)).passed


def test_row_order_enforced_when_gold_has_order_by(db):
    gold = "SELECT statename, SUM(fatals) s FROM accidents GROUP BY 1 ORDER BY s DESC"
    agent = "SELECT statename, SUM(fatals) s FROM accidents GROUP BY 1 ORDER BY s ASC"
    score = score_case(_case(db, gold), _ran(agent))
    assert not score.passed and "ORDER BY" in score.reason


def test_float_tolerance(db):
    gold = "SELECT AVG(fatals) FROM accidents"
    agent = "SELECT SUM(fatals) * 1.0 / COUNT(*) FROM accidents"  # same value, different float path
    assert score_case(_case(db, gold), _ran(agent)).passed


# --- safety: the SQL under test is model output -----------------------------

def test_destructive_query_is_refused_by_the_engine(db):
    """mode=ro means we don't rely on a blocklist that a model could evade."""
    gold = "SELECT COUNT(*) FROM accidents"
    score = score_case(_case(db, gold), _ran("DELETE FROM accidents"))
    assert not score.passed and "readonly" in score.reason.lower()
    # and the table really is intact
    con = sqlite3.connect(db)
    assert con.execute("SELECT COUNT(*) FROM accidents").fetchone()[0] == 5
    con.close()


def test_stacked_statements_rejected(db):
    gold = "SELECT COUNT(*) FROM accidents"
    score = score_case(_case(db, gold), _ran("SELECT 1; DROP TABLE accidents"))
    assert not score.passed and not score.skipped


def test_malformed_sql_fails_with_the_engine_error(db):
    gold = "SELECT COUNT(*) FROM accidents"
    score = score_case(_case(db, gold), _ran("SELECT nope FROM accidents"))
    assert not score.passed and "agent SQL failed" in score.reason


# --- trajectory selection ---------------------------------------------------

def test_grades_the_last_query_after_schema_exploration(db):
    """Agents inspect the schema first; the final query is the answer."""
    gold = "SELECT COUNT(*) FROM accidents"
    result = AgentResult(answer="5", trajectory=[
        ToolCall(name="run_sql", arguments={"sql": "SELECT name FROM sqlite_master"}, output="..."),
        ToolCall(name="run_sql", arguments={"sql": "SELECT COUNT(*) FROM accidents"}, output="5"),
    ])
    assert score_case(_case(db, gold), result).passed


def test_recovery_after_a_failed_query_is_not_punished(db):
    gold = "SELECT COUNT(*) FROM accidents"
    result = AgentResult(answer="5", trajectory=[
        ToolCall(name="run_sql", arguments={"sql": "SELECT * FROM crashes"}, error="no such table"),
        ToolCall(name="run_sql", arguments={"sql": "SELECT COUNT(*) FROM accidents"}, output="5"),
    ])
    assert score_case(_case(db, gold), result).passed


def test_a_failed_final_query_reports_that_error_not_an_earlier_query(db):
    """Regression from the first real run.

    The agent explored, got the right rows, then ran a verification query that
    errored. Falling back to the exploratory query failed the case with
    "column count: expected 1, got 3" — a verdict on a query the agent never
    offered as its answer. The final call's own error is the truthful reason.
    """
    gold = "SELECT COUNT(*) FROM accidents"
    result = AgentResult(answer="5", trajectory=[
        ToolCall(name="run_sql",
                 arguments={"sql": "SELECT statename, COUNT(*), SUM(fatals) FROM accidents GROUP BY 1"},
                 output="..."),
        ToolCall(name="run_sql", arguments={"sql": "SELECT COUNT(*) FROM p JOIN q ON p.year=q.year"},
                 error="OperationalError: no such column: p.year"),
    ])
    score = score_case(_case(db, gold), result)
    assert not score.passed
    assert "final query errored" in score.reason and "p.year" in score.reason
    assert "column count" not in score.reason


def test_right_result_set_but_no_answer_fails(db):
    """Regression from a real Haiku run.

    f-08 repeat 2 produced the correct result set and returned no prose at all,
    and scored a clean pass — the agent told the user nothing. Execution grading
    certifies the query, and must not silently certify a missing answer too.
    """
    gold = "SELECT COUNT(*) FROM accidents"
    score = score_case(_case(db, gold), _ran("SELECT COUNT(*) FROM accidents", answer="  "))
    assert not score.passed and "no answer" in score.reason


def test_answering_without_querying_fails(db):
    """The fabrication trap, SQL edition — a fluent answer and no query."""
    score = score_case(_case(db, "SELECT COUNT(*) FROM accidents"),
                       AgentResult(answer="There were 5 crashes."))
    assert not score.passed and not score.skipped
    assert "never called" in score.reason


def test_all_queries_errored_reports_the_last_error(db):
    result = AgentResult(answer="?", trajectory=[
        ToolCall(name="run_sql", arguments={"sql": "SELECT * FROM nope"}, error="no such table: nope"),
    ])
    score = score_case(_case(db, "SELECT COUNT(*) FROM accidents"), result)
    assert not score.passed and "no such table" in score.reason


# --- containment: gold's SHAPE is not part of the question ------------------
#
# The first real sweep failed 15 of 20 attempts on column count alone, while the
# agents' answers were correct. Gold's arity is an artifact of how the reference
# query was written; the question ("which month had the most fatal crashes?")
# never asked for exactly one column. These pin that down.

def test_extra_context_columns_pass(db):
    """f-08/f-01: the agent showed its work. That is better, not wrong."""
    gold = "SELECT statename FROM accidents GROUP BY statename ORDER BY COUNT(*) DESC LIMIT 1"
    agent = ("SELECT statename, COUNT(*) AS n FROM accidents "
             "GROUP BY statename ORDER BY n DESC LIMIT 1")
    assert score_case(_case(db, gold), _ran(agent)).passed


def test_extra_columns_do_not_rescue_a_wrong_value(db):
    """Containment must not become 'any row mentioning anything'."""
    gold = "SELECT statename FROM accidents GROUP BY statename ORDER BY COUNT(*) DESC LIMIT 1"
    agent = "SELECT statename, COUNT(*) FROM accidents GROUP BY statename ORDER BY COUNT(*) ASC LIMIT 1"
    assert not score_case(_case(db, gold), _ran(agent)).passed


def test_fewer_columns_than_gold_still_fails(db):
    """Containment is one-directional: gold ⊆ agent, never the reverse."""
    gold = "SELECT statename, SUM(fatals) FROM accidents GROUP BY statename"
    agent = "SELECT statename FROM accidents GROUP BY statename"
    assert not score_case(_case(db, gold), _ran(agent)).passed


def test_row_count_still_guards_grain(db):
    """The check that survived containment — one row is not five."""
    gold = "SELECT SUM(fatals) FROM accidents"
    agent = "SELECT fatals FROM accidents"
    score = score_case(_case(db, gold), _ran(agent))
    assert not score.passed and "row count" in score.reason


# --- numeric precision: gold declares how precise its claim is --------------

def test_agent_may_skip_the_rounding_gold_applied(db):
    """f-07: ROUND(x,1) -> 1.6 is a one-decimal claim; 1.6 vs 1.6000000001 is
    the same answer, and an unrounded 1.6 is too."""
    gold = "SELECT ROUND(AVG(fatals), 1) FROM accidents"
    agent = "SELECT AVG(fatals) FROM accidents"          # 1.6 exactly here
    assert score_case(_case(db, gold), _ran(agent)).passed


def test_precision_matching_does_not_hide_a_wrong_number(db):
    """The reason this is precision-matching and not a blanket epsilon."""
    gold = "SELECT ROUND(AVG(fatals), 1) FROM accidents"   # 1.6
    agent = "SELECT ROUND(AVG(fatals) + 1, 1) FROM accidents"
    assert not score_case(_case(db, gold), _ran(agent)).passed


def test_units_error_is_not_a_precision_error(db):
    """0.502 vs 50.2 must stay a failure — that is a units bug in the answer."""
    gold = "SELECT ROUND(AVG(fatals) / 10.0, 3) FROM accidents"
    agent = "SELECT ROUND(AVG(fatals) * 10.0, 3) FROM accidents"
    assert not score_case(_case(db, gold), _ran(agent)).passed


# --- harness/dataset problems must not distort the pass rate ----------------

def test_broken_gold_sql_is_skipped_not_failed(db):
    """A dataset bug is not the agent's fault; counting it as FAIL corrupts the rate."""
    score = score_case(_case(db, "SELECT * FROM table_that_does_not_exist"),
                       _ran("SELECT COUNT(*) FROM accidents"))
    assert score.skipped and not score.passed
    assert "GOLD SQL FAILED" in score.reason


def test_missing_db_is_skipped_not_failed():
    case = Case.model_validate({"id": "x", "question": "q", "grading_mode": "execution",
                                "expected": "SELECT 1"})
    score = score_case(case, _ran("SELECT 1"))
    assert score.skipped and not score.passed
