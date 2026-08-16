"""Execution grading: compare RESULT SETS, not query strings (S2).

Why this module exists at all: two differently-written queries can be equally
correct, and two near-identical queries can differ on exactly the row that
matters. String-matching SQL therefore rewards luck. The only honest question
is "does it return the same data as the gold query?"

We grade the agent's **trajectory**, not its prose answer — the SQL it actually
ran, pulled from its tool calls. That is the difference between evaluating an
agent and benchmarking a text-to-SQL completion.

Three safety rules, because the SQL under test is model output:
  1. read-only connection  — SQLite itself refuses writes, so no blocklist to
     outsmart. A regex banning "DROP" is security theatre; `mode=ro` is not.
  2. one statement per call — `execute()` rejects "SELECT 1; DROP TABLE x".
  3. row cap + wall-clock timeout — a runaway query must not hang the suite or
     pull 92k rows into memory on every repeat.
"""

from __future__ import annotations

import math
import re
import sqlite3
import time
from pathlib import Path

#: Tool the agent is expected to call. The adapter and the scorer agree on this.
SQL_TOOL = "run_sql"

#: Argument keys accepted for the SQL string, in priority order.
SQL_ARG_KEYS = ("sql", "query", "statement")

MAX_ROWS = 1000
TIMEOUT_S = 5.0

_ORDER_BY_RE = re.compile(r"\border\s+by\b", re.IGNORECASE)


class SQLError(Exception):
    """The query did not run. Distinct from 'ran and returned the wrong rows'."""


def _connect(db_path: str | Path) -> sqlite3.Connection:
    # mode=ro is the actual guard against a destructive agent query: SQLite
    # rejects any write at the engine level, whatever the SQL says.
    return sqlite3.connect(f"file:{Path(db_path).resolve()}?mode=ro", uri=True)


def run_query(db_path: str | Path, sql: str, *,
              max_rows: int = MAX_ROWS,
              timeout_s: float = TIMEOUT_S) -> tuple[list[tuple], bool]:
    """Run one read-only statement. Returns (rows, truncated).

    Raises SQLError for anything SQLite refuses — bad syntax, unknown column,
    attempted write, multiple statements, or the timeout firing.
    """
    conn = _connect(db_path)
    deadline = time.monotonic() + timeout_s
    # Fires every N VM instructions; a non-zero return aborts the query. This
    # is the only way to bound a long SELECT in stdlib sqlite3 — `timeout=` is
    # lock-acquisition only and would not help here.
    conn.set_progress_handler(lambda: int(time.monotonic() > deadline), 10_000)
    try:
        rows = conn.execute(sql).fetchmany(max_rows + 1)
    except sqlite3.Error as e:
        raise SQLError(f"{type(e).__name__}: {e}") from e
    finally:
        conn.close()
    return [tuple(r) for r in rows[:max_rows]], len(rows) > max_rows


def order_matters(gold_sql: str) -> bool:
    """Row order is only significant when the gold query asked for it.

    Deliberately conservative: an ORDER BY inside a subquery also trips this,
    making grading stricter rather than more lenient. Being wrongly strict
    shows up as a visible failure; being wrongly lenient hides a real bug.
    """
    return bool(_ORDER_BY_RE.search(gold_sql))


def _decimals(value) -> int | None:
    """How many decimal places the gold value *states*. None if it isn't a
    finite decimal literal (integers, NaN, exponent notation)."""
    try:
        text = repr(float(value))
    except (TypeError, ValueError):
        return None
    if "e" in text or "n" in text or "." not in text:  # 1e-05, nan, inf
        return None
    return len(text.split(".")[1])


def _cells_equal(gold, got) -> bool:
    """Compare one cell. **Directional — `gold` first.**

    Gold declares its own precision: `ROUND(AVG(age), 1)` returning 44.4 is a
    one-decimal claim, so an agent that computes 44.4357 and doesn't round has
    the same answer, not a different one. We therefore compare at gold's stated
    precision rather than loosening tolerance globally — a blanket epsilon
    would also swallow genuinely wrong numbers, which is the failure class the
    whole harness exists to catch. 44.4357 passes against 44.4; 41.2 does not.

    Argument order matters: _cells_equal(got, gold) is a different question.
    """
    if isinstance(gold, float) or isinstance(got, float):
        try:
            g, r = float(gold), float(got)
        except (TypeError, ValueError):
            return False
        if math.isclose(g, r, rel_tol=1e-6, abs_tol=1e-9):
            return True
        places = _decimals(gold)
        return places is not None and round(r, places) == g
    return gold == got


def _rows_equal(gold_row: tuple, got_row: tuple) -> bool:
    """Strict, same-shape equality. Kept for callers that want exactness;
    `compare` uses containment instead — see _row_contains."""
    return (len(gold_row) == len(got_row)
            and all(_cells_equal(g, r) for g, r in zip(gold_row, got_row)))


def _row_contains(gold_row: tuple, got_row: tuple) -> bool:
    """Every gold value appears somewhere in the agent's row.

    Why containment and not equality: gold's *shape* is an artifact of how the
    reference query happened to be written, not part of the question. Asked
    "which month had the most fatal crashes?", an agent that returns
    ('October', 3489) has answered better than one returning ('October',) —
    it showed its work. Failing it teaches the agent to withhold context.

    Greedy match, mirroring _multiset_rows: matching by value rather than by
    position means an agent may also order its columns differently.
    """
    if len(got_row) < len(gold_row):
        return False
    remaining = list(got_row)
    for g in gold_row:
        for i, cell in enumerate(remaining):
            if _cells_equal(g, cell):
                del remaining[i]
                break
        else:
            return False
    return True


def _multiset_rows(gold: list[tuple], got: list[tuple]) -> bool:
    """Order-insensitive row matching, using containment per row.

    Greedy rather than sort-then-zip: sorting would key on exact values, so two
    rows equal *within tolerance* could sort into different positions and
    compare unequal. O(n^2), bounded by MAX_ROWS.
    """
    remaining = list(got)
    for g in gold:
        for i, r in enumerate(remaining):
            if _row_contains(g, r):
                del remaining[i]
                break
        else:
            return False
    return not remaining


def compare(gold: list[tuple], got: list[tuple], *, ordered: bool) -> tuple[bool, str]:
    """Compare two result sets. Returns (equal, diagnostic-if-not).

    The rule is **containment, not equality**: every gold value must appear in
    the agent's corresponding row, but extra columns are not an error. Column
    names are ignored too — an agent may alias differently and still be right.

    Row *count* is still exact, and that is what keeps "wrong grain" a failure:
    one row where five were asked for, or per-crash rows where the question was
    per-person, still fails loudly.

    Known limitation: containment can pass a row that crams several candidate
    values together — gold ('Texas',) is satisfied by ('California', 3877,
    'Texas'). Exact row counts bound the damage, and the alternative measured
    75% of correct answers as failures, so this is the better trade — but it is
    a real hole, not an oversight.
    """
    if len(gold) != len(got):
        return False, f"row count: expected {len(gold)}, got {len(got)}"

    if ordered:
        for i, (g, r) in enumerate(zip(gold, got)):
            if not _row_contains(g, r):
                return False, (f"row {i} (gold has ORDER BY): expected values "
                               f"{g} not found in {r}")
        return True, ""

    if not _multiset_rows(gold, got):
        missing = [g for g in gold if not any(_row_contains(g, r) for r in got)]
        return False, (f"same row count, different rows; no agent row contains "
                       f"{missing[0] if missing else '?'}")
    return True, ""


def last_sql(trajectory, tool: str = SQL_TOOL) -> tuple[str | None, str | None]:
    """The agent's final SQL call. Returns (sql, error-if-that-call-failed).

    The *last* call, not the last successful one. An agent may inspect the
    schema before asking its real question, so the final query is the one it
    answered from. Recovery is still unpunished — a successful retry simply
    *is* the last call.

    Grading the last *successful* call was the original rule and it was wrong:
    a real run explored, then tried a verification query that errored, and the
    scorer silently fell back to the exploratory query and failed the case with
    "column count: expected 1, got 3" — describing a query the agent never
    offered as its answer. Reporting the broken query is the truthful verdict.
    """
    for call in reversed(trajectory):
        if call.name != tool:
            continue
        for key in SQL_ARG_KEYS:
            value = call.arguments.get(key)
            if isinstance(value, str) and value.strip():
                return value, call.error
        return None, call.error
    return None, None
