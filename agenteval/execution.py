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


def _cells_equal(a, b) -> bool:
    if isinstance(a, float) or isinstance(b, float):
        try:
            return math.isclose(float(a), float(b), rel_tol=1e-6, abs_tol=1e-9)
        except (TypeError, ValueError):
            return False
    return a == b


def _rows_equal(a: tuple, b: tuple) -> bool:
    return len(a) == len(b) and all(_cells_equal(x, y) for x, y in zip(a, b))


def _multiset_equal(gold: list[tuple], got: list[tuple]) -> bool:
    """Order-insensitive comparison that still honours float tolerance.

    Greedy match rather than sort-then-zip: sorting would key on exact values,
    so two rows equal *within tolerance* could sort into different positions
    and compare unequal. O(n^2), bounded by MAX_ROWS.
    """
    remaining = list(got)
    for g in gold:
        for i, r in enumerate(remaining):
            if _rows_equal(g, r):
                del remaining[i]
                break
        else:
            return False
    return not remaining


def compare(gold: list[tuple], got: list[tuple], *, ordered: bool) -> tuple[bool, str]:
    """Compare two result sets. Returns (equal, diagnostic-if-not).

    Column *names* are ignored — an agent may alias differently and still be
    right. Column *count* and values are not.
    """
    if len(gold) != len(got):
        return False, f"row count: expected {len(gold)}, got {len(got)}"
    if gold and got and len(gold[0]) != len(got[0]):
        return False, f"column count: expected {len(gold[0])}, got {len(got[0])}"

    if ordered:
        for i, (g, r) in enumerate(zip(gold, got)):
            if not _rows_equal(g, r):
                return False, f"row {i} differs (gold has ORDER BY): expected {g}, got {r}"
        return True, ""

    if not _multiset_equal(gold, got):
        missing = [g for g in gold if not any(_rows_equal(g, r) for r in got)]
        return False, f"same row count, different rows; first missing: {missing[0] if missing else '?'}"
    return True, ""


def last_sql(trajectory, tool: str = SQL_TOOL) -> str | None:
    """The last *successful* SQL the agent ran, or None.

    Last, because an agent may inspect the schema before asking its real
    question — the final query is the one it answered from. Successful,
    because retrying after a syntax error is good agent behaviour, and
    grading the discarded attempt would punish recovery.
    """
    for call in reversed(trajectory):
        if call.name != tool or call.error:
            continue
        for key in SQL_ARG_KEYS:
            value = call.arguments.get(key)
            if isinstance(value, str) and value.strip():
                return value
    return None
