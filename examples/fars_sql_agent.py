"""A real Claude agent with one tool: SQL over the FARS crash database.

This is the first adapter that calls a real model, which is the point — until
now the harness only ever graded a deterministic toy, so `pass^k` could never
differ from `pass@1` and every latency/token figure was zero.

    export ANTHROPIC_API_KEY=...            # or: ant auth login
    python scripts/load_fars.py             # builds datasets/fars.db
    agenteval run --dataset datasets/fars_v1.jsonl \
                  --adapter examples.fars_sql_agent:FarsSQLAgent --repeats 5

Two deliberate choices:

* **A hand-written tool loop**, not the SDK's `tool_runner` helper. The helper
  is the right call in production, but this loop is the thing an interviewer
  asks about, and it is ~30 readable lines.
* **The agent executes SQL through `agenteval.execution.run_query`** — the same
  read-only, row-capped, timeout-bounded path the scorer uses. The agent
  therefore cannot reach data the grader couldn't, and a destructive query is
  refused by SQLite for the agent exactly as it is for the scorer.
"""

from __future__ import annotations

import os
import sqlite3
from pathlib import Path

from agenteval.adapter import AgentAdapter, AgentResult, ToolCall
from agenteval.cases import Case
from agenteval.execution import SQL_TOOL, SQLError, run_query

def load_dotenv(path: str | Path = ".env") -> None:
    """Read KEY=value lines from .env into the environment.

    Six lines of parsing instead of a dependency. `setdefault` matters: a real
    environment variable always wins over the file, so `ANTHROPIC_API_KEY=... \
python -m ...` and CI secrets are never silently overridden by a stale .env.
    """
    p = Path(path)
    if not p.exists():
        return
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


MODEL = os.environ.get("AGENTEVAL_MODEL", "claude-opus-5")
# Thinking is on by default on Opus 5 and max_tokens caps thinking + text
# together, so a tight budget truncates the answer rather than the reasoning.
MAX_TOKENS = 4096
MAX_TURNS = 6          # a stuck agent must cost a bounded amount
MAX_RESULT_CHARS = 4000  # what we feed back per query

TOOLS = [{
    "name": SQL_TOOL,
    "description": (
        "Run one read-only SQL SELECT against the crash database and return the "
        "rows. Call this before answering any question about the data — never "
        "answer from memory. You may call it more than once (e.g. to inspect the "
        "schema first). Only a single statement per call."
    ),
    "input_schema": {
        "type": "object",
        "properties": {"sql": {"type": "string", "description": "One SQL SELECT statement."}},
        "required": ["sql"],
        "additionalProperties": False,
    },
}]

SYSTEM_PREAMBLE = """You are a road-safety data analyst answering questions against a \
SQLite database of crash records.

Rules:
- Always answer from a run_sql query. Never answer from prior knowledge about \
crash statistics, even if you are confident.
- If the database cannot answer the question — the year, state, or field is not \
in it — say so plainly and do not guess. Declining is the correct answer when \
the data is absent.
- Columns ending in NAME are human-readable decodes; prefer them over their \
numeric counterparts.
- Answer in one or two sentences, stating the number you found.

Your FINAL run_sql call is your answer, and it is read as such:
- It must return exactly what was asked and nothing more. If the question asks \
for one value, return one column and one row. If it asks for a ranked list of \
N, return N rows with only the columns named.
- Do not append extra context columns to it. Supporting counts are useful in \
prose, but the final query is the answer itself.
- Explore freely first — check the schema, sanity-check a total, whatever helps \
— but the last query you run must be the answer query. Do not run a \
verification query after it.

Schema:
"""


#: Enumerate distinct values for a decode column only if it has at most this many.
MAX_ENUM_VALUES = 12


def describe_schema(db_path: str | Path) -> str:
    """Introspect the DB rather than hardcoding a schema string.

    Hardcoding drifts the moment the loader's column allowlist changes, and it
    would not transfer to the Chicago database later.

    For low-cardinality `*NAME` decode columns we also list the actual values.
    This is the single highest-value thing to put in a text-to-SQL prompt: with
    it the model writes `WHERE day_weekname = 'Saturday'`; without it it guesses
    between 'Saturday', 'SAT', and 'Sat' and the query silently returns 0 rows.
    """
    con = sqlite3.connect(f"file:{Path(db_path).resolve()}?mode=ro", uri=True)
    try:
        tables = [r[0] for r in con.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")]
        parts = []
        for table in tables:
            n = con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            cols = [r[1] for r in con.execute(f"PRAGMA table_info({table})")]
            lines = [f"{table} ({n:,} rows)", "  " + ", ".join(cols)]
            for col in cols:
                if not col.endswith("name"):
                    continue
                rows = con.execute(
                    f"SELECT DISTINCT {col} FROM {table} LIMIT {MAX_ENUM_VALUES + 1}"
                ).fetchall()
                if len(rows) <= MAX_ENUM_VALUES:
                    values = sorted(str(r[0]) for r in rows if r[0] is not None)
                    lines.append(f"    {col} in ({', '.join(repr(v) for v in values)})")
            parts.append("\n".join(lines))
        return "\n\n".join(parts)
    finally:
        con.close()


def _format_rows(rows: list[tuple], truncated: bool) -> str:
    if not rows:
        return "(0 rows)"
    body = "\n".join(str(r) for r in rows)
    if len(body) > MAX_RESULT_CHARS:
        body = body[:MAX_RESULT_CHARS] + "\n... (output truncated)"
    note = "\n(row cap hit — result set is incomplete)" if truncated else ""
    return f"{len(rows)} row(s):\n{body}{note}"


class FarsSQLAgent(AgentAdapter):
    name = "fars-sql-agent"

    def __init__(self, model: str = MODEL) -> None:
        import anthropic  # imported here so the core harness never needs it

        load_dotenv()  # in __init__, not at import — importing stays side-effect free
        self.model = os.environ.get("AGENTEVAL_MODEL", model)
        self.client = anthropic.Anthropic()
        self._schema_cache: dict[str, str] = {}

    def _system(self, db: str) -> list[dict]:
        if db not in self._schema_cache:
            self._schema_cache[db] = describe_schema(db)
        return [{
            "type": "text",
            "text": SYSTEM_PREAMBLE + self._schema_cache[db],
            # Byte-identical across every case and repeat, so it caches: reads
            # cost ~0.1x. Note the threshold — Opus 5 will not cache a prefix
            # under 512 tokens, and it fails *silently* (no error, just
            # cache_read_input_tokens stuck at 0). The bare column list was ~274
            # tokens and never cached; the decode values pushed it to ~781. They
            # earn their place on accuracy, not on making the cache work.
            "cache_control": {"type": "ephemeral"},
        }]

    def run(self, case: Case) -> AgentResult:
        """Answer one case. Never raises — the runner records .error as a failure."""
        result = AgentResult()
        db = case.db
        if not db:
            result.error = "case has no `db`; FarsSQLAgent needs one"
            return result
        if not Path(db).exists():
            result.error = f"{db} not found — run: python scripts/load_fars.py"
            return result

        messages: list[dict] = [{"role": "user", "content": case.question}]
        try:
            for _ in range(MAX_TURNS):
                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=MAX_TOKENS,
                    system=self._system(db),
                    tools=TOOLS,
                    messages=messages,
                )
                usage = response.usage
                # Cached reads are still input the model saw; counting only the
                # uncached remainder would understate the prompt by ~90%.
                result.tokens_in += (usage.input_tokens
                                     + (usage.cache_read_input_tokens or 0)
                                     + (usage.cache_creation_input_tokens or 0))
                result.tokens_out += usage.output_tokens

                if response.stop_reason == "refusal":
                    result.error = "model refused the request"
                    return result

                tool_uses = [b for b in response.content if b.type == "tool_use"]
                if not tool_uses:
                    result.answer = "".join(
                        b.text for b in response.content if b.type == "text").strip()
                    return result

                messages.append({"role": "assistant", "content": response.content})
                tool_results = []
                for block in tool_uses:
                    sql = (block.input or {}).get("sql", "")
                    call = ToolCall(name=block.name, arguments={"sql": sql})
                    try:
                        rows, truncated = run_query(db, sql)
                        call.output = _format_rows(rows, truncated)
                        payload, is_error = call.output, False
                    except SQLError as e:
                        # Hand the error back rather than aborting: recovering
                        # from a bad query is behaviour worth measuring, and the
                        # scorer grades the last *successful* call.
                        call.error = str(e)
                        payload, is_error = f"SQL error: {e}", True
                    result.trajectory.append(call)
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": payload,
                        "is_error": is_error,
                    })
                messages.append({"role": "user", "content": tool_results})

            result.error = f"gave up after {MAX_TURNS} tool turns"
            return result

        except Exception as e:  # adapters must not raise; the run records a failure
            result.error = f"{type(e).__name__}: {e}"
            return result
