# AgentEval

An eval + tracing harness for LLM agents: run any agent against a versioned
golden set, score it with layered graders, and report **pass@1, pass^k
reliability, per-tag rates, latency, cost, and tool errors** — with a
regression diff between versions.

Why it exists: models can one-shot agents now. Deciding what *correct* means
in a domain, and proving an agent is *reliable* — not just right once — is
the part that doesn't commoditize. This harness measures exactly that.

## 2-minute demo (no API key needed)

```bash
pip install -e ".[dev]"
pytest -q
agenteval run --dataset examples/toy_v1.jsonl \
              --adapter examples.toy_agent:ToyAgent --repeats 3
```

You'll see a report like:

```
cases: 12   graded: 12   skipped: 0
pass@1: 83.3%   pass^3: 83.3%   flaky: 0
by tag:
  glossary      5/6
  math          4/4
  ...
failures (first repeat):
  toy-06  [keyword] missing keywords: ['delay']
  toy-11  [unanswerable] agent answered instead of declining — possible fabrication
```

Both failures are designed: `toy-06` shows an incomplete answer being
caught, and `toy-11` is a fabrication trap — the agent pattern-matches
"AADT" and recites a definition instead of admitting it has no 2025 data
for segment S-104. Exactly the failure mode that matters in production.

## Plugging in your own agent

Implement one interface (`agenteval/adapter.py`):

```python
from agenteval.adapter import AgentAdapter, AgentResult
from agenteval.cases import Case

class MyAgent(AgentAdapter):
    name = "my-agent"
    def run(self, case: Case) -> AgentResult:
        answer, tool_calls = my_loop(case.question)
        return AgentResult(answer=answer, trajectory=tool_calls)
```

Then: `agenteval run --dataset datasets/transit_v1.jsonl --adapter mypkg.agent:MyAgent --repeats 5`

A worked example lives in [`examples/fars_sql_agent.py`](examples/fars_sql_agent.py) — a
real Claude agent with a single `run_sql` tool over the FARS database:

```bash
pip install -e ".[dev,claude]"
export ANTHROPIC_API_KEY=...
python scripts/load_fars.py
agenteval run --dataset datasets/fars_v1.jsonl \
              --adapter examples.fars_sql_agent:FarsSQLAgent --repeats 5
```

The harness core depends on nothing but `pydantic`; the vendor SDK is an optional
extra used only by that example. Model independence is the point of the adapter
interface, so `pip install agenteval` must not drag in an SDK.

## Re-grading a saved run

```bash
agenteval rescore reports/run_<id>.json
```

A run file holds two different kinds of thing: **observations** (what the agent did —
answer, trajectory, tokens, latency) and **judgments** (passed / reason). Observations are
empirical and permanent. Judgments are derived, and go stale the moment you change a
scorer. `rescore` keeps the observations, re-derives the judgments, and writes a *new*
run file — the original is never touched, because the old verdicts are the "before" half
of any honest write-up.

It costs nothing and calls no model, but the real reason it exists is that **it isolates one
variable**: to know whether a grading change helped, the trajectories have to be held fixed.
A fresh sweep changes the grader *and* resamples the model, and then the movement is
unattributable. It also turns every saved run into a regression fixture for the scorer
itself — change `execution.py`, rescore the corpus, and see whether any verdict flipped
that you didn't intend.

It refuses to invent verdicts it can't derive. A run recorded before trajectories were
persisted has `tool_calls: 3` and an empty trajectory: the agent *did* query, the harness
simply failed to write it down. That is skipped loudly, never scored as a failure — while
zero tool calls with no trajectory is not data loss but the fabrication trap firing, and
still fails. It cannot re-measure latency or tokens (those belong to the original call and
are carried through unchanged), cannot tell you what a model does *today*, and — since
execution grading re-executes SQL — is only reproducible against a fixed database.

## Seeing a run

The text report gives you the numbers. The dashboard shows you *why* you got them:

```bash
python dashboard/serve.py          # newest run in reports/
python dashboard/serve.py reports/run_<id>.json
```

It opens a reliability report: pass@1 beside pass^k with the gap between them, then one
row per question and one square per repeat — so non-determinism is something you can
see rather than infer. Clicking a row opens the attempt: the grader's verdict and its
exact reason, the agent's full trajectory (including failed calls, and which one was
graded), and the gold and agent **result sets side by side**.

`serve.py` is stdlib-only. It exists because a `RunRecord` deliberately stores what
happened, not what it meant: it has `case_id` but not the question, and the agent's SQL
but not what that SQL returned. The server joins the run against its dataset and replays
both queries read-only to reconstruct the comparison.

## Datasets

- `examples/toy_v1.jsonl` — 12 self-contained cases for the demo.
- `datasets/fars_v1.jsonl` — 9 execution cases + 1 fabrication trap over **real
  NHTSA FARS 2023 data** (37,769 fatal crashes, 92,768 people). Build the DB
  with `python scripts/load_fars.py`; it is generated, never committed.
- `datasets/transit_v1.jsonl` — 25 transportation seed cases: definitional
  (graded today), SQL cases (targeting a schema that predates FARS — being
  retired), countermeasure-recommendation cases with rubrics (activate in S3
  with a calibrated judge), and fabrication traps (`unanswerable`).

**Execution grading** compares result sets, not query strings, and grades the
SQL the agent *actually ran* (pulled from its `run_sql` tool calls) rather than
its prose answer — the difference between evaluating an agent and benchmarking
a text-to-SQL completion. Row order matters only when the gold query has an
`ORDER BY`; column aliases are ignored; floats compare within tolerance. The
connection is opened `mode=ro`, so a destructive query is refused by SQLite
itself rather than by a blocklist a model could talk its way around.

Grading modes: `exact`, `keyword`, `numeric`, `execution` (compare SQL
**result sets**, not query strings), `judge` (rubric + calibrated
LLM-as-judge), `unanswerable` (the agent must decline, not fabricate).

## Metrics that matter

- **pass@1** — first attempt correct.
- **pass^k** — correct on *all* k repeats. The pass@1 vs pass^k gap is the
  reliability story most evals never tell.
- **flaky cases** — pass some repeats, fail others; the first thing to fix.
- latency p50/p95, tokens/cost, tool-error rate.

## Roadmap (build order)

- [x] **S1** — dataset schema, adapter interface, serial runner, deterministic scorers, CLI
- [x] **S2** — execution-based SQL grading: runs the agent's **trajectory** SQL
      against a read-only DB and compares result sets
- [ ] **S3** — LLM-as-judge + calibration vs ~30 hand-labeled cases (report agreement)
- [x] **S3.5** — `agenteval rescore`: re-derive verdicts from saved trajectories, no model
      calls. Pulled forward from S4 because a scorer fix left every saved run stale
- [ ] **S4** — `agenteval diff`: newly-failing / newly-fixed / flaky / cost delta
- [ ] **S5** — OTel tracing → self-hosted Phoenix; failure→trace links; CI regression gate

## How this harness lies to you (and the guards)

1. **An uncalibrated judge is a vibe.** Judge scores mean nothing until you
   report judge–human agreement on labeled cases (S3).
2. **String-matching SQL rewards luck.** Two different queries can both be
   right; grade the *result set* (S2).
3. **pass@1 hides flakiness.** A demo that works once isn't a system that
   works; that's why repeats and pass^k are first-class.
4. **A grader can be wrong more confidently than an agent.** The first real
   run scored 0/8 while every answer was correct — the agent returned extra
   context columns and ran sanity checks after finding the answer, and no
   output contract had been stated. A red suite is a claim about the harness
   until you have read the trajectories.
5. **An underspecified question is a fake failure.** "What share…" that never
   says units marks `50.2` wrong against a gold of `0.502`. Execution grading
   compares exact values, so its questions must pin down units and rounding —
   that is dataset authoring, not model error.
