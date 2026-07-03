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

## Datasets

- `examples/toy_v1.jsonl` — 12 self-contained cases for the demo.
- `datasets/transit_v1.jsonl` — 25 transportation seed cases: definitional
  (graded today), SQL cases (activate in S2 with execution-based grading),
  countermeasure-recommendation cases with rubrics (activate in S3 with a
  calibrated judge), and fabrication traps (`unanswerable`).

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
- [ ] **S2** — execution-based SQL grading (result-set comparison)
- [ ] **S3** — LLM-as-judge + calibration vs ~30 hand-labeled cases (report agreement)
- [ ] **S4** — `agenteval diff`: newly-failing / newly-fixed / flaky / cost delta
- [ ] **S5** — OTel tracing → self-hosted Phoenix; failure→trace links; CI regression gate

## How this harness lies to you (and the guards)

1. **An uncalibrated judge is a vibe.** Judge scores mean nothing until you
   report judge–human agreement on labeled cases (S3).
2. **String-matching SQL rewards luck.** Two different queries can both be
   right; grade the *result set* (S2).
3. **pass@1 hides flakiness.** A demo that works once isn't a system that
   works; that's why repeats and pass^k are first-class.
