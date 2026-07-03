"""Metrics: pass@1, pass^k, flakiness, per-tag rates, latency, cost.

pass@1  — first attempt passed (among graded cases)
pass^k  — ALL k repeats passed. "88% pass@1 but 61% pass^5" is the
          reliability story; this gap is what most evals never measure.
flaky   — passed some repeats but not all.
"""

from __future__ import annotations

import statistics
from collections import defaultdict

from pydantic import BaseModel, Field

from .runner import RepeatResult, RunRecord


class Summary(BaseModel):
    cases_total: int
    cases_graded: int
    cases_skipped: int
    pass_at_1: float
    pass_pow_k: float
    k: int
    flaky_cases: list[str] = Field(default_factory=list)
    by_tag: dict[str, str] = Field(default_factory=dict)  # tag -> "passed/total"
    latency_p50_s: float
    latency_p95_s: float
    tokens_in: int
    tokens_out: int
    tool_errors: int
    failures: list[RepeatResult] = Field(default_factory=list)  # first-repeat failures


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    idx = min(int(round(pct / 100 * (len(values) - 1))), len(values) - 1)
    return values[idx]


def summarize(run: RunRecord) -> Summary:
    by_case: dict[str, list[RepeatResult]] = defaultdict(list)
    for r in run.results:
        by_case[r.case_id].append(r)
    for reps in by_case.values():
        reps.sort(key=lambda r: r.repeat)

    graded = {cid: reps for cid, reps in by_case.items() if not reps[0].skipped}
    skipped = len(by_case) - len(graded)

    pass1 = [reps[0].passed for reps in graded.values()]
    passk = [all(r.passed for r in reps) for reps in graded.values()]
    flaky = sorted(cid for cid, reps in graded.items()
                   if any(r.passed for r in reps) and not all(r.passed for r in reps))

    tag_hit: dict[str, int] = defaultdict(int)
    tag_all: dict[str, int] = defaultdict(int)
    for reps in graded.values():
        first = reps[0]
        for tag in first.tags or ["untagged"]:
            tag_all[tag] += 1
            tag_hit[tag] += first.passed

    latencies = [r.latency_s for r in run.results]
    return Summary(
        cases_total=len(by_case),
        cases_graded=len(graded),
        cases_skipped=skipped,
        pass_at_1=round(statistics.mean(pass1), 4) if pass1 else 0.0,
        pass_pow_k=round(statistics.mean(passk), 4) if passk else 0.0,
        k=run.repeats,
        flaky_cases=flaky,
        by_tag={t: f"{tag_hit[t]}/{tag_all[t]}" for t in sorted(tag_all)},
        latency_p50_s=round(_percentile(latencies, 50), 4),
        latency_p95_s=round(_percentile(latencies, 95), 4),
        tokens_in=sum(r.tokens_in for r in run.results),
        tokens_out=sum(r.tokens_out for r in run.results),
        tool_errors=sum(r.tool_errors for r in run.results),
        failures=[reps[0] for reps in graded.values() if not reps[0].passed],
    )
