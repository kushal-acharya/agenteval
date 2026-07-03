"""Plain-text report. TODO(S5): HTML report with failure->trace links."""

from __future__ import annotations

from .metrics import Summary, summarize
from .runner import RunRecord


def render(run: RunRecord) -> str:
    s: Summary = summarize(run)
    lines = [
        f"AgentEval run {run.run_id}",
        f"  adapter={run.adapter}  dataset={run.dataset}  repeats={run.repeats}",
        "",
        f"  cases: {s.cases_total}   graded: {s.cases_graded}   "
        f"skipped: {s.cases_skipped} (grading modes pending S2/S3)",
        f"  pass@1: {s.pass_at_1:.1%}   pass^{s.k}: {s.pass_pow_k:.1%}   "
        f"flaky: {len(s.flaky_cases)}{' ' + str(s.flaky_cases) if s.flaky_cases else ''}",
        "",
        "  by tag:",
    ]
    lines += [f"    {tag:<16} {rate}" for tag, rate in s.by_tag.items()]
    lines += [
        "",
        f"  latency p50/p95: {s.latency_p50_s}s / {s.latency_p95_s}s   "
        f"tokens: {s.tokens_in} in / {s.tokens_out} out   tool errors: {s.tool_errors}",
    ]
    if s.failures:
        lines += ["", "  failures (first repeat):"]
        lines += [f"    {f.case_id:<12} [{f.scorer}] {f.reason}" for f in s.failures]
    return "\n".join(lines)
