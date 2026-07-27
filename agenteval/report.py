"""Plain-text report. TODO(S5): HTML report with failure->trace links."""

from __future__ import annotations

from pathlib import Path

from .metrics import Summary, summarize
from .runner import RunRecord, load_run


def render(run: RunRecord) -> str:
    s: Summary = summarize(run)
    lines = [
        f"AgentEval run {run.run_id}",
        f"  adapter={run.adapter}  dataset={run.dataset}  repeats={run.repeats}",
        "",
        f"  cases: {s.cases_total}   graded: {s.cases_graded}   "
        f"skipped: {s.cases_skipped}"
        # Skips are never silent: judge cases (S3) and any case the harness can't
        # grade are excluded from the rate rather than counted as failures.
        f"{' (ungradable — see failure reasons)' if s.cases_skipped else ''}",
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


def render_history(reports_dir: str | Path = "reports",
                   dataset: str | None = None) -> str:
    """Every saved run, oldest first — the record of what was measured when.

    Derived by scanning `reports/` rather than kept as a separate ledger: a
    ledger can drift from the run files it describes, and the run files are the
    evidence. Slower, always true.
    """
    paths = sorted(Path(reports_dir).glob("run_*.json"))
    rows: list[tuple[RunRecord, Summary]] = []
    for p in paths:
        try:
            run = load_run(p)
        except Exception:   # a half-written or hand-edited file must not hide the rest
            continue
        if dataset and Path(run.dataset).name != Path(dataset).name:
            continue
        rows.append((run, summarize(run)))

    if not rows:
        return f"No runs in {reports_dir}/. Run `agenteval run ...` first."

    head = (f"  {'when':<17}{'adapter':<26}{'dataset':<18}{'k':>2}"
            f"{'cases':>7}{'pass@1':>9}{'pass^k':>9}{'gap':>7}{'flaky':>7}")
    lines = [f"{len(rows)} run(s) in {reports_dir}/", "", head, "  " + "-" * (len(head) - 2)]
    for run, s in rows:
        gap = (s.pass_at_1 - s.pass_pow_k) * 100
        lines.append(
            f"  {run.created_at[:16].replace('T', ' '):<17}"
            f"{run.adapter.split(':')[-1][:25]:<26}"
            f"{Path(run.dataset).name[:17]:<18}"
            f"{s.k:>2}{s.cases_graded:>7}{s.pass_at_1:>8.1%}{s.pass_pow_k:>9.1%}"
            f"{gap:>6.1f}{'':1}{len(s.flaky_cases):>7}"
        )
    # The gap is the point of the table: a run that cannot produce one has not
    # measured reliability, whatever its pass rate says.
    if all(abs(s.pass_at_1 - s.pass_pow_k) < 1e-9 for _, s in rows):
        lines += ["", "  Every run shows a 0.0-point gap: nothing here has measured",
                  "  reliability yet — either the agent is deterministic or the set",
                  "  is too easy to discriminate."]
    return "\n".join(lines)
