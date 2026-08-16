"""Re-grade a saved run with the current scorer. No model calls.

A run file holds two different kinds of thing, and it is worth being explicit
about which is which:

  observations   what the agent actually did — answer, trajectory, tokens,
                 latency. Empirical, permanent, and untouched by anything we
                 change in this repo afterwards.
  judgments      passed / skipped / scorer / reason. *Derived* — the output of
                 running a scorer over those observations. Provisional by
                 nature: change the scorer and the correct judgment changes,
                 though nothing about the agent did.

Rescoring throws away the judgments, keeps the observations, and re-derives.
That is why persisting the trajectory mattered: before it, only the derived
number survived, so a scorer fix could not be applied to history.

Why it earns its place:

  * Model calls cost money and minutes; scoring is free and instant. You will
    change the scorer far more often than you will pay for a sweep.
  * **It isolates one variable.** To know whether a grading change helped, the
    trajectories must be held fixed. A fresh re-run changes the grader *and*
    resamples the model, and then the movement is unattributable.
  * Every saved run becomes a regression fixture for the scorer itself — change
    `execution.py`, rescore the corpus, and see whether any verdict flipped
    that you did not intend.

What it deliberately cannot do:

  * Re-measure latency, tokens or cost. Those belong to the original call and
    are carried through unchanged — never present them as fresh.
  * Tell you what the model does *today*. Rescoring a July run yields a
    corrected July result, not an August one.
  * Recover a run whose trajectory was never recorded (see `_trajectory_lost`).
  * Be reproducible across a changing database. EXECUTION grading re-executes
    SQL, so a rescore is only stable relative to a fixed DB.
"""

from __future__ import annotations

import hashlib
import uuid
from datetime import datetime, timezone
from pathlib import Path

from pydantic import BaseModel

from .adapter import AgentResult
from .cases import Case
from .runner import RepeatResult, RunRecord
from .scorers import Score, score_case


class Flip(BaseModel):
    """One verdict that changed. The point of a rescore is *which* moved."""
    case_id: str
    repeat: int
    was: str          # PASS | FAIL | SKIP
    now: str
    reason: str       # the new scorer's reason


def _verdict(passed: bool, skipped: bool) -> str:
    return "SKIP" if skipped else ("PASS" if passed else "FAIL")


def _trajectory_lost(r: RepeatResult) -> bool:
    """True when the trajectory was never persisted, as opposed to never existing.

    These are different facts and must not be conflated. An execution case with
    no tool calls is the fabrication trap firing — a legitimate FAIL. A run
    recorded before trajectory persistence shipped has `tool_calls: 3` and an
    empty trajectory: the agent *did* query, we simply failed to write it down.
    Scoring that as a failure would invent a result; it has to be skipped.
    """
    return r.tool_calls > 0 and not r.trajectory


def _as_agent_result(r: RepeatResult) -> AgentResult:
    """Rebuild the scorer's input from stored observations.

    Lossless by construction: every field AgentResult carries is already on
    RepeatResult, which is the property that makes rescoring possible at all.
    """
    return AgentResult(
        answer=r.answer,
        trajectory=r.trajectory,
        tokens_in=r.tokens_in,
        tokens_out=r.tokens_out,
        error=r.agent_error,
    )


def dataset_fingerprint(path: str | Path) -> str:
    """Short content hash of the dataset.

    Recorded because a rescore silently mixes two changes if the golden set was
    edited since the run — a question tightened, gold SQL fixed. The hash does
    not prevent that; it makes it visible, which is the most the harness can
    honestly do.
    """
    p = Path(path)
    if not p.exists():
        return "missing"
    return hashlib.sha256(p.read_bytes()).hexdigest()[:12]


def rescore_run(run: RunRecord, cases: list[Case],
                dataset_path: str | Path | None = None) -> tuple[RunRecord, list[Flip]]:
    """Re-derive every judgment in `run` from its stored observations.

    Returns the new record and the list of verdicts that moved. The input is
    never mutated: the old verdicts are the "before" half of any honest
    write-up, and overwriting them destroys the only evidence that the harness
    was ever wrong.
    """
    by_id = {c.id: c for c in cases}
    fresh: list[RepeatResult] = []
    flips: list[Flip] = []

    for r in run.results:
        case = by_id.get(r.case_id)

        if case is None:
            score = Score(passed=False, skipped=True, scorer="rescore",
                          reason=f"case {r.case_id!r} is not in the dataset any more")
        elif _trajectory_lost(r):
            score = Score(passed=False, skipped=True, scorer="rescore",
                          reason=(f"no trajectory recorded ({r.tool_calls} tool call(s) "
                                  "happened but predate trajectory persistence) — "
                                  "cannot re-derive"))
        else:
            score = score_case(case, _as_agent_result(r))

        # Observations copied verbatim; only the judgment fields are replaced.
        fresh.append(r.model_copy(update={
            "passed": score.passed,
            "skipped": score.skipped,
            "scorer": score.scorer,
            "reason": score.reason,
        }))

        was, now = _verdict(r.passed, r.skipped), _verdict(score.passed, score.skipped)
        if was != now:
            flips.append(Flip(case_id=r.case_id, repeat=r.repeat,
                              was=was, now=now, reason=score.reason))

    new = RunRecord(
        run_id=f"{datetime.now(timezone.utc):%Y%m%d-%H%M%S}-{uuid.uuid4().hex[:6]}",
        created_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        adapter=run.adapter,
        dataset=run.dataset,
        repeats=run.repeats,
        results=fresh,
        rescored_from=run.run_id,
        rescored_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        dataset_sha=dataset_fingerprint(dataset_path or run.dataset),
    )
    return new, flips


def render_flips(run: RunRecord, flips: list[Flip], *, limit: int = 25) -> str:
    """Human-readable diff. The new pass rate is not the interesting part."""
    lines = [
        f"rescored {run.rescored_from} -> {run.run_id}",
        f"  adapter={run.adapter}  dataset={run.dataset}  "
        f"sha={run.dataset_sha}  repeats={run.repeats}",
        "",
    ]
    if not flips:
        lines.append("  no verdicts changed — the scorer agrees with the saved run.")
        return "\n".join(lines)

    gained = sum(1 for f in flips if f.now == "PASS")
    lost = sum(1 for f in flips if f.was == "PASS")
    skipped = sum(1 for f in flips if f.now == "SKIP")
    lines.append(f"  {len(flips)} verdict(s) changed: "
                 f"{gained} now pass, {lost} no longer pass, {skipped} now skipped")
    # Losses first: a rescore that quietly breaks passing cases is the outcome
    # you most need to see, and it is the one a happy headline number hides.
    ordered = sorted(flips, key=lambda f: (f.was != "PASS", f.case_id, f.repeat))
    lines.append("")
    for f in ordered[:limit]:
        lines.append(f"  {f.case_id:<8} r{f.repeat}  {f.was} -> {f.now}   {f.reason[:78]}")
    if len(ordered) > limit:
        lines.append(f"  ... and {len(ordered) - limit} more")
    return "\n".join(lines)
