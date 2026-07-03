"""Serial runner: dataset x repeats -> RunRecord (saved as JSON).

Repeats are first-class because run-to-run reliability (pass^k) is the
project's signature metric. Concurrency, hard timeouts, cost caps and
response caching are deliberate later additions (S4) — keep S1 simple.
"""

from __future__ import annotations

import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

from pydantic import BaseModel, Field

from .adapter import AgentAdapter, AgentResult
from .cases import Case
from .scorers import score_case


class RepeatResult(BaseModel):
    case_id: str
    repeat: int
    tags: list[str] = Field(default_factory=list)
    answer: str = ""
    passed: bool = False
    skipped: bool = False
    scorer: str = ""
    reason: str = ""
    latency_s: float = 0.0
    tokens_in: int = 0
    tokens_out: int = 0
    tool_calls: int = 0
    tool_errors: int = 0
    agent_error: str | None = None
    # TODO(S5): trace_url — link every failure to its exact trace.


class RunRecord(BaseModel):
    run_id: str
    created_at: str
    adapter: str
    dataset: str
    repeats: int
    results: list[RepeatResult] = Field(default_factory=list)


def run_suite(adapter: AgentAdapter, cases: list[Case], dataset_name: str,
              repeats: int = 1) -> RunRecord:
    run = RunRecord(
        run_id=f"{datetime.now(timezone.utc):%Y%m%d-%H%M%S}-{uuid.uuid4().hex[:6]}",
        created_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        adapter=adapter.name,
        dataset=dataset_name,
        repeats=repeats,
    )
    for case in cases:
        for rep in range(repeats):
            t0 = time.perf_counter()
            try:
                result = adapter.run(case)
            except Exception as e:  # adapters shouldn't raise, but never crash the suite
                result = AgentResult(error=f"{type(e).__name__}: {e}")
            latency = time.perf_counter() - t0

            score = score_case(case, result)
            run.results.append(RepeatResult(
                case_id=case.id,
                repeat=rep,
                tags=case.tags,
                answer=result.answer,
                passed=score.passed,
                skipped=score.skipped,
                scorer=score.scorer,
                reason=score.reason,
                latency_s=round(latency, 4),
                tokens_in=result.tokens_in,
                tokens_out=result.tokens_out,
                tool_calls=len(result.trajectory),
                tool_errors=sum(1 for t in result.trajectory if t.error),
                agent_error=result.error,
            ))
    return run


def save_run(run: RunRecord, out_dir: str | Path = "reports") -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"run_{run.run_id}.json"
    path.write_text(run.model_dump_json(indent=2))
    return path


def load_run(path: str | Path) -> RunRecord:
    return RunRecord.model_validate_json(Path(path).read_text())
