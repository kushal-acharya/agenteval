#!/usr/bin/env python3
"""Serve the dashboard with the newest run already loaded.

    python dashboard/serve.py            # newest run in reports/
    python dashboard/serve.py <run.json> # a specific run

Why this exists: opened over file://, the page cannot read sibling files —
Chrome blocks it — so the only way in was a file picker. Served over HTTP the
page can fetch its own data, so the run loads on open.

It also does the join the run file can't: a RunRecord stores case_id but not
the question or the gold SQL, and it stores the agent's SQL but not what that
SQL returned. This reads the dataset and (read-only) replays both queries so
the dashboard can show gold vs agent result sets side by side.
"""

from __future__ import annotations

import http.server
import json
import posixpath
import socketserver
import sqlite3
import sys
import threading
import time
import urllib.parse
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REPORTS = ROOT / "reports"
PORT = 8765
#: Same bound the scorer uses. A debugging view that can hang is worse than none.
REPLAY_TIMEOUT_S = 5.0


def newest_run() -> Path:
    runs = sorted(REPORTS.glob("run_*.json"), key=lambda p: p.stat().st_mtime)
    if not runs:
        sys.exit(f"No run files in {REPORTS}. Run `agenteval run ...` first.")
    return runs[-1]


def load_cases(dataset: str) -> dict[str, dict]:
    path = ROOT / dataset
    if not path.exists():
        return {}
    out = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            c = json.loads(line)
            out[c["id"]] = c
    return out


def run_sql(db: Path, sql: str, limit: int = 50):
    """Replay a query read-only. Returns (cols, rows, error) — never raises.

    Mirrors agenteval.execution.run_query's guarantees on purpose: mode=ro,
    one statement, wall-clock bound. The dashboard replays whatever SQL a model
    wrote, so it needs the same protections the scorer has.

    The error is *returned* rather than swallowed: "the query failed and here is
    why" and "there is no database" are different things, and a debugging view
    that renders both as an empty table is lying to you.
    """
    if not sql or not db or not db.exists():
        return None, None, None
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    deadline = time.monotonic() + REPLAY_TIMEOUT_S
    con.set_progress_handler(lambda: int(time.monotonic() > deadline), 10_000)
    try:
        cur = con.execute(sql)
        cols = [d[0] for d in cur.description] if cur.description else []
        return cols, [list(r) for r in cur.fetchmany(limit)], None
    except sqlite3.Error as e:
        return None, None, f"{type(e).__name__}: {e}"
    finally:
        con.close()


def build(run_path: Path) -> dict:
    raw = json.loads(run_path.read_text())
    cases_meta = load_cases(raw.get("dataset", ""))

    by_case: dict[str, list] = {}
    for r in raw.get("results", []):
        by_case.setdefault(r["case_id"], []).append(r)

    cases = []
    for cid, reps in by_case.items():
        reps.sort(key=lambda r: r["repeat"])
        meta = cases_meta.get(cid, {})
        db = ROOT / meta["db"] if meta.get("db") else None
        gold_sql = meta.get("expected") if meta.get("grading_mode") == "execution" else None

        gold_cols, gold_rows, gold_err = (
            run_sql(db, gold_sql) if (db and isinstance(gold_sql, str)) else (None, None, None))

        results = []
        for r in reps:
            traj = r.get("trajectory") or []
            steps = []
            for i, t in enumerate(traj):
                args = t.get("arguments") or {}
                steps.append({
                    "n": t.get("name", "tool"),
                    "sql": args.get("sql") or args.get("query") or json.dumps(args),
                    "out": (t.get("output") or "")[:600],
                    "err": t.get("error"),
                    "graded": i == len(traj) - 1,
                })
            agent_sql = steps[-1]["sql"] if steps else None
            a_cols, a_rows, a_err = (
                run_sql(db, agent_sql) if (db and agent_sql) else (None, None, None))
            results.append({
                "pass": bool(r.get("passed")), "skipped": bool(r.get("skipped")),
                "reason": r.get("reason", ""), "answer": r.get("answer", ""),
                "latency": round(r.get("latency_s", 0), 2),
                "tin": r.get("tokens_in", 0), "tout": r.get("tokens_out", 0),
                "sql": agent_sql,
                "cols": a_cols or [], "rows": a_rows, "sql_error": a_err,
                "steps": steps,
            })

        cases.append({
            "id": cid,
            "tags": reps[0].get("tags", []),
            "question": meta.get("question", cid),
            "gold": gold_sql,
            "gold_cols": gold_cols,
            "gold_rows": gold_rows,
            "gold_error": gold_err,
            "unanswerable": meta.get("grading_mode") == "unanswerable",
            "results": results,
        })

    return {
        "run_id": raw.get("run_id", run_path.stem), "created_at": raw.get("created_at", ""),
        "adapter": raw.get("adapter", ""), "dataset": raw.get("dataset", ""),
        "repeats": raw.get("repeats", 1), "cases": cases,
        "taxonomy": [], "sample": False,
    }


def main() -> None:
    run_path = Path(sys.argv[1]) if len(sys.argv) > 1 else newest_run()
    view = build(run_path)
    out = REPORTS / "latest.json"
    out.write_text(json.dumps(view, indent=2, default=str))

    n_fail = sum(1 for c in view["cases"] for r in c["results"] if not r["pass"])
    print(f"→ {run_path.name}: {len(view['cases'])} cases × {view['repeats']} repeats, "
          f"{n_fail} failed attempts")
    print(f"→ wrote {out.relative_to(ROOT)}")

    class Handler(http.server.SimpleHTTPRequestHandler):
        """Serves the dashboard and its data — and nothing else.

        The handler is rooted at the repo so the page can reach
        ../reports/latest.json, which means without this allowlist it would
        happily serve `.env` (a live API key) and `.git/` to anything that can
        reach localhost. Binding to 127.0.0.1 is not sufficient on its own:
        any local process, and any page you visit that guesses the port, can
        issue the request.
        """

        ALLOWED = ("/dashboard", "/reports/latest.json")

        def __init__(self, *a, **kw):
            super().__init__(*a, directory=str(ROOT), **kw)

        def send_head(self):
            path = posixpath.normpath(urllib.parse.unquote(
                urllib.parse.urlparse(self.path).path))
            if path == "/":
                self.send_response(302)
                self.send_header("Location", "/dashboard/")
                self.end_headers()
                return None
            hidden = any(p.startswith(".") for p in path.split("/") if p)
            if hidden or not any(path == a or path.startswith(a + "/")
                                 for a in self.ALLOWED):
                self.send_error(404, "Not found")
                return None
            return super().send_head()

        def log_message(self, *a):  # keep the console quiet
            pass

    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.TCPServer(("127.0.0.1", PORT), Handler) as httpd:
        url = f"http://127.0.0.1:{PORT}/dashboard/"
        print(f"→ {url}   (ctrl-c to stop)")
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nstopped")


if __name__ == "__main__":
    main()
