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


def build(run_path: Path, _cache: dict | None = None) -> dict:
    raw = json.loads(run_path.read_text())
    cases_meta = load_cases(raw.get("dataset", ""))
    # Replaying the same SQL once per repeat is pure waste — agents converge on
    # near-identical final queries, so this collapses ~50 executions to a handful.
    cache: dict = {} if _cache is None else _cache

    def replay(db, sql):
        key = (str(db), sql)
        if key not in cache:
            cache[key] = run_sql(db, sql)
        return cache[key]

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
            replay(db, gold_sql) if (db and isinstance(gold_sql, str)) else (None, None, None))

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
                replay(db, agent_sql) if (db and agent_sql) else (None, None, None))
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
        "sample": False,
    }


def _esc(s: str) -> str:
    """The taxonomy renderer inserts `d` as raw HTML, and `d` carries model-written
    SQL. Escape here so a query containing angle brackets cannot inject markup."""
    return (str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))


#: (match on the scorer's reason, title, why it happens). Order matters — first hit wins.
FAILURE_KINDS = [
    ("column count", "Output-contract violation",
     "The answer was right; the <em>shape</em> was not. The agent appended context "
     "columns the question never asked for. Nothing to do with SQL ability — it is "
     "instruction-following on output format, and it is the single largest failure "
     "class for smaller models."),
    ("no answer", "Right query, nothing said",
     "The result set matched and the agent returned no prose at all. Execution "
     "grading certifies the query; it must not also certify an answer that was "
     "never given to the user."),
    ("different rows", "Wrong values — a real error",
     "The query ran and returned the wrong numbers. This is the class that matters: "
     "the output is plausible, well-formatted, and wrong."),
    ("row count", "Wrong grain",
     "Right table, wrong unit of analysis — rows per crash where the question asked "
     "per person, or a grouped result where one value was wanted."),
    ("final query errored", "Broken SQL",
     "The agent's last query failed against the database — a hallucinated column or "
     "an invalid join. Visible and self-announcing, so the least dangerous class."),
    ("never called", "Answered without querying",
     "The agent produced a fluent answer having run no query at all. Fabrication in "
     "its purest form."),
    ("declining", "Fabrication under absence",
     "Asked for data the database does not contain, the agent reported a finding "
     "instead of declining. The failure that matters most in a liability-bearing "
     "domain: a stated zero reads as evidence, not as absence."),
]


def taxonomy(view: dict) -> list[dict]:
    """Group every failed attempt by why it failed, with a real example each.

    The pass rate says something is wrong; this says what — and unlike the rate,
    it survives a change of model, which is what makes it worth reporting.
    """
    buckets: dict[str, dict] = {}
    for case in view["cases"]:
        for i, r in enumerate(case["results"]):
            if r["pass"] or r["skipped"]:
                continue
            why = r.get("reason", "")
            title, expl = next(((t, e) for m, t, e in FAILURE_KINDS if m in why),
                               ("Other", "Uncategorised failure."))
            b = buckets.setdefault(title, {"n": 0, "t": title, "why": expl, "ex": None})
            b["n"] += 1
            if b["ex"] is None:
                b["ex"] = {"case": case["id"], "repeat": i + 1, "reason": why,
                           "gold": case.get("gold"), "sql": r.get("sql")}
    out = []
    for b in sorted(buckets.values(), key=lambda x: -x["n"]):
        ex, d = b["ex"], b["why"]
        if ex:
            d += (f"<br><br><b>{_esc(ex['case'])} · repeat {ex['repeat']}</b> — "
                  f"<code>{_esc(ex['reason'])}</code>")
            if ex.get("gold") and ex.get("sql"):
                d += (f"<br><br>gold: <code>{_esc(ex['gold'][:150])}</code>"
                      f"<br>agent: <code>{_esc(ex['sql'][:150])}</code>")
        out.append({"n": b["n"], "t": b["t"], "d": d})
    return out


def summarize(view: dict) -> dict:
    """Headline numbers for one run — mirrors agenteval.metrics.summarize."""
    graded = [c for c in view["cases"] if not (c["results"] and c["results"][0]["skipped"])]
    n = len(graded) or 1
    p1 = sum(1 for c in graded if c["results"][0]["pass"]) / n
    pk = sum(1 for c in graded if all(r["pass"] for r in c["results"])) / n
    flaky = [c["id"] for c in graded
             if any(r["pass"] for r in c["results"]) and not all(r["pass"] for r in c["results"])]
    return {"run_id": view["run_id"], "created_at": view["created_at"],
            "adapter": view["adapter"], "dataset": Path(view["dataset"]).name,
            "repeats": view["repeats"], "cases": len(graded),
            "p1": round(p1, 4), "pk": round(pk, 4), "gap": round((p1 - pk) * 100, 1),
            "flaky": len(flaky)}


def main() -> None:
    args = [a for a in sys.argv[1:] if a != "--publish"]
    publish = "--publish" in sys.argv
    explicit = Path(args[0]) if args else None
    paths = sorted(REPORTS.glob("run_*.json"), key=lambda p: p.stat().st_mtime)
    if not paths:
        sys.exit(f"No run files in {REPORTS}. Run `agenteval run ...` first.")

    # Build every run, not just one: "keep a history" means the page can move
    # between runs without a server round-trip, and a comparison across models
    # is only possible when they are all in hand at once.
    views, history, cache = [], [], {}   # one replay cache across all runs
    for p in paths:
        try:
            v = build(p, cache)
        except Exception as e:                    # one bad file must not sink the rest
            print(f"  ! skipped {p.name}: {type(e).__name__}: {e}")
            continue
        v["taxonomy"] = taxonomy(v)
        views.append(v)
        history.append(summarize(v))

    active = next((i for i, v in enumerate(views)
                   if explicit and v["run_id"] in explicit.name), len(views) - 1)
    payload = dict(views[active])
    payload["history"] = history
    payload["runs"] = views
    payload["active"] = active

    if publish:
        # GitHub Pages serves a directory, not a server, so the published page
        # gets its data as a *sibling* file. Everything here is already public —
        # FARS questions, gold SQL, and the SQL a model wrote against it — but
        # this is the one command in the repo that puts bytes on the internet,
        # so it is explicit and separate from the local view rather than a flag
        # on it.
        # ToyAgent is a deterministic fixture for the test suite, not a result.
        # Publishing it puts "83% pass@1" next to real model numbers on a page
        # whose whole point is that the numbers mean something.
        real = [i for i, v in enumerate(views) if "toy_agent" not in v["adapter"].lower()]
        if not real:
            sys.exit("Nothing to publish: every run is a ToyAgent fixture.")
        pub = dict(views[real[-1]])
        pub["runs"] = [views[i] for i in real]
        pub["history"] = [history[i] for i in real]
        pub["active"] = len(real) - 1

        docs = ROOT / "docs"
        docs.mkdir(exist_ok=True)
        (docs / "data.json").write_text(json.dumps(pub, default=str))
        (docs / "index.html").write_text((ROOT / "dashboard" / "index.html").read_text())
        (docs / ".nojekyll").write_text("")   # keep Pages from eating dotfiles
        size = (docs / "data.json").stat().st_size / 1024
        print(f"→ published {len(real)} real run(s) → docs/ ({size:.0f} KB)")
        for i in real:
            print(f"    {history[i]['adapter'][:34]:<35} pass@1 {history[i]['p1']:.0%}  "
                  f"pass^k {history[i]['pk']:.0%}")
        print("  commit docs/, then Settings → Pages → source: /docs")
        return

    out = REPORTS / "latest.json"
    out.write_text(json.dumps(payload, indent=2, default=str))

    print(f"→ {len(views)} run(s) built; showing {history[active]['run_id']}")
    for i, h in enumerate(history):
        mark = "→" if i == active else " "
        print(f"  {mark} {h['created_at'][:16].replace('T',' ')}  {h['adapter'][:34]:<35}"
              f"k={h['repeats']}  pass@1 {h['p1']:.0%}  pass^k {h['pk']:.0%}  "
              f"gap {h['gap']:.1f}  flaky {h['flaky']}")
    print(f"→ wrote {out.relative_to(ROOT)} ({out.stat().st_size/1024:.0f} KB)")

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
