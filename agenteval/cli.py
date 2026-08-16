"""CLI: `agenteval run`, `agenteval report`, `agenteval diff` (S4).

Example (2-minute demo, no API key needed):
    agenteval run --dataset examples/toy_v1.jsonl \
                  --adapter examples.toy_agent:ToyAgent --repeats 3
"""

from __future__ import annotations

import argparse
import importlib
import os
import sys

from .cases import load_dataset
from .report import render, render_history
from .rescore import render_flips, rescore_run
from .runner import load_run, run_suite, save_run


def _load_adapter(spec: str):
    """Load 'package.module:ClassName' from the current working directory."""
    module_name, _, class_name = spec.partition(":")
    if not class_name:
        raise SystemExit(f"adapter must look like 'module:ClassName', got {spec!r}")
    sys.path.insert(0, os.getcwd())
    cls = getattr(importlib.import_module(module_name), class_name)
    adapter = cls()
    # Only name it after the import spec if the adapter didn't name itself. An
    # adapter knows things the spec cannot express — which model it is running —
    # and without that a Haiku run and an Opus run are indistinguishable in the
    # run record, making a model comparison impossible after the fact.
    if "name" not in vars(adapter):
        adapter.name = spec
    return adapter


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="agenteval")
    sub = parser.add_subparsers(dest="command", required=True)

    p_run = sub.add_parser("run", help="run a dataset against an agent adapter")
    p_run.add_argument("--dataset", required=True)
    p_run.add_argument("--adapter", required=True, help="module:ClassName")
    p_run.add_argument("--repeats", type=int, default=1)
    p_run.add_argument("--out", default="reports")

    p_rep = sub.add_parser("report", help="print the report for a saved run")
    p_rep.add_argument("run_file")

    p_hist = sub.add_parser("history", help="every saved run, oldest first")
    p_hist.add_argument("--dir", default="reports")
    p_hist.add_argument("--dataset", help="only runs of this dataset")

    p_res = sub.add_parser(
        "rescore",
        help="re-grade a saved run with the current scorer (no model calls)")
    p_res.add_argument("run_file")
    p_res.add_argument("--dataset", help="override the dataset recorded in the run")
    p_res.add_argument("--out", default="reports")

    sub.add_parser("diff", help="regression diff between two runs (lands in S4)")

    args = parser.parse_args(argv)

    if args.command == "run":
        cases = load_dataset(args.dataset)
        adapter = _load_adapter(args.adapter)
        run = run_suite(adapter, cases, dataset_name=args.dataset, repeats=args.repeats)
        path = save_run(run, args.out)
        print(render(run))
        print(f"\n  saved: {path}")
        return 0

    if args.command == "report":
        print(render(load_run(args.run_file)))
        return 0

    if args.command == "history":
        print(render_history(args.dir, dataset=args.dataset))
        return 0

    if args.command == "rescore":
        old = load_run(args.run_file)
        dataset = args.dataset or old.dataset
        cases = load_dataset(dataset)
        new, flips = rescore_run(old, cases, dataset_path=dataset)
        path = save_run(new, args.out)
        # The diff comes first and the report second, deliberately: the useful
        # output of a rescore is which verdicts moved, not the headline rate.
        print(render_flips(new, flips))
        print()
        print(render(new))
        print(f"\n  saved: {path}   (original {args.run_file} untouched)")
        return 0

    if args.command == "diff":
        # TODO(S4): newly-failing / newly-fixed / flaky / cost+latency delta,
        # then wire into CI as the regression gate (S5).
        raise SystemExit("`agenteval diff` lands in S4 — see the build order.")

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
