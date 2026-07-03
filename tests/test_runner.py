from agenteval.cases import load_dataset
from agenteval.metrics import summarize
from agenteval.runner import run_suite
from examples.toy_agent import ToyAgent


def test_toy_suite_end_to_end():
    cases = load_dataset("examples/toy_v1.jsonl")
    run = run_suite(ToyAgent(), cases, dataset_name="toy_v1", repeats=2)

    assert len(run.results) == len(cases) * 2

    s = summarize(run)
    assert s.cases_total == len(cases)
    assert s.cases_skipped == 0          # toy set has no execution/judge cases
    assert s.pass_at_1 >= 0.75           # deterministic agent, known dataset
    assert s.pass_pow_k <= 1.0

    # designed failures: toy-06 (missing keyword) and toy-11 (fabrication
    # trap: agent recites the AADT definition instead of declining)
    failed_ids = {f.case_id for f in s.failures}
    assert {"toy-06", "toy-11"} <= failed_ids


def test_transit_dataset_loads_and_skips_pending_modes():
    cases = load_dataset("datasets/transit_v1.jsonl")
    assert len(cases) == 25
    run = run_suite(ToyAgent(), cases, dataset_name="transit_v1", repeats=1)
    s = summarize(run)
    # execution (8) + judge (5) cases are skipped until S2/S3
    assert s.cases_skipped == 13
