from agenteval.adapter import AgentResult
from agenteval.cases import Case, GradingMode
from agenteval.scorers import score_case


def _case(**kw) -> Case:
    base = dict(id="t", question="q", grading_mode="exact")
    base.update(kw)
    return Case.model_validate(base)


def test_exact_normalizes():
    case = _case(grading_mode="exact", expected="42")
    assert score_case(case, AgentResult(answer="  42. ")).passed


def test_keyword_reports_missing():
    case = _case(grading_mode="keyword", keywords=["alpha", "beta"])
    score = score_case(case, AgentResult(answer="only alpha here"))
    assert not score.passed
    assert "beta" in score.reason


def test_numeric_tolerance():
    case = _case(grading_mode="numeric", expected=840, tolerance=0.5)
    assert score_case(case, AgentResult(answer="The result is 840.2 vehicles")).passed
    assert not score_case(case, AgentResult(answer="around 900")).passed


def test_unanswerable_requires_decline():
    case = _case(grading_mode="unanswerable")
    assert score_case(case, AgentResult(answer="I don't have enough information.")).passed
    assert not score_case(case, AgentResult(answer="It was 12,400 vehicles.")).passed


def test_execution_and_judge_are_skipped_not_failed():
    for mode in (GradingMode.EXECUTION, GradingMode.JUDGE):
        score = score_case(_case(grading_mode=mode, expected="x"), AgentResult(answer="y"))
        assert score.skipped and not score.passed


def test_agent_error_fails():
    score = score_case(_case(expected="42"), AgentResult(error="boom"))
    assert not score.passed and "boom" in score.reason
