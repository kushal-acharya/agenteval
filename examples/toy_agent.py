"""A deterministic 2-tool toy agent so the harness demos with NO API key.

Tools: glossary_lookup (transportation terms) + calculator.
It declines anything else — which is exactly what UNANSWERABLE cases test.

Swap in a real agent by implementing the same AgentAdapter interface
(see README "Plugging in your own agent").
"""

from __future__ import annotations

import ast
import operator
import re

from agenteval.adapter import AgentAdapter, AgentResult, ToolCall
from agenteval.cases import Case

GLOSSARY = {
    "aadt": "AADT stands for Annual Average Daily Traffic — total yearly traffic volume divided by 365.",
    "kabco": "KABCO is the injury severity scale: K fatal, A suspected serious injury, B suspected minor injury, C possible injury, O property damage only.",
    "crash modification factor": "A Crash Modification Factor is a multiplier for the expected change in crash frequency after a countermeasure is applied.",
    "cmf": "A Crash Modification Factor is a multiplier for the expected change in crash frequency after a countermeasure is applied.",
    "85th percentile": "The 85th percentile speed is the speed at or below which 85 percent of vehicles travel; commonly used to set speed limits.",
    "hsip": "HSIP is the Highway Safety Improvement Program, a federal-aid program to reduce fatalities and serious injuries on public roads.",
    "level of service": "Level of Service grades traffic operations from A (free flow) to F (breakdown).",
    "los": "Level of Service grades traffic operations from A (free flow) to F (breakdown).",
    "vmt": "VMT means Vehicle Miles Traveled, the total miles driven by all vehicles in an area over a period.",
    "pdo": "PDO means a property damage only crash — no injuries.",
}

_EXPR_RE = re.compile(r"-?\d+(?:\.\d+)?(?:\s*[-+*/]\s*-?\d+(?:\.\d+)?)+")
_OPS = {ast.Add: operator.add, ast.Sub: operator.sub,
        ast.Mult: operator.mul, ast.Div: operator.truediv,
        ast.USub: operator.neg}


def _safe_eval(expr: str) -> float:
    def ev(node):
        if isinstance(node, ast.Expression):
            return ev(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
            return _OPS[type(node.op)](ev(node.left), ev(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
            return _OPS[type(node.op)](ev(node.operand))
        raise ValueError(f"unsupported expression: {expr}")
    return ev(ast.parse(expr, mode="eval"))


class ToyAgent(AgentAdapter):
    name = "toy-agent"

    def run(self, case: Case) -> AgentResult:
        q = case.question.lower()
        trajectory: list[ToolCall] = []

        # tool 1: calculator
        m = _EXPR_RE.search(case.question)
        if m:
            try:
                value = _safe_eval(m.group())
                answer = f"{value:g}"
                trajectory.append(ToolCall(name="calculator",
                                           arguments={"expression": m.group()},
                                           output=answer))
                return AgentResult(answer=answer, trajectory=trajectory)
            except ValueError as e:
                trajectory.append(ToolCall(name="calculator",
                                           arguments={"expression": m.group()},
                                           error=str(e)))

        # tool 2: glossary lookup
        for term, definition in GLOSSARY.items():
            if term in q:
                trajectory.append(ToolCall(name="glossary_lookup",
                                           arguments={"term": term},
                                           output=definition))
                return AgentResult(answer=definition, trajectory=trajectory)

        return AgentResult(
            answer="I don't have enough information to answer that.",
            trajectory=trajectory,
        )
