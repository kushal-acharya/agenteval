"""The agent adapter — THE key design decision.

Anything that implements `AgentAdapter.run(case) -> AgentResult` can be
evaluated: your raw loop, a LangGraph app, an HTTP endpoint, Claude Code.
This is what makes AgentEval a harness, not a script welded to one agent.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from pydantic import BaseModel, Field

from .cases import Case


class ToolCall(BaseModel):
    name: str
    arguments: dict = Field(default_factory=dict)
    output: str = ""
    error: str | None = None  # non-None => counted as a tool error


class AgentResult(BaseModel):
    answer: str = ""
    trajectory: list[ToolCall] = Field(default_factory=list)
    tokens_in: int = 0
    tokens_out: int = 0
    error: str | None = None  # agent crashed / gave up entirely


class AgentAdapter(ABC):
    """Implement this once per agent-under-test."""

    #: human-readable name recorded in run files
    name: str = "unnamed-agent"

    @abstractmethod
    def run(self, case: Case) -> AgentResult:
        """Answer one case. Must not raise — catch and set .error instead."""
        raise NotImplementedError
