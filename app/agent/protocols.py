"""Provider- and transport-neutral protocols for the agent control plane."""

from __future__ import annotations

from typing import Any, Protocol

from app.contracts.agent_runtime import (
    AgentDecision,
    AgentDecisionRequest,
    AgentModelIdentity,
    AgentToolObservation,
    AgentToolSpec,
)
from app.contracts.common import HealthComponent
from app.contracts.identity import PrincipalContext


class AgentDecisionModel(Protocol):
    """A replaceable model adapter that selects exactly one next action."""

    @property
    def identity(self) -> AgentModelIdentity: ...

    def health(self) -> HealthComponent: ...

    def decide(self, request: AgentDecisionRequest) -> AgentDecision: ...


class AgentToolContext(Protocol):
    """Minimum execution context shared by Python, HTTP, or future MCP tools."""

    @property
    def task_id(self) -> str: ...

    @property
    def job_id(self) -> str: ...

    @property
    def action_id(self) -> str: ...

    @property
    def principal(self) -> PrincipalContext: ...


class AgentTool(Protocol):
    """One validated capability exposed to a decision model."""

    @property
    def spec(self) -> AgentToolSpec: ...

    def execute(
        self,
        context: AgentToolContext,
        arguments: dict[str, Any],
    ) -> AgentToolObservation: ...
