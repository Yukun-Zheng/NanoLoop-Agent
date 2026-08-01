"""Typed tool registration and execution boundary."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, ValidationError

from app.agent.protocols import AgentTool, AgentToolContext
from app.contracts.agent_runtime import (
    AgentToolObservation,
    AgentToolOutcome,
    AgentToolSpec,
)


class UnknownAgentToolError(LookupError):
    pass


class InvalidAgentToolArgumentsError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class RegisteredAgentTool:
    tool: AgentTool
    arguments_model: type[BaseModel]


class AgentToolRegistry:
    """In-process registry whose public contract also fits HTTP or MCP adapters."""

    def __init__(self, tools: list[RegisteredAgentTool] | None = None) -> None:
        self._tools: dict[str, RegisteredAgentTool] = {}
        for item in tools or []:
            self.register(item)

    def register(self, item: RegisteredAgentTool) -> None:
        name = item.tool.spec.name
        if name in self._tools:
            raise ValueError(f"duplicate agent tool: {name}")
        expected_schema = item.arguments_model.model_json_schema()
        if item.tool.spec.input_schema != expected_schema:
            raise ValueError(f"tool schema does not match arguments model: {name}")
        self._tools[name] = item

    def specs(self) -> list[AgentToolSpec]:
        return [self._tools[name].tool.spec for name in sorted(self._tools)]

    def get(self, name: str) -> RegisteredAgentTool:
        try:
            return self._tools[name]
        except KeyError as error:
            raise UnknownAgentToolError(name) from error

    def execute(
        self,
        name: str,
        *,
        context: AgentToolContext,
        arguments: dict[str, Any],
    ) -> AgentToolObservation:
        registered, normalized = self.validate_arguments(name, arguments)
        observation = registered.tool.execute(context, normalized)
        if observation.outcome not in {AgentToolOutcome.OK, AgentToolOutcome.ERROR}:
            raise TypeError("tool returned an unsupported outcome")
        return observation

    def validate_arguments(
        self,
        name: str,
        arguments: dict[str, Any],
    ) -> tuple[RegisteredAgentTool, dict[str, Any]]:
        registered = self.get(name)
        try:
            parsed = registered.arguments_model.model_validate(arguments)
        except ValidationError as error:
            raise InvalidAgentToolArgumentsError(str(error)) from error
        normalized = parsed.model_dump(mode="python", exclude_none=True)
        return registered, normalized
