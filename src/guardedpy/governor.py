"""Deterministic governance for the continuous agent's fixed tool set."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import PurePosixPath
from typing import Literal, TypeAlias
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from guardedpy.config import HarnessConfig
from guardedpy.conversation import ToolCall, ToolDefinition, Turn

GovernanceVerdict: TypeAlias = Literal["allow", "approval_required", "deny"]


@dataclass(frozen=True)
class GovernanceDecision:
    verdict: GovernanceVerdict
    rule_id: str
    code: str
    normalized_call: str
    approval_id: UUID | None = None


class _Args(BaseModel):
    model_config = ConfigDict(extra="forbid")


class _ListArgs(_Args):
    path: str = "."


class _ReadArgs(_Args):
    path: str
    offset: int = Field(default=0, ge=0)
    limit: int = Field(default=200, ge=1, le=200)


class _PatchArgs(_Args):
    unified_diff: str = Field(max_length=65_536)


class _PytestArgs(_Args):
    nodes: tuple[str, ...] = Field(default=(), max_length=20)


class _NoArgs(_Args):
    pass


class _DeleteArgs(_Args):
    path: str


_MODELS: dict[str, type[_Args]] = {
    "list_files": _ListArgs, "read_file": _ReadArgs, "apply_patch": _PatchArgs,
    "run_pytest": _PytestArgs, "git_diff": _NoArgs, "git_status": _NoArgs,
    "delete_path": _DeleteArgs,
}


def governed_tool_definitions() -> tuple[ToolDefinition, ...]:
    return tuple(
        ToolDefinition(name, f"Governed {name} tool.", model.model_json_schema())
        for name, model in _MODELS.items()
    )


def parse_tool_call(call: ToolCall) -> _Args:
    model = _MODELS.get(call.name)
    if model is None:
        raise ValueError("invalid_tool_call")
    try:
        parsed = model.model_validate(json.loads(call.arguments_json))
    except (json.JSONDecodeError, ValidationError):
        raise ValueError("invalid_tool_call") from None
    path = getattr(parsed, "path", None)
    if path is not None:
        _validate_path(path)
    if isinstance(parsed, _PytestArgs):
        for node in parsed.nodes:
            path_text = node.split("::", 1)[0]
            _validate_path(path_text)
            if node.startswith("-"):
                raise ValueError("invalid_tool_call")
    return parsed


def _validate_path(value: object) -> None:
    if not isinstance(value, str) or not value or "\\" in value:
        raise ValueError("invalid_tool_call")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError("path_outside_project")
    if value != "." and any(part.startswith(".") for part in path.parts):
        raise ValueError("protected_path")


class ToolGovernor:
    def __init__(self, config: HarnessConfig) -> None:
        self._config = config
        self._approvals: dict[UUID, tuple[UUID, UUID, UUID, str, str]] = {}

    def normalized_call(self, call: ToolCall) -> str:
        parsed = parse_tool_call(call)
        return json.dumps(
            {"name": call.name, "arguments": parsed.model_dump(mode="json")},
            sort_keys=True, separators=(",", ":"),
        )

    def decide(self, turn: Turn, item_id: UUID, call: ToolCall) -> GovernanceDecision:
        try:
            normalized = self.normalized_call(call)
        except ValueError as error:
            return GovernanceDecision("deny", "tool.schema", str(error), "")
        if turn.mode in ("plan", "review") and call.name not in {"list_files", "read_file", "git_diff", "git_status"}:
            return GovernanceDecision("deny", "mode.read_only", "mode_read_only", normalized)
        if call.name == "delete_path":
            approval_id = uuid4()
            self._approvals[approval_id] = (turn.session_id, turn.id, item_id, normalized, "pending")
            return GovernanceDecision("approval_required", "delete.approval", "approval_required", normalized, approval_id)
        return GovernanceDecision("allow", "tool.contained", "allowed", normalized)

    def resolve(self, session_id: UUID, turn_id: UUID, item_id: UUID, normalized_call: str, approval_id: UUID, accepted: bool) -> GovernanceDecision:
        expected = self._approvals.get(approval_id)
        if expected != (session_id, turn_id, item_id, normalized_call, "pending"):
            return GovernanceDecision("deny", "approval.identity", "approval_stale", normalized_call)
        self._approvals[approval_id] = (*expected[:4], "resolved")
        code = "allowed" if accepted else "approval_rejected"
        return GovernanceDecision("allow" if accepted else "deny", "approval.once", code, normalized_call, approval_id)
