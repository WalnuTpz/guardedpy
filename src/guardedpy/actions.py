"""Validated action schema for untrusted model output."""

from __future__ import annotations

from hashlib import sha256
import json
from typing import Annotated, Literal, TypeAlias

from pydantic import BaseModel, Field, TypeAdapter


MEMORY_PROPOSAL_TEXT_MAX_LENGTH = 500


class _ActionBase(BaseModel):
    summary: str

    def stable_hash(self) -> str:
        return stable_hash(self)


class ListFilesAction(_ActionBase):
    kind: Literal["list_files"]
    path: str = "."


class ReadFileAction(_ActionBase):
    kind: Literal["read_file"]
    path: str
    offset: int = 0
    limit: int = 200


class ApplyPatchAction(_ActionBase):
    kind: Literal["apply_patch"]
    diff: str


class DeletePathAction(_ActionBase):
    kind: Literal["delete_path"]
    path: str


class RunPytestAction(_ActionBase):
    kind: Literal["run_pytest"]
    targets: tuple[str, ...] = ()


class RunCommandAction(_ActionBase):
    kind: Literal["run_command"]
    args: tuple[str, ...]


class RequestApprovalAction(_ActionBase):
    kind: Literal["request_approval"]
    reason: str


class ProposeMemoryAction(_ActionBase):
    kind: Literal["propose_memory"]
    text: str = Field(min_length=1, max_length=MEMORY_PROPOSAL_TEXT_MAX_LENGTH)


class FinishAction(_ActionBase):
    kind: Literal["finish"]
    status: Literal["completed", "blocked"]


Action: TypeAlias = Annotated[
    ListFilesAction
    | ReadFileAction
    | ApplyPatchAction
    | DeletePathAction
    | RunPytestAction
    | RunCommandAction
    | RequestApprovalAction
    | ProposeMemoryAction
    | FinishAction,
    Field(discriminator="kind"),
]

_ACTION_ADAPTER = TypeAdapter(Action)


def parse_action(payload: str) -> Action:
    """Parse one model JSON payload into one of the supported actions."""
    return _ACTION_ADAPTER.validate_json(payload)


def stable_hash(action: Action) -> str:
    """Hash a canonical JSON representation of an action."""
    canonical = json.dumps(
        action.model_dump(mode="json"),
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return sha256(canonical.encode()).hexdigest()
