"""Versioned, loopback-composed JSON controls over :class:`LocalRuntime`."""

from __future__ import annotations

from pathlib import Path
from threading import Thread
from typing import Literal
from uuid import UUID

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from guardedpy.config import HarnessConfig
from guardedpy.credentials import CredentialBackendUnavailableError
from guardedpy.domain import ApprovalDecision, TaskMode, TaskStatus
from guardedpy.runtime import (
    LocalRuntime,
    RuntimeBusyError,
    RuntimeCommandRuleNotFoundError,
    RuntimeMemoryNotFoundError,
    RuntimeNotConfiguredError,
    RuntimeTaskNotFoundError,
)


_INVALID_REQUEST = "请求参数无效。"
_TASK_NOT_FOUND = "未找到任务。"
_MEMORY_NOT_FOUND = "未找到记忆。"
_COMMAND_RULE_NOT_FOUND = "未找到命令规则。"
_APPROVAL_STALE = "审批请求已失效。"
_CREDENTIAL_UNAVAILABLE = "凭据存储不可用。"


class SetupUpdate(BaseModel):
    """The non-secret, local project setup persisted by the runtime."""

    model_config = ConfigDict(extra="forbid")

    project_root: str
    source_dirs: tuple[str, ...] = Field(min_length=1)
    test_dirs: tuple[str, ...] = Field(min_length=1)
    pytest_command: tuple[str, ...] = Field(min_length=1)
    model: str = "deepseek-chat"
    timeout_seconds: int = Field(default=30, ge=5, le=120)


class TaskCreate(BaseModel):
    """A bounded task request without model context or tool payloads."""

    model_config = ConfigDict(extra="forbid")

    description: str
    mode: TaskMode
    bugfix_target: str | None = None


class ApprovalResolution(BaseModel):
    """A decision bound by the runtime's persisted safe action hash."""

    model_config = ConfigDict(extra="forbid")

    action_hash: str | None = None
    decision: Literal["reject", "once", "always"]


class CredentialUpdate(BaseModel):
    """Write-only credential input; no model reads a configured key."""

    model_config = ConfigDict(extra="forbid")

    api_key: str = Field(min_length=1)


def create_api_router(runtime: LocalRuntime) -> APIRouter:
    """Build only the versioned local-control routes backed by one runtime."""
    router = APIRouter(prefix="/api/v1")

    @router.get("/setup")
    async def setup_status() -> dict[str, object]:
        try:
            credential = runtime.credential_status()
        except CredentialBackendUnavailableError as error:
            raise HTTPException(status_code=503, detail=_CREDENTIAL_UNAVAILABLE) from error
        config = runtime.config
        root = runtime.project_root
        return {
            "configured": config is not None and root is not None,
            "credential_configured": credential.configured,
            "project_root": str(root) if root is not None else None,
            "source_dirs": [str(path) for path in config.source_dirs] if config else [],
            "test_dirs": [str(path) for path in config.test_dirs] if config else [],
            "pytest_command": list(config.pytest_command) if config else [],
            "model": config.model if config else None,
            "timeout_seconds": config.timeout_seconds if config else None,
        }

    @router.put("/setup")
    async def update_setup(body: SetupUpdate) -> dict[str, object]:
        root = Path(body.project_root).expanduser()
        if not root.is_dir():
            raise HTTPException(status_code=422, detail=_INVALID_REQUEST)
        root = root.resolve()
        try:
            config = HarnessConfig(
                source_dirs=body.source_dirs,
                test_dirs=body.test_dirs,
                pytest_command=body.pytest_command,
                model=body.model,
                timeout_seconds=body.timeout_seconds,
            )
        except ValidationError:
            raise HTTPException(status_code=422, detail=_INVALID_REQUEST) from None
        if any(not (root / path).is_dir() for path in config.source_dirs + config.test_dirs):
            raise HTTPException(status_code=422, detail=_INVALID_REQUEST)
        try:
            runtime.setup(root, config, api_key=None)
        except RuntimeBusyError as error:
            raise HTTPException(status_code=409, detail=str(error)) from error
        except CredentialBackendUnavailableError as error:
            raise HTTPException(status_code=503, detail=_CREDENTIAL_UNAVAILABLE) from error
        except (OSError, ValueError):
            raise HTTPException(status_code=422, detail=_INVALID_REQUEST) from None
        return await setup_status()

    @router.post("/tasks", status_code=201)
    async def create_task(body: TaskCreate) -> dict[str, object]:
        try:
            task = runtime.create_task(body.description, body.mode, body.bugfix_target)
        except RuntimeBusyError as error:
            raise HTTPException(status_code=409, detail=str(error)) from error
        except RuntimeNotConfiguredError as error:
            raise HTTPException(status_code=409, detail=str(error)) from error
        except ValueError:
            raise HTTPException(status_code=422, detail=_INVALID_REQUEST) from None
        Thread(target=runtime.run, args=(task.id,), daemon=True).start()
        return task.model_dump(mode="json")

    @router.get("/tasks/{task_id}")
    async def read_task(task_id: UUID) -> dict[str, object]:
        try:
            return runtime.task(task_id).model_dump(mode="json")
        except RuntimeTaskNotFoundError as error:
            raise HTTPException(status_code=404, detail=_TASK_NOT_FOUND) from error

    @router.get("/tasks/{task_id}/events")
    async def task_events(task_id: UUID) -> list[dict[str, object]]:
        try:
            events = runtime.events(task_id)
        except RuntimeTaskNotFoundError as error:
            raise HTTPException(status_code=404, detail=_TASK_NOT_FOUND) from error
        return [event.model_dump(mode="json") for event in events]

    @router.post("/tasks/{task_id}/cancel")
    async def cancel_task(task_id: UUID) -> dict[str, object]:
        try:
            return runtime.cancel(task_id).model_dump(mode="json")
        except RuntimeBusyError as error:
            raise HTTPException(status_code=409, detail=str(error)) from error
        except RuntimeTaskNotFoundError as error:
            raise HTTPException(status_code=404, detail=_TASK_NOT_FOUND) from error

    @router.post("/tasks/{task_id}/approval")
    async def resolve_approval(task_id: UUID, body: ApprovalResolution) -> dict[str, object]:
        try:
            task = runtime.task(task_id)
        except RuntimeTaskNotFoundError as error:
            raise HTTPException(status_code=404, detail=_TASK_NOT_FOUND) from error
        if task.status is not TaskStatus.WAITING_APPROVAL or not body.action_hash:
            raise HTTPException(status_code=409, detail=_APPROVAL_STALE)
        try:
            accepted = runtime.resolve_approval(task_id, body.action_hash, body.decision)
        except RuntimeBusyError as error:
            raise HTTPException(status_code=409, detail=str(error)) from error
        except RuntimeTaskNotFoundError as error:
            raise HTTPException(status_code=404, detail=_TASK_NOT_FOUND) from error
        if not accepted:
            if runtime.task(task_id).status is TaskStatus.BLOCKED:
                return runtime.task(task_id).model_dump(mode="json")
            raise HTTPException(status_code=409, detail=_APPROVAL_STALE)
        Thread(target=runtime.run, args=(task_id,), daemon=True).start()
        return runtime.task(task_id).model_dump(mode="json")

    @router.get("/memories")
    async def list_memories() -> dict[str, list[dict[str, object]]]:
        try:
            proposals = runtime.memory_proposals()
            approved = runtime.memories()
        except RuntimeNotConfiguredError as error:
            raise HTTPException(status_code=409, detail=str(error)) from error
        return {
            "proposals": [_memory_payload(entry) for entry in proposals],
            "approved": [_memory_payload(entry) for entry in approved],
        }

    @router.post("/memories/{memory_id}/approve")
    async def approve_memory(memory_id: UUID) -> dict[str, object]:
        try:
            return _memory_payload(runtime.approve_memory(memory_id))
        except RuntimeBusyError as error:
            raise HTTPException(status_code=409, detail=str(error)) from error
        except RuntimeNotConfiguredError as error:
            raise HTTPException(status_code=409, detail=str(error)) from error
        except RuntimeMemoryNotFoundError as error:
            raise HTTPException(status_code=404, detail=_MEMORY_NOT_FOUND) from error

    @router.delete("/memories/{memory_id}", status_code=204)
    async def delete_memory(memory_id: UUID) -> None:
        try:
            runtime.delete_memory(memory_id)
        except RuntimeBusyError as error:
            raise HTTPException(status_code=409, detail=str(error)) from error
        except RuntimeNotConfiguredError as error:
            raise HTTPException(status_code=409, detail=str(error)) from error
        except RuntimeMemoryNotFoundError as error:
            raise HTTPException(status_code=404, detail=_MEMORY_NOT_FOUND) from error

    @router.get("/command-rules")
    async def list_command_rules() -> list[dict[str, object]]:
        try:
            return [_command_rule_payload(rule) for rule in runtime.command_rules()]
        except RuntimeNotConfiguredError as error:
            raise HTTPException(status_code=409, detail=str(error)) from error

    @router.delete("/command-rules/{rule_id}", status_code=204)
    async def delete_command_rule(rule_id: str) -> None:
        try:
            runtime.delete_command_rule(rule_id)
        except RuntimeBusyError as error:
            raise HTTPException(status_code=409, detail=str(error)) from error
        except RuntimeNotConfiguredError as error:
            raise HTTPException(status_code=409, detail=str(error)) from error
        except RuntimeCommandRuleNotFoundError as error:
            raise HTTPException(status_code=404, detail=_COMMAND_RULE_NOT_FOUND) from error

    @router.get("/credentials")
    async def credential_status() -> dict[str, bool]:
        try:
            return {"configured": runtime.credential_status().configured}
        except CredentialBackendUnavailableError as error:
            raise HTTPException(status_code=503, detail=_CREDENTIAL_UNAVAILABLE) from error

    @router.put("/credentials", status_code=204)
    async def update_credential(body: CredentialUpdate) -> None:
        if not body.api_key.strip():
            raise HTTPException(status_code=422, detail=_INVALID_REQUEST)
        try:
            runtime.update_credential(body.api_key)
        except RuntimeBusyError as error:
            raise HTTPException(status_code=409, detail=str(error)) from error
        except RuntimeNotConfiguredError as error:
            raise HTTPException(status_code=409, detail=str(error)) from error
        except CredentialBackendUnavailableError as error:
            raise HTTPException(status_code=503, detail=_CREDENTIAL_UNAVAILABLE) from error

    @router.delete("/credentials", status_code=204)
    async def clear_credential() -> None:
        try:
            runtime.clear_credential()
        except RuntimeBusyError as error:
            raise HTTPException(status_code=409, detail=str(error)) from error
        except RuntimeNotConfiguredError as error:
            raise HTTPException(status_code=409, detail=str(error)) from error
        except CredentialBackendUnavailableError as error:
            raise HTTPException(status_code=503, detail=_CREDENTIAL_UNAVAILABLE) from error

    return router


def _memory_payload(entry: object) -> dict[str, object]:
    return {
        "id": str(entry.id),  # type: ignore[attr-defined]
        "task_id": str(entry.task_id),  # type: ignore[attr-defined]
        "text": entry.text,  # type: ignore[attr-defined]
        "approved_at": entry.approved_at,  # type: ignore[attr-defined]
    }


def _command_rule_payload(rule: object) -> dict[str, object]:
    return {
        "id": rule.id,  # type: ignore[attr-defined]
        "kind": rule.kind.value,  # type: ignore[attr-defined]
        "branch": rule.branch,  # type: ignore[attr-defined]
        "package_specs": list(rule.package_specs),  # type: ignore[attr-defined]
    }
