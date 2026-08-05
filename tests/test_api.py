"""Offline ASGI contracts for the loopback-safe JSON control adapter."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

import httpx
import pytest

from guardedpy.actions import RunCommandAction
from guardedpy.command_rules import CommandRuleStore
from guardedpy.credentials import CredentialStatus
from guardedpy.domain import PolicyVerdict, TaskStatus
from guardedpy.events import EventStore, RunEvent
from guardedpy.llm import ScriptedLLM
from guardedpy.orchestrator import TaskOrchestrator


async def _send(app: Any, method: str, path: str, **kwargs: Any) -> httpx.Response:
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        return await client.request(method, path, **kwargs)


def _request(app: Any, method: str, path: str, **kwargs: Any) -> httpx.Response:
    return asyncio.run(_send(app, method, path, **kwargs))


@dataclass
class FakeCredentials:
    configured: bool = False

    def status(self) -> CredentialStatus:
        return CredentialStatus(configured=self.configured)

    def set_key(self, key: str) -> None:
        del key
        self.configured = True

    def clear_key(self) -> None:
        self.configured = False


class ImmediateThread:
    """Run the bounded task loop synchronously through the old HTML seam."""

    def __init__(self, *, target: Any, args: tuple[Any, ...], daemon: bool) -> None:
        self.target = target
        self.args = args
        self.daemon = daemon

    def start(self) -> None:
        self.target(*self.args)


def _project_root(tmp_path: Path) -> Path:
    root = tmp_path / "project"
    (root / "src").mkdir(parents=True)
    (root / "tests").mkdir()
    return root


def _setup_data(root: Path) -> dict[str, str]:
    return {
        "project_root": str(root),
        "source_dirs": "src",
        "test_dirs": "tests",
        "pytest_command": "pytest",
        "model": "deepseek-chat",
        "timeout_seconds": "30",
        "api_key": "test-key",
    }


def _waiting_app(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Any, UUID]:
    """Build a real runtime task paused on a non-permanent approval decision."""
    import guardedpy.web as web

    root = _project_root(tmp_path)
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    monkeypatch.setattr(web, "Thread", ImmediateThread)

    def factory(project_root: Path, config: Any, memory: Any) -> TaskOrchestrator:
        return TaskOrchestrator(
            project_root,
            ScriptedLLM(['{"kind":"delete_path","summary":"PRIVATE-PATCH","path":"obsolete.txt"}']),
            memory_store=memory,
        )

    app = web.create_app("local", web.WebServices(FakeCredentials(), factory))
    assert _request(app, "POST", "/setup", data=_setup_data(root)).status_code == 303
    created = _request(
        app,
        "POST",
        "/tasks",
        data={
            "mode": "bugfix",
            "description": "Remove obsolete file",
            "bugfix_target": "tests/test_value.py::test_value_is_fixed",
        },
    )
    assert created.status_code == 303
    task_id = UUID(created.headers["location"].rsplit("/", 1)[-1])
    EventStore(root).append(
        RunEvent(
            task_id=task_id,
            task_status=TaskStatus.WAITING_APPROVAL,
            policy_verdict=PolicyVerdict.APPROVAL_REQUIRED,
            action_projection="删除项目内文件",
            policy_rule_id="delete.approval_required",
            policy_reason="需要审批。",
        )
    )
    return app, task_id


def _api_app(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Any, Path]:
    """Build an unconfigured real runtime for setup and management controls."""
    import guardedpy.web as web

    root = _project_root(tmp_path)
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    app = web.create_app(
        "local",
        web.WebServices(
            FakeCredentials(),
            lambda project_root, config, memory: (_ for _ in ()).throw(
                AssertionError("management endpoints must not start an orchestrator")
            ),
        ),
    )
    return app, root


def test_api_feedback_and_approval_expose_only_safe_event_fields(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Catches an API adapter returning an event's raw patch or credential fields."""
    app, waiting_id = _waiting_app(tmp_path, monkeypatch)

    response = _request(app, "GET", f"/api/v1/tasks/{waiting_id}/events")

    assert response.status_code == 200
    event = response.json()[-1]
    assert event["action_projection"] == "删除项目内文件"
    assert "PRIVATE-PATCH" not in json.dumps(event, ensure_ascii=False)
    assert "api_key" not in event


def test_api_rejects_remote_host_configuration_and_stale_approval(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Catches remote-listen input or a non-permanent approval being accepted."""
    app, waiting_id = _waiting_app(tmp_path, monkeypatch)

    remote = _request(
        app,
        "PUT",
        "/api/v1/setup",
        json={
            "project_root": str(tmp_path / "project"),
            "source_dirs": ["src"],
            "test_dirs": ["tests"],
            "pytest_command": ["pytest"],
            "model": "deepseek-chat",
            "timeout_seconds": 30,
            "host": "0.0.0.0",
        },
    )
    response = _request(
        app, "POST", f"/api/v1/tasks/{waiting_id}/approval", json={"decision": "always"}
    )

    assert remote.status_code == 422
    assert remote.json() == {"detail": "请求参数无效。"}
    assert response.status_code == 409
    assert response.json() == {"detail": "审批请求已失效。"}


def test_api_returns_the_fixed_validation_error_for_invalid_setup_values(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Catches malformed persisted setup escaping as a server error."""
    app, _ = _waiting_app(tmp_path, monkeypatch)

    response = _request(
        app,
        "PUT",
        "/api/v1/setup",
        json={
            "project_root": str(tmp_path / "project"),
            "source_dirs": ["src"],
            "test_dirs": ["tests"],
            "pytest_command": ["pytest"],
            "model": "  ",
            "timeout_seconds": 30,
        },
    )

    assert response.status_code == 422
    assert response.json() == {"detail": "请求参数无效。"}


def test_api_uses_the_runtime_for_nonsecret_management_controls(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Catches API management routes using parallel state or returning a credential."""
    app, root = _api_app(tmp_path, monkeypatch)

    setup = _request(
        app,
        "PUT",
        "/api/v1/setup",
        json={
            "project_root": str(root),
            "source_dirs": ["src"],
            "test_dirs": ["tests"],
            "pytest_command": ["pytest"],
            "model": "deepseek-chat",
            "timeout_seconds": 30,
        },
    )
    initial_credentials = _request(app, "GET", "/api/v1/credentials")
    stored = _request(app, "PUT", "/api/v1/credentials", json={"api_key": "test-key"})
    configured_credentials = _request(app, "GET", "/api/v1/credentials")
    runtime = app.state.runtime
    proposal = runtime._memory_store.propose(uuid4(), "Remember focused tests")
    listed_memories = _request(app, "GET", "/api/v1/memories")
    approved = _request(app, "POST", f"/api/v1/memories/{proposal.id}/approve")
    deleted_memory = _request(app, "DELETE", f"/api/v1/memories/{proposal.id}")
    rule = CommandRuleStore(root).add_from(
        RunCommandAction(
            kind="run_command",
            summary="check whitespace",
            args=("git", "diff", "--no-ext-diff", "--check"),
        ),
        current_branch=None,
    )
    listed_rules = _request(app, "GET", "/api/v1/command-rules")
    deleted_rule = _request(app, "DELETE", f"/api/v1/command-rules/{rule.id}")
    cleared = _request(app, "DELETE", "/api/v1/credentials")
    missing_task = _request(app, "GET", f"/api/v1/tasks/{uuid4()}")
    missing_api_route = _request(app, "GET", "/api/v1/not-a-route")

    assert initial_credentials.json() == {"configured": False}
    assert stored.status_code == 204
    assert setup.status_code == 200
    assert setup.json()["credential_configured"] is False
    assert "api_key" not in setup.json()
    assert configured_credentials.json() == {"configured": True}
    assert listed_memories.json()["proposals"] == [
        {"id": str(proposal.id), "task_id": str(proposal.task_id), "text": proposal.text, "approved_at": None}
    ]
    assert approved.status_code == 200
    assert deleted_memory.status_code == 204
    assert listed_rules.json() == [{"id": rule.id, "kind": "git_diff_check", "branch": None, "package_specs": []}]
    assert deleted_rule.status_code == 204
    assert cleared.status_code == 204
    assert missing_task.status_code == 404
    assert missing_task.json() == {"detail": "未找到任务。"}
    assert missing_api_route.status_code == 404
    assert missing_api_route.json() == {"detail": "未找到资源。"}
