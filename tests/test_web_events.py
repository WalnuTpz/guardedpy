"""Offline ASGI coverage for the safe task timeline and memory controls."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

import httpx
import pytest

from guardedpy.actions import RunCommandAction, parse_action
from guardedpy.command_rules import CommandRuleStore
from guardedpy.credentials import CredentialStatus
from guardedpy.domain import PolicyVerdict, TaskStatus
from guardedpy.events import EventStore, FeedbackAudit, RunEvent, StopReason
from guardedpy.llm import ScriptedLLM
from guardedpy.orchestrator import TaskOrchestrator
from guardedpy.workspace import ToolResult


async def _request(app: Any, method: str, path: str, **kwargs: Any) -> httpx.Response:
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        return await client.request(method, path, **kwargs)


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
    """A deterministic daemon-thread seam that runs the supplied loop inline."""

    started: list["ImmediateThread"] = []

    def __init__(self, *, target: Any, args: tuple[Any, ...], daemon: bool) -> None:
        self.target = target
        self.args = args
        self.daemon = daemon

    def start(self) -> None:
        self.started.append(self)
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


def _action(**payload: object) -> str:
    return json.dumps(payload)


def _waiting_app(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, responses: list[str]
) -> tuple[Any, Path, list[TaskOrchestrator]]:
    """Create an app whose real task loop pauses on a scripted unsafe action."""
    import guardedpy.web as web

    root = _project_root(tmp_path)
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    ImmediateThread.started = []
    monkeypatch.setattr(web, "Thread", ImmediateThread)
    orchestrators: list[TaskOrchestrator] = []

    def factory(project_root: Path, config: Any, memory: Any) -> TaskOrchestrator:
        orchestrator = TaskOrchestrator(project_root, ScriptedLLM(responses), memory_store=memory)
        orchestrators.append(orchestrator)
        return orchestrator

    app = web.create_app("local", web.WebServices(FakeCredentials(), factory))
    assert asyncio.run(_request(app, "POST", "/setup", data=_setup_data(root))).status_code == 303
    created = asyncio.run(
        _request(
            app,
            "POST",
            "/tasks",
            data={
                "mode": "bugfix",
                "description": "Remove obsolete file",
                "bugfix_target": "tests/test_value.py::test_value_is_fixed",
            },
        )
    )
    assert created.status_code == 303
    assert app.state.local.task is not None
    assert app.state.local.task.status is TaskStatus.WAITING_APPROVAL
    return app, root, orchestrators


def _waiting_hash(root: Path, task_id: UUID) -> str:
    action_hash = EventStore(root).events_for(task_id)[-1].action_hash
    assert action_hash is not None
    return action_hash


def test_task_detail_and_events_expose_only_stored_audit_projection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Catches timeline pages or polling endpoints rendering raw diffs or context."""
    delete = _action(kind="delete_path", summary="remove obsolete file", path="obsolete.txt")
    app, root, _ = _waiting_app(tmp_path, monkeypatch, [delete])
    task = app.state.local.task
    assert task is not None
    EventStore(root).append(
        RunEvent(
            task_id=task.id,
            task_status=TaskStatus.WAITING_APPROVAL,
            action=parse_action(
                _action(
                    kind="apply_patch",
                    summary="replace secret",
                    diff="--- a/secret.py\n+++ b/secret.py\n@@ -1 +1 @@\n-old\n+new\n",
                )
            ),
            policy_verdict=PolicyVerdict.APPROVAL_REQUIRED,
            feedback=FeedbackAudit(kind="passed", node_id="tests/test_secret.py::test_safe"),
            stop_reason=StopReason.ROUND_LIMIT,
        )
    )

    page = asyncio.run(_request(app, "GET", f"/tasks/{task.id}"))
    events = asyncio.run(_request(app, "GET", f"/tasks/{task.id}/events"))
    script = (Path(__file__).parents[1] / "src" / "guardedpy" / "static" / "app.js").read_text()

    assert page.status_code == 200
    assert "apply source patch" in page.text
    assert "approval_required" in page.text
    assert "pytest passed" in page.text
    assert "round_limit" in page.text
    assert "app.js" in page.text
    assert "--- a/secret.py" not in page.text
    assert events.status_code == 200
    payload = events.json()
    assert payload
    assert all(set(event) == {
        "task_id", "task_status", "action_summary", "action_hash", "policy_verdict",
        "approval_granted", "feedback_kind", "feedback_excerpt", "retry_count",
        "action_projection", "affected_project", "policy_rule_id", "policy_reason",
        "stop_reason", "id", "created_at",
    } for event in payload)
    assert "--- a/secret.py" not in events.text
    assert "clearInterval(interval)" in script
    assert "terminal" in script


def test_approval_requires_the_exact_waiting_hash_and_starts_one_continuation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Catches hash substitution, replay, or more than one resumed background loop."""
    target = tmp_path / "project" / "obsolete.txt"
    target.parent.mkdir(parents=True)
    target.write_text("remove only when approved\n")
    delete = _action(kind="delete_path", summary="remove obsolete file", path="obsolete.txt")
    finish = _action(kind="finish", summary="stop", status="blocked")
    app, root, _ = _waiting_app(tmp_path, monkeypatch, [delete, finish])
    task = app.state.local.task
    assert task is not None
    action_hash = _waiting_hash(root, task.id)

    wrong = asyncio.run(
        _request(app, "POST", f"/tasks/{task.id}/approval", data={"action_hash": "wrong", "decision": "once"})
    )
    assert wrong.status_code == 409
    assert target.exists()
    accepted = asyncio.run(
        _request(app, "POST", f"/tasks/{task.id}/approval", data={"action_hash": action_hash, "decision": "once"})
    )
    replay = asyncio.run(
        _request(app, "POST", f"/tasks/{task.id}/approval", data={"action_hash": action_hash, "decision": "once"})
    )

    assert accepted.status_code == 303
    assert accepted.headers["location"] == f"/tasks/{task.id}"
    assert target.exists() is False
    assert len(ImmediateThread.started) == 2
    assert replay.status_code == 409


def test_rejected_approval_blocks_without_starting_a_background_loop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Catches a rejected action continuing the model loop or accepting a stale repeat."""
    delete = _action(kind="delete_path", summary="remove obsolete file", path="obsolete.txt")
    app, root, _ = _waiting_app(tmp_path, monkeypatch, [delete])
    task = app.state.local.task
    assert task is not None
    action_hash = _waiting_hash(root, task.id)

    rejected = asyncio.run(
        _request(app, "POST", f"/tasks/{task.id}/approval", data={"action_hash": action_hash, "decision": "reject"})
    )
    stale = asyncio.run(
        _request(app, "POST", f"/tasks/{task.id}/approval", data={"action_hash": action_hash, "decision": "reject"})
    )

    assert rejected.status_code == 303
    assert rejected.headers["location"] == f"/tasks/{task.id}"
    assert task.status is TaskStatus.BLOCKED
    assert len(ImmediateThread.started) == 1
    assert stale.status_code == 409


def test_approval_page_shows_safe_rule_reason_and_three_decisions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Catches approval controls omitting permanent scope or exposing raw action details."""
    command = _action(
        kind="run_command",
        summary="install package with hidden context",
        args=["python", "-m", "pip", "install", "example-package==1.2.3"],
    )
    app, _, _ = _waiting_app(tmp_path, monkeypatch, [command])
    task = app.state.local.task
    assert task is not None

    page = asyncio.run(_request(app, "GET", f"/tasks/{task.id}"))

    assert page.status_code == 200
    assert "Policy rule: command.approval_required" in page.text
    assert "Policy reason: the constrained command requires approval" in page.text
    assert 'value="reject"' in page.text
    assert 'value="once"' in page.text
    assert 'value="always"' in page.text
    assert "Allow once" in page.text
    assert "Always allow this rule" in page.text
    assert "hidden context" not in page.text


@pytest.mark.parametrize(
    ("pending_action", "expected_projection", "unsafe_text"),
    [
        pytest.param(
            _action(
                kind="run_command",
                summary="MODEL-SUMMARY-PIP",
                args=["python", "-m", "pip", "install", "example-package==1.2.3"],
            ),
            "Command: python -m pip install example-package==1.2.3",
            "MODEL-SUMMARY-PIP",
            id="pip",
        ),
        pytest.param(
            _action(
                kind="run_command",
                summary="MODEL-SUMMARY-GIT",
                args=["git", "diff", "--no-ext-diff", "--check"],
            ),
            "Command: git diff --no-ext-diff --check",
            "MODEL-SUMMARY-GIT",
            id="git",
        ),
        pytest.param(
            _action(
                kind="delete_path",
                summary="MODEL-SUMMARY-PATH",
                path="obsolete.txt",
            ),
            "Path: obsolete.txt",
            "MODEL-SUMMARY-PATH",
            id="path",
        ),
        pytest.param(
            _action(
                kind="apply_patch",
                summary="MODEL-SUMMARY-PATCH",
                diff=(
                    "--- a/README.md\n+++ b/README.md\n@@ -1 +1 @@\n"
                    "-old\n+RAW-PATCH-SECRET\n"
                ),
            ),
            "Paths: README.md",
            "RAW-PATCH-SECRET",
            id="patch-path",
        ),
    ],
)
def test_approval_page_projects_only_validated_command_or_path_and_affected_project(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    pending_action: str,
    expected_projection: str,
    unsafe_text: str,
) -> None:
    """Catches an approval card hiding decision inputs or rendering model-controlled text."""
    app, root, _ = _waiting_app(tmp_path, monkeypatch, [pending_action])
    task = app.state.local.task
    assert task is not None

    page = asyncio.run(_request(app, "GET", f"/tasks/{task.id}"))
    feed = asyncio.run(_request(app, "GET", f"/tasks/{task.id}/events"))

    assert expected_projection in page.text
    assert f"Project: {root.resolve()}" in page.text
    assert unsafe_text not in page.text
    assert unsafe_text not in feed.text
    waiting = feed.json()[-1]
    assert waiting["action_projection"] == expected_projection
    assert waiting["affected_project"] == str(root.resolve())


@pytest.mark.parametrize(
    ("decision", "expected_rule", "expected_reason", "granted", "approval_text"),
    [
        pytest.param(
            "once",
            "approval.granted",
            "user approved this exact action once",
            True,
            "approval: granted",
            id="once",
        ),
        pytest.param(
            "always",
            "approval.granted_always",
            "user approved a constrained persistent command rule",
            True,
            "approval: granted",
            id="always",
        ),
        pytest.param(
            "reject",
            "approval.declined",
            "user declined the action",
            False,
            "approval: rejected",
            id="reject",
        ),
    ],
)
def test_waiting_and_resolved_approval_timeline_persists_actual_decision_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    decision: str,
    expected_rule: str,
    expected_reason: str,
    granted: bool,
    approval_text: str,
) -> None:
    """Catches inferred or dropped policy metadata after once/always/reject resolution."""
    command = _action(
        kind="run_command",
        summary="MODEL-SUMMARY-MUST-STAY-HIDDEN",
        args=["git", "diff", "--no-ext-diff", "--check"],
    )
    finish = _action(kind="finish", summary="stop", status="blocked")
    monkeypatch.setattr(
        TaskOrchestrator,
        "_run_command",
        lambda _self, _action: ToolResult(True, "simulated command", {}),
    )
    app, root, _ = _waiting_app(tmp_path, monkeypatch, [command, finish])
    task = app.state.local.task
    assert task is not None
    action_hash = _waiting_hash(root, task.id)

    resolved_response = asyncio.run(
        _request(
            app,
            "POST",
            f"/tasks/{task.id}/approval",
            data={"action_hash": action_hash, "decision": decision},
        )
    )
    feed = asyncio.run(_request(app, "GET", f"/tasks/{task.id}/events"))
    page = asyncio.run(_request(app, "GET", f"/tasks/{task.id}"))

    assert resolved_response.status_code == 303
    events = feed.json()
    waiting = next(event for event in events if event["task_status"] == "waiting_approval")
    resolved = next(event for event in events if event["approval_granted"] is granted)
    assert waiting["policy_rule_id"] == "command.read_only_approval_required"
    assert waiting["policy_reason"] == "the read-only Git whitespace check requires approval"
    assert resolved["policy_rule_id"] == expected_rule
    assert resolved["policy_reason"] == expected_reason
    assert resolved["action_projection"] == "Command: git diff --no-ext-diff --check"
    assert resolved["affected_project"] == str(root.resolve())
    assert approval_text in page.text
    assert expected_rule in page.text
    assert expected_reason in page.text
    assert "MODEL-SUMMARY-MUST-STAY-HIDDEN" not in page.text + feed.text

    script = (Path(__file__).parents[1] / "src" / "guardedpy" / "static" / "app.js").read_text()
    assert "approval_granted" in script
    assert "action_projection" in script
    assert "affected_project" in script
    assert "policy_rule_id" in script
    assert "policy_reason" in script


def test_terminal_task_page_does_not_load_polling_script(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Catches a terminal task page starting a polling timer after render."""
    delete = _action(kind="delete_path", summary="remove obsolete file", path="obsolete.txt")
    app, _, _ = _waiting_app(tmp_path, monkeypatch, [delete])
    task = app.state.local.task
    assert task is not None
    task.status = TaskStatus.CANCELLED

    page = asyncio.run(_request(app, "GET", f"/tasks/{task.id}"))

    assert page.status_code == 200
    assert "/static/app.js" not in page.text


def test_command_rules_are_project_scoped_listable_and_revocable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Catches a permanent rule UI that leaks project hashes or cannot revoke a rule."""
    delete = _action(kind="delete_path", summary="remove obsolete file", path="obsolete.txt")
    app, root, _ = _waiting_app(tmp_path, monkeypatch, [delete])
    rule = CommandRuleStore(root).add_from(
        RunCommandAction(
            kind="run_command",
            summary="not rendered",
            args=("python", "-m", "pip", "install", "example-package==1.2.3"),
        ),
        None,
    )

    page = asyncio.run(_request(app, "GET", "/settings/command-rules"))
    revoked = asyncio.run(
        _request(app, "POST", f"/settings/command-rules/{rule.id}/delete")
    )
    missing = asyncio.run(
        _request(app, "POST", f"/settings/command-rules/{rule.id}/delete")
    )

    assert page.status_code == 200
    assert "Python package install" in page.text
    assert "example-package==1.2.3" in page.text
    assert rule.project_hash not in page.text
    assert "not rendered" not in page.text
    assert revoked.status_code == 303
    assert CommandRuleStore(root).list_rules() == []
    assert missing.status_code == 404


def test_terminal_task_keeps_its_original_event_root_after_reconfiguration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Catches old task detail reading the newly configured project's event database."""
    delete = _action(kind="delete_path", summary="remove obsolete file", path="obsolete.txt")
    app, original_root, _ = _waiting_app(tmp_path, monkeypatch, [delete])
    task = app.state.local.task
    assert task is not None
    action_hash = _waiting_hash(original_root, task.id)
    assert asyncio.run(
        _request(
            app,
            "POST",
            f"/tasks/{task.id}/approval",
            data={"action_hash": action_hash, "decision": "reject"},
        )
    ).status_code == 303

    replacement_root = _project_root(tmp_path / "replacement")
    EventStore(replacement_root).append(
        RunEvent(task_id=task.id, task_status=TaskStatus.COMPLETED)
    )
    replaced = asyncio.run(
        _request(app, "POST", "/setup", data=_setup_data(replacement_root))
    )
    detail = asyncio.run(_request(app, "GET", f"/tasks/{task.id}"))
    feed = asyncio.run(_request(app, "GET", f"/tasks/{task.id}/events"))

    assert replaced.status_code == 303
    assert detail.status_code == 200
    assert f"Project: {original_root.resolve()}" in detail.text
    assert "delete.approval_required" in detail.text
    assert feed.status_code == 200
    events = feed.json()
    assert events
    assert all(event["affected_project"] == str(original_root.resolve()) for event in events)
    assert all(event["task_status"] != "completed" for event in events)


def test_memory_controls_keep_proposals_pending_until_approval_and_404_unknown_ids(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Catches memory persistence before consent or controls accepting an unknown UUID."""
    delete = _action(kind="delete_path", summary="remove obsolete file", path="obsolete.txt")
    app, _, _ = _waiting_app(tmp_path, monkeypatch, [delete])
    memory_store = app.state.local.memory_store
    assert memory_store is not None
    proposal = memory_store.propose(uuid4(), "Remember the parser naming convention")

    queue = asyncio.run(_request(app, "GET", "/memories"))
    approved = asyncio.run(_request(app, "POST", f"/memories/{proposal.id}/approve"))
    assert approved.status_code == 303
    assert memory_store.search("parser")
    deleted = asyncio.run(_request(app, "POST", f"/memories/{proposal.id}/delete"))
    unknown = asyncio.run(_request(app, "POST", f"/memories/{uuid4()}/delete"))
    unknown_approve = asyncio.run(_request(app, "POST", f"/memories/{uuid4()}/approve"))

    assert queue.status_code == 200
    assert proposal.text in queue.text
    assert deleted.status_code == 303
    assert memory_store.search("parser") == []
    assert unknown.status_code == 404
    assert unknown_approve.status_code == 404


def test_unknown_task_resources_return_not_found(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Catches task detail, event, or approval routes inventing missing task state."""
    delete = _action(kind="delete_path", summary="remove obsolete file", path="obsolete.txt")
    app, _, _ = _waiting_app(tmp_path, monkeypatch, [delete])
    unknown = uuid4()

    detail = asyncio.run(_request(app, "GET", f"/tasks/{unknown}"))
    events = asyncio.run(_request(app, "GET", f"/tasks/{unknown}/events"))
    approval = asyncio.run(_request(app, "POST", f"/tasks/{unknown}/approval", data={"action_hash": "x", "decision": "once"}))

    assert detail.status_code == 404
    assert events.status_code == 404
    assert approval.status_code == 404
