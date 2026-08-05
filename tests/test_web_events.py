"""Offline ASGI coverage for the safe task timeline and memory controls."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from html.parser import HTMLParser
import json
from pathlib import Path
import re
from typing import Any
from uuid import UUID, uuid4

import httpx
import pytest
from pydantic import ValidationError

from guardedpy.actions import RunCommandAction, parse_action
from guardedpy.command_rules import CommandRuleStore
from guardedpy.config import local_state_path
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


class _TaskDetailDom(HTMLParser):
    """Record actual parsed ancestor relationships in a rendered task detail page."""

    _VOID_TAGS = frozenset(
        {
            "area", "base", "br", "col", "embed", "hr", "img", "input",
            "link", "meta", "param", "source", "track", "wbr",
        }
    )

    def __init__(self) -> None:
        super().__init__()
        self._next_element_id = 0
        self._ancestors: list[tuple[int, str, dict[str, str]]] = []
        self._text_captures: list[tuple[str, list[str]]] = []
        self._inside_navigation = 0
        self._inside_context_title = False
        self._context_title_fragments: list[str] = []
        self.regions: list[str] = []
        self.decision_values: list[str] = []
        self.raw_status: str | None = None
        self.visible_status: str | None = None
        self.current_navigation_hrefs: list[str] = []
        self.context_title: str | None = None
        self.event_list_polling_root_id: int | None = None
        self.status_polling_root_id: int | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attributes = {name: value or "" for name, value in attrs}
        element_id = self._next_element_id
        self._next_element_id += 1
        if tag == "section" and attributes.get("aria-label"):
            self.regions.append(attributes["aria-label"])
        if tag == "nav":
            self._inside_navigation += 1
        if tag == "a" and self._inside_navigation and attributes.get("aria-current") == "page":
            self.current_navigation_hrefs.append(attributes.get("href", ""))
        if "context-bar-title" in attributes.get("class", "").split():
            self._inside_context_title = True
        if tag == "button" and attributes.get("name") == "decision":
            self.decision_values.append(attributes.get("value", ""))
        if "data-events-url" in attributes:
            self.raw_status = attributes.get("data-current-status")
        if "data-event-list" in attributes:
            self.event_list_polling_root_id = self._polling_root_id()
        if "data-task-status" in attributes:
            self.status_polling_root_id = self._polling_root_id()
            self._text_captures.append((tag, []))
        if tag not in self._VOID_TAGS:
            self._ancestors.append((element_id, tag, attributes))

    def handle_startendtag(self, _tag: str, _attrs: list[tuple[str, str | None]]) -> None:
        return

    def handle_endtag(self, tag: str) -> None:
        if self._text_captures and self._text_captures[-1][0] == tag:
            _, fragments = self._text_captures.pop()
            self.visible_status = "".join(fragments).strip()
        if tag == "span" and self._inside_context_title:
            self.context_title = "".join(self._context_title_fragments).strip()
            self._inside_context_title = False
        if tag == "nav":
            self._inside_navigation -= 1
        for index in range(len(self._ancestors) - 1, -1, -1):
            if self._ancestors[index][1] == tag:
                del self._ancestors[index:]
                return

    def handle_data(self, data: str) -> None:
        for _, fragments in self._text_captures:
            fragments.append(data)
        if self._inside_context_title:
            self._context_title_fragments.append(data)

    def _polling_root_id(self) -> int | None:
        for element_id, _tag, attributes in reversed(self._ancestors):
            if "data-events-url" in attributes:
                return element_id
        return None


@dataclass(frozen=True)
class _PollEffects:
    operations: tuple[str, ...]
    visible_status: str | None = None


class _PollingScriptHarness:
    """Parse the complete polling IIFE and execute its observable control flow."""

    _POLL = re.compile(
        r"""\s*
        const\s+response\s*=\s*await\s+fetch\(timeline\.dataset\.eventsUrl,\s*\{\s*headers:\s*\{\s*Accept:\s*\"application/json\"\s*\}\s*\}\);\s*
        if\s*\(!response\.ok\)\s*\{\s*return;\s*\}\s*
        const\s+events\s*=\s*await\s+response\.json\(\);\s*
        const\s+latest\s*=\s*events\.at\(-1\);\s*
        if\s*\(!latest\)\s*\{\s*return;\s*\}\s*
        if\s*\(terminal\.has\(latest\.task_status\)\)\s*\{\s*clearInterval\(interval\);\s*\}\s*
        if\s*\(timeline\.dataset\.currentStatus\s*!==\s*latest\.task_status\)\s*\{\s*window\.location\.reload\(\);\s*return;\s*\}\s*
        if\s*\(status\)\s*\{\s*status\.textContent\s*=\s*statusLabel\(latest\.task_status\);\s*status\.dataset\.status\s*=\s*latest\.task_status;\s*\}\s*
        if\s*\(list\)\s*\{\s*list\.replaceChildren\(\.\.\.events\.map\(renderEvent\)\);\s*\}\s*
        """,
        re.VERBOSE | re.DOTALL,
    )

    def __init__(self, source: str) -> None:
        self._validate_balanced_source(source)
        stripped = source.strip()
        assert stripped.startswith("(() => {") and stripped.endswith("})();")
        assert source.count("const poll = async () =>") == 1
        assert source.count("const interval = window.setInterval(poll, 2000);") == 1
        assert source.count("poll();") == 1
        assert re.search(
            r'const timeline = document\.querySelector\("\[data-events-url\]"\);\s*'
            r'if \(!timeline \|\| timeline\.dataset\.terminal === "true"\) \{\s*return;\s*\}',
            source,
        )
        labels_match = re.search(
            r"const STATUS_LABELS = Object\.freeze\(\{(?P<body>.*?)\}\);", source, re.DOTALL
        )
        assert labels_match is not None
        self.labels = dict(
            re.findall(r'(\w+):\s*"([^"]+)"', labels_match.group("body"))
        )
        assert self.labels == {
            "pending": "待处理",
            "running": "运行中",
            "waiting_approval": "等待审批",
            "completed": "已完成",
            "blocked": "已阻止",
            "cancelled": "已取消",
            "interrupted": "已中断",
        }
        terminal_match = re.search(r"const terminal = new Set\(\[(?P<body>.*?)\]\);", source)
        assert terminal_match is not None
        self.terminal = frozenset(re.findall(r'"([^"]+)"', terminal_match.group("body")))
        assert self.terminal == {"completed", "blocked", "cancelled", "interrupted"}
        detail_span = self._block_after(source, "const detailSpan = (className, text) =>")
        render_event = self._block_after(source, "const renderEvent = (event) =>")
        poll = self._block_after(source, "const poll = async () =>")
        assert "span.textContent = text;" in detail_span
        assert not {"innerHTML", "outerHTML", "insertAdjacentHTML"}.intersection(
            set(re.findall(r"[A-Za-z_$][\w$]*", detail_span + render_event))
        )
        assert set(re.findall(r"event\.(\w+)", render_event)) == {
            "task_status",
            "action_summary",
            "action_projection",
            "affected_project",
            "policy_verdict",
            "policy_rule_id",
            "policy_reason",
            "approval_granted",
            "feedback_excerpt",
            "feedback_node_id",
            "stop_reason",
        }
        assert self._POLL.fullmatch(poll)

    def poll(
        self,
        *,
        current_status: str,
        latest_status: str | None,
        response_ok: bool = True,
    ) -> _PollEffects:
        operations = ["fetch"]
        if not response_ok:
            return _PollEffects(tuple(operations))
        if latest_status is None:
            return _PollEffects(tuple(operations))
        if latest_status in self.terminal:
            operations.append("clearInterval")
        if current_status != latest_status:
            operations.append("reload")
            return _PollEffects(tuple(operations))
        operations.extend(("updateStatus", "replaceChildren"))
        return _PollEffects(tuple(operations), self.labels.get(latest_status, latest_status))

    @staticmethod
    def _validate_balanced_source(source: str) -> None:
        pairs = {"(": ")", "[": "]", "{": "}"}
        stack: list[str] = []
        quote: str | None = None
        escaped = False
        for character in source:
            if quote is not None:
                if escaped:
                    escaped = False
                elif character == "\\":
                    escaped = True
                elif character == quote:
                    quote = None
                continue
            if character in {'"', "'", "`"}:
                quote = character
            elif character in pairs:
                stack.append(pairs[character])
            elif character in pairs.values():
                assert stack and stack.pop() == character
        assert quote is None and not stack

    @staticmethod
    def _block_after(source: str, marker: str) -> str:
        start = source.index(marker)
        opening = source.index("{", start)
        depth = 0
        quote: str | None = None
        escaped = False
        for index in range(opening, len(source)):
            character = source[index]
            if quote is not None:
                if escaped:
                    escaped = False
                elif character == "\\":
                    escaped = True
                elif character == quote:
                    quote = None
                continue
            if character in {'"', "'", "`"}:
                quote = character
            elif character == "{":
                depth += 1
            elif character == "}":
                depth -= 1
                if depth == 0:
                    return source[opening + 1 : index]
        raise AssertionError(f"unterminated block after {marker!r}")


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
        "approval_granted", "permanent_eligible", "feedback_kind", "feedback_excerpt",
        "feedback_node_id", "retry_count",
        "action_projection", "affected_project", "policy_rule_id", "policy_reason",
        "stop_reason", "id", "created_at",
    } for event in payload)
    assert "--- a/secret.py" not in events.text


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


def test_command_approval_page_shows_safe_rule_reason_and_all_decisions(
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
    assert "策略规则" in page.text
    assert "command.approval_required" in page.text
    assert "策略原因" in page.text
    assert "the constrained command requires approval" in page.text
    assert 'value="reject"' in page.text
    assert 'value="once"' in page.text
    assert 'value="always"' in page.text
    assert "仅允许一次" in page.text
    assert "始终允许此规则" in page.text
    assert "hidden context" not in page.text


@pytest.mark.parametrize(
    ("pending_action", "final_decision"),
    [
        pytest.param(
            _action(kind="delete_path", summary="remove file", path="obsolete.txt"),
            "once",
            id="delete-once",
        ),
        pytest.param(
            _action(
                kind="apply_patch",
                summary="edit non-code file",
                diff="--- a/README.md\n+++ b/README.md\n@@ -1 +1 @@\n-old\n+new\n",
            ),
            "reject",
            id="patch-reject",
        ),
        pytest.param(
            _action(kind="request_approval", summary="ask", reason="model-controlled reason"),
            "once",
            id="explicit-request-once",
        ),
    ],
)
def test_non_command_approval_hides_always_and_keeps_pending_after_forgery(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    pending_action: str,
    final_decision: str,
) -> None:
    """Catches UI or routing treating non-command approval as permanently eligible."""
    app, root, _ = _waiting_app(tmp_path, monkeypatch, [pending_action])
    task = app.state.local.task
    assert task is not None
    action_hash = _waiting_hash(root, task.id)

    page = asyncio.run(_request(app, "GET", f"/tasks/{task.id}"))
    forged = asyncio.run(
        _request(
            app,
            "POST",
            f"/tasks/{task.id}/approval",
            data={"action_hash": action_hash, "decision": "always"},
        )
    )
    assert forged.status_code == 409
    assert task.status is TaskStatus.WAITING_APPROVAL
    resolved = asyncio.run(
        _request(
            app,
            "POST",
            f"/tasks/{task.id}/approval",
            data={"action_hash": action_hash, "decision": final_decision},
        )
    )

    assert page.status_code == 200
    assert 'value="always"' not in page.text
    detail = _TaskDetailDom()
    detail.feed(page.text)
    assert "always" not in detail.decision_values
    assert resolved.status_code == 303


def test_task_detail_polling_root_owns_status_and_event_list_in_parsed_dom(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Catches polling queries targeting siblings instead of descendants of their root."""
    delete = _action(kind="delete_path", summary="remove obsolete file", path="obsolete.txt")
    app, _, _ = _waiting_app(tmp_path, monkeypatch, [delete])
    task = app.state.local.task
    assert task is not None

    page = asyncio.run(_request(app, "GET", f"/tasks/{task.id}"))
    document = _TaskDetailDom()
    document.feed(page.text)

    assert page.status_code == 200
    assert document.status_polling_root_id is not None
    assert document.event_list_polling_root_id is not None
    assert document.status_polling_root_id == document.event_list_polling_root_id


def test_task_detail_prioritizes_governance_regions_and_translates_waiting_status(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Catches a responsive detail view exposing raw status or moving approval after history."""
    command = _action(
        kind="run_command",
        summary="install package with hidden context",
        args=["python", "-m", "pip", "install", "example-package==1.2.3"],
    )
    app, _, _ = _waiting_app(tmp_path, monkeypatch, [command])
    task = app.state.local.task
    assert task is not None

    page = asyncio.run(_request(app, "GET", f"/tasks/{task.id}"))
    detail = _TaskDetailDom()
    detail.feed(page.text)

    assert page.status_code == 200
    assert detail.regions[:3] == ["任务状态", "操作审批", "审计时间线"]
    assert detail.decision_values == ["reject", "once", "always"]
    assert detail.raw_status == "waiting_approval"
    assert detail.visible_status == "等待审批"
    assert detail.context_title == "任务详情"
    assert detail.current_navigation_hrefs == ["/tasks/new"]
    assert detail.event_list_polling_root_id == detail.status_polling_root_id


def test_polling_protocol_reloads_running_page_when_latest_event_requires_approval(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Catches a live approval event leaving a running page without a server-rendered card."""
    delete = _action(kind="delete_path", summary="remove obsolete file", path="obsolete.txt")
    app, root, _ = _waiting_app(tmp_path, monkeypatch, [delete])
    task = app.state.local.task
    assert task is not None
    task.status = TaskStatus.RUNNING
    EventStore(root).append(RunEvent(task_id=task.id, task_status=TaskStatus.RUNNING))

    running_page = asyncio.run(_request(app, "GET", f"/tasks/{task.id}"))
    EventStore(root).append(
        RunEvent(task_id=task.id, task_status=TaskStatus.WAITING_APPROVAL)
    )
    feed = asyncio.run(_request(app, "GET", f"/tasks/{task.id}/events"))
    harness = _PollingScriptHarness(
        (Path(__file__).parents[1] / "src" / "guardedpy" / "static" / "app.js").read_text()
    )

    assert running_page.status_code == 200
    assert 'data-current-status="running"' in running_page.text
    assert "ACTION APPROVAL REQUIRED" not in running_page.text
    assert feed.json()[-1]["task_status"] == "waiting_approval"
    assert harness.poll(current_status="running", latest_status="waiting_approval") == _PollEffects(
        ("fetch", "reload")
    )


def test_polling_harness_executes_cleanup_reload_update_and_error_branches() -> None:
    """Catches broken whole-script wiring or any polling branch with the wrong effects."""
    harness = _PollingScriptHarness(
        (Path(__file__).parents[1] / "src" / "guardedpy" / "static" / "app.js").read_text()
    )

    assert harness.poll(current_status="running", latest_status="completed") == _PollEffects(
        ("fetch", "clearInterval", "reload")
    )
    assert harness.poll(current_status="running", latest_status="running") == _PollEffects(
        ("fetch", "updateStatus", "replaceChildren"), "运行中"
    )
    assert harness.poll(
        current_status="running", latest_status="running", response_ok=False
    ) == _PollEffects(("fetch",))
    assert harness.poll(current_status="running", latest_status=None) == _PollEffects(("fetch",))


def test_task_detail_renders_bounded_feedback_node_id_without_raw_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Catches raw feedback data crossing the fixed audit schema into the timeline."""
    delete = _action(kind="delete_path", summary="remove obsolete file", path="obsolete.txt")
    app, root, _ = _waiting_app(tmp_path, monkeypatch, [delete])
    task = app.state.local.task
    assert task is not None
    raw_output = "RAW-PYTEST-OUTPUT-MUST-NOT-RENDER"
    with pytest.raises(ValidationError):
        FeedbackAudit(
            kind="assertion_failure",
            node_id="tests/test_value.py",
            raw_output=raw_output,
        )
    EventStore(root).append(
        RunEvent(
            task_id=task.id,
            task_status=TaskStatus.WAITING_APPROVAL,
            feedback=FeedbackAudit(
                kind="assertion_failure",
                node_id="tests/test_value.py",
            ),
        )
    )

    page = asyncio.run(_request(app, "GET", f"/tasks/{task.id}"))
    feed = asyncio.run(_request(app, "GET", f"/tasks/{task.id}/events"))

    assert page.status_code == 200
    assert "tests/test_value.py" in page.text
    assert "tests/test_value.py" in feed.text
    assert raw_output not in page.text + feed.text


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
                    "-old\n+++ b/RAW-PATCH-DATA\n"
                ),
            ),
            "Paths: README.md",
            "RAW-PATCH-DATA",
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
    assert f"项目：{root.resolve()}" in page.text
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
            "审批：已同意",
            id="once",
        ),
        pytest.param(
            "always",
            "approval.granted_always",
            "user approved a constrained persistent command rule",
            True,
            "审批：已同意",
            id="always",
        ),
        pytest.param(
            "reject",
            "approval.declined",
            "user declined the action",
            False,
            "审批：已拒绝",
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

    _PollingScriptHarness(
        (Path(__file__).parents[1] / "src" / "guardedpy" / "static" / "app.js").read_text()
    )


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
    assert f"项目：{original_root.resolve()}" in detail.text
    assert "delete.approval_required" in detail.text
    assert feed.status_code == 200
    events = feed.json()
    assert events
    assert all(event["affected_project"] == str(original_root.resolve()) for event in events)
    assert all(event["task_status"] != "completed" for event in events)


def test_reconfigured_two_task_history_survives_fresh_app_with_original_roots(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Catches task/config history remaining process-local or rebinding to the latest root."""
    import guardedpy.web as web

    first_root = _project_root(tmp_path / "first")
    second_root = _project_root(tmp_path / "second")
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    ImmediateThread.started = []
    monkeypatch.setattr(web, "Thread", ImmediateThread)
    created_orchestrators: list[tuple[Path, str]] = []

    def factory(project_root: Path, config: Any, memory: Any) -> TaskOrchestrator:
        created_orchestrators.append((project_root, config.model))
        return TaskOrchestrator(
            project_root,
            ScriptedLLM([_action(kind="finish", summary="stop", status="blocked")]),
            memory_store=memory,
        )

    credentials = FakeCredentials()
    app = web.create_app("local", web.WebServices(credentials, factory))
    assert asyncio.run(
        _request(app, "POST", "/setup", data=_setup_data(first_root))
    ).status_code == 303
    first_created = asyncio.run(
        _request(
            app,
            "POST",
            "/tasks",
            data={"mode": "feature", "description": "First historical task"},
        )
    )
    assert first_created.status_code == 303
    assert first_created.headers["location"].startswith("/tasks/")
    assert first_created.headers["location"] != "/tasks/new"
    first_id = UUID(first_created.headers["location"].rsplit("/", 1)[-1])

    reconfigured = asyncio.run(
        _request(
            app,
            "POST",
            "/setup",
            data={**_setup_data(second_root), "api_key": "", "model": "deepseek-reasoner"},
        )
    )
    second_created = asyncio.run(
        _request(
            app,
            "POST",
            "/tasks",
            data={"mode": "feature", "description": "Second historical task"},
        )
    )
    assert second_created.status_code == 303
    assert second_created.headers["location"].startswith("/tasks/")
    assert second_created.headers["location"] != "/tasks/new"
    second_id = UUID(second_created.headers["location"].rsplit("/", 1)[-1])

    assert first_created.headers["location"] == f"/tasks/{first_id}"
    assert reconfigured.status_code == 303
    assert second_created.headers["location"] == f"/tasks/{second_id}"
    assert created_orchestrators == [
        (first_root.resolve(), "deepseek-chat"),
        (second_root.resolve(), "deepseek-reasoner"),
    ]
    assert credentials.configured is True
    assert "test-key" not in local_state_path().read_text()

    fresh_app = web.create_app("local", web.WebServices(credentials, factory))
    setup_page = asyncio.run(_request(fresh_app, "GET", "/setup"))
    task_page = asyncio.run(_request(fresh_app, "GET", "/tasks/new"))
    first_detail = asyncio.run(_request(fresh_app, "GET", f"/tasks/{first_id}"))
    first_feed = asyncio.run(_request(fresh_app, "GET", f"/tasks/{first_id}/events"))
    second_detail = asyncio.run(_request(fresh_app, "GET", f"/tasks/{second_id}"))

    assert len(created_orchestrators) == 2
    assert setup_page.status_code == 200
    assert str(second_root.resolve()) in setup_page.text
    assert "deepseek-reasoner" in setup_page.text
    assert "test-key" not in setup_page.text
    assert task_page.status_code == 200
    assert f'href="/tasks/{first_id}"' in task_page.text
    assert f'href="/tasks/{second_id}"' in task_page.text
    assert first_detail.status_code == 200
    assert second_detail.status_code == 200
    assert first_feed.status_code == 200
    assert all(
        event["affected_project"] == str(first_root.resolve())
        for event in first_feed.json()
    )
    assert EventStore(first_root).tasks()[0].config.model == "deepseek-chat"
    assert EventStore(second_root).tasks()[0].config.model == "deepseek-reasoner"


def test_fresh_app_marks_indexed_unfinished_task_interrupted_without_resuming_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Catches startup resuming an unfinished loop or dropping it from task history."""
    import guardedpy.web as web

    root = _project_root(tmp_path)
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    factory_calls: list[Path] = []

    class DormantThread:
        def __init__(self, **kwargs: Any) -> None:
            del kwargs

        def start(self) -> None:
            return None

    class PendingOrchestrator:
        def submit(self, task: Any) -> Any:
            return task

        def run(self, task: Any) -> Any:
            raise AssertionError("an interrupted task must not resume")

        def cancel(self, task_id: UUID) -> Any:
            raise AssertionError(task_id)

        def resolve_approval(self, task_id: UUID, action_hash: str, *, decision: str) -> bool:
            raise AssertionError((task_id, action_hash, decision))

    def factory(project_root: Path, config: Any, memory: Any) -> PendingOrchestrator:
        del config, memory
        factory_calls.append(project_root)
        return PendingOrchestrator()

    monkeypatch.setattr(web, "Thread", DormantThread)
    credentials = FakeCredentials()
    app = web.create_app("local", web.WebServices(credentials, factory))
    assert asyncio.run(
        _request(app, "POST", "/setup", data=_setup_data(root))
    ).status_code == 303
    created = asyncio.run(
        _request(
            app,
            "POST",
            "/tasks",
            data={"mode": "feature", "description": "Interrupted on restart"},
        )
    )
    assert created.status_code == 303
    assert created.headers["location"].startswith("/tasks/")
    assert created.headers["location"] != "/tasks/new"
    task_id = UUID(created.headers["location"].rsplit("/", 1)[-1])
    assert len(factory_calls) == 1

    fresh_app = web.create_app("local", web.WebServices(credentials, factory))
    detail = asyncio.run(_request(fresh_app, "GET", f"/tasks/{task_id}"))
    feed = asyncio.run(_request(fresh_app, "GET", f"/tasks/{task_id}/events"))

    assert len(factory_calls) == 1
    assert detail.status_code == 200
    assert "interrupted" in detail.text
    assert feed.status_code == 200
    assert feed.json()[-1]["task_status"] == "interrupted"
    assert feed.json()[-1]["stop_reason"] == "service_restarted"


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
