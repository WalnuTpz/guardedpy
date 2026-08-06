"""Safe plain-text fallback contracts for GuardedPy."""

from __future__ import annotations

from io import StringIO
from pathlib import Path
import subprocess
from uuid import uuid4

import pytest

from guardedpy.config import HarnessConfig
from guardedpy.credentials import CredentialStatus
from guardedpy.discovery import ProjectProfile
from guardedpy.domain import CommandApprovalRule, CommandRuleKind, TaskIntent, TaskState, TaskStatus
from guardedpy.events import StoredRunEvent
from guardedpy.memory import MemoryEntry


class _Runtime:
    """Lifecycle fake whose behavior is asserted through the plain renderer."""

    def __init__(self, profile: ProjectProfile, *, approval: bool = False) -> None:
        self.project_root = profile.root
        self.config = HarnessConfig(profile=profile)
        self.approval = approval
        self.updated = False
        self.created: list[TaskState] = []
        self.rule_id = "rule-to-revoke"
        self.memory_id = uuid4()
        self.revoked: list[str] = []
        self.approved: list[object] = []
        self.removed: list[object] = []

    def create_task(self, description: str, intent: TaskIntent = TaskIntent.CODING, review_path: str | None = None) -> TaskState:
        task = TaskState(description=description, intent=intent, config=self.config, review_path=review_path)
        self.created.append(task)
        return task

    def run(self, task_id: object) -> TaskState:
        task = next(task for task in self.created if task.id == task_id)
        task.status = TaskStatus.WAITING_APPROVAL if self.approval else TaskStatus.COMPLETED
        return task

    def events(self, task_id: object) -> list[StoredRunEvent]:
        del task_id
        return []

    def credential_status(self) -> CredentialStatus:
        return CredentialStatus(configured=False)

    def update_credential(self, key: str) -> None:
        del key
        self.updated = True

    def command_rules(self) -> list[CommandApprovalRule]:
        return [
            CommandApprovalRule(
                id=self.rule_id,
                kind=CommandRuleKind.GIT_DIFF_CHECK,
                project_hash="safe",
            )
        ]

    def delete_command_rule(self, rule_id: str) -> bool:
        self.revoked.append(rule_id)
        return True

    def memory_proposals(self) -> list[MemoryEntry]:
        return [MemoryEntry(id=self.memory_id, task_id=uuid4(), text="remember tests")]

    def memories(self) -> list[MemoryEntry]:
        return []

    def approve_memory(self, memory_id: object) -> MemoryEntry:
        self.approved.append(memory_id)
        return self.memory_proposals()[0]

    def delete_memory(self, memory_id: object) -> None:
        self.removed.append(memory_id)


def _profile(tmp_path: Path) -> ProjectProfile:
    from guardedpy.discovery import discover_project

    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "app.py").write_text("VALUE = 1\n")
    (tmp_path / "tests").mkdir()
    return discover_project(tmp_path)


def test_noninteractive_task_stops_on_approval_without_consuming_stdin(tmp_path: Path) -> None:
    """Catches piped automation silently approving a dangerous action."""
    from guardedpy.terminal import run_noninteractive_task

    runtime = _Runtime(_profile(tmp_path), approval=True)
    output = StringIO()

    code = run_noninteractive_task(runtime, "remove artifact", TaskIntent.CODING, output)

    assert code == 1
    assert "waiting_approval" in output.getvalue()
    assert "需要人工审批，非交互模式已安全停止。" in output.getvalue()


def test_plain_session_refuses_piped_credential_entry_and_lists_only_supported_commands(tmp_path: Path) -> None:
    """Catches the fallback accepting secrets or advertising a retired command surface."""
    from guardedpy.terminal import run_plain_session

    runtime = _Runtime(_profile(tmp_path))
    output = StringIO()

    code = run_plain_session(runtime, StringIO("/help\n/credentials update\n/exit\n"), output)

    rendered = output.getvalue()
    assert code == 0
    assert runtime.updated is False
    assert "非交互终端不能录入凭据。" in rendered
    for command in ("/new", "/plan", "/review", "/tests", "/diff", "/doctor", "/help"):
        assert command in rendered
    for retired in ("/init", "/task", "/serve", "!shell"):
        assert retired not in rendered


def test_plain_help_is_grouped_and_treats_status_as_unknown(tmp_path: Path) -> None:
    """Catches the retired status command returning through the safe renderer."""
    from guardedpy.terminal import run_plain_session

    output = StringIO()
    code = run_plain_session(_Runtime(_profile(tmp_path)), StringIO("/help\n/status\n/exit\n"), output)

    assert code == 0
    rendered = output.getvalue()
    for group in ("会话与对话", "任务与检查", "设置与安全"):
        assert group in rendered
    assert "/status" not in rendered
    assert rendered.endswith("未知命令。\n")


def test_plain_session_requires_exact_command_names_before_mutating_defaults(tmp_path: Path) -> None:
    """Catches a slash-prefix typo changing persistent model or workflow state."""
    from guardedpy.terminal import run_plain_session

    runtime = _Runtime(_profile(tmp_path))
    output = StringIO()

    code = run_plain_session(
        runtime,
        StringIO("/modeloops deepseek-v4-pro\n/planish inspect\n/exit\n"),
        output,
    )

    assert code == 0
    assert not hasattr(runtime, "defaults")
    assert output.getvalue() == "未知命令。\n未知命令。\n"


def test_plain_session_manages_only_explicit_permission_and_memory_identifiers(tmp_path: Path) -> None:
    """Catches rule or memory management being a display-only command surface."""
    from guardedpy.terminal import run_plain_session

    runtime = _Runtime(_profile(tmp_path))
    output = StringIO()

    code = run_plain_session(
        runtime,
        StringIO(
            f"/permissions revoke {runtime.rule_id}\n"
            f"/memory approve {runtime.memory_id}\n"
            f"/memory remove {runtime.memory_id}\n/exit\n"
        ),
        output,
    )

    assert code == 0
    assert runtime.revoked == [runtime.rule_id]
    assert runtime.approved == [runtime.memory_id]
    assert runtime.removed == [runtime.memory_id]


@pytest.mark.parametrize("command", ("/tests", "/diff"))
def test_plain_read_only_subprocess_commands_use_the_configured_time_budget(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, command: str
) -> None:
    """Catches a redirected read-only command waiting indefinitely for its subprocess."""
    from guardedpy.terminal import run_plain_session

    runtime = _Runtime(_profile(tmp_path))
    runtime.config = runtime.config.model_copy(update={"timeout_seconds": 5})
    captured: dict[str, object] = {}

    def bounded_run(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        captured["args"] = args
        captured["timeout"] = kwargs["timeout"]
        return subprocess.CompletedProcess(args[0], 0, stdout="")

    monkeypatch.setattr("guardedpy.terminal.subprocess.run", bounded_run)
    output = StringIO()

    code = run_plain_session(runtime, StringIO(f"{command}\n/exit\n"), output)

    assert code == 0
    assert captured["timeout"] == 5
