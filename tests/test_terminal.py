"""Safe plain-text fallback contracts for GuardedPy."""

from __future__ import annotations

from io import StringIO
from pathlib import Path

from guardedpy.config import HarnessConfig
from guardedpy.credentials import CredentialStatus
from guardedpy.discovery import ProjectProfile
from guardedpy.domain import TaskIntent, TaskState, TaskStatus
from guardedpy.events import StoredRunEvent


class _Runtime:
    """Lifecycle fake whose behavior is asserted through the plain renderer."""

    def __init__(self, profile: ProjectProfile, *, approval: bool = False) -> None:
        self.project_root = profile.root
        self.config = HarnessConfig(profile=profile)
        self.approval = approval
        self.updated = False
        self.created: list[TaskState] = []

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
