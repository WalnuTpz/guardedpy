"""Offline contracts for the local GuardedPy terminal client."""

from __future__ import annotations

from io import StringIO
from pathlib import Path
from uuid import UUID

import pytest

from guardedpy.config import HarnessConfig
from guardedpy.credentials import CredentialStatus
from guardedpy.domain import FeedbackKind, PolicyVerdict, TaskMode, TaskState, TaskStatus
from guardedpy.events import StopReason, StoredRunEvent


def _config() -> HarnessConfig:
    return HarnessConfig(
        source_dirs=(Path("src"),),
        test_dirs=(Path("tests"),),
        pytest_command=("pytest",),
    )


class FakeRuntime:
    """A local-only runtime fake that exposes terminal boundary behavior."""

    def __init__(
        self,
        *,
        wait_for_approval: bool = False,
        interrupt: bool = False,
        consume_reject: bool = False,
    ) -> None:
        self.configured = False
        self.project_root: Path | None = None
        self.config: HarnessConfig | None = None
        self.created: list[TaskState] = []
        self.decisions: list[tuple[UUID, str, str]] = []
        self.cancelled: list[UUID] = []
        self._wait_for_approval = wait_for_approval
        self._interrupt = interrupt
        self._consume_reject = consume_reject

    def credential_status(self) -> CredentialStatus:
        return CredentialStatus(configured=self.configured)

    def create_task(
        self, description: str, mode: TaskMode, bugfix_target: str | None
    ) -> TaskState:
        task = TaskState(
            description=description,
            mode=mode,
            bugfix_target=bugfix_target,
            config=self.config or _config(),
        )
        self.created.append(task)
        return task

    def run(self, task_id: UUID) -> TaskState:
        task = self.task(task_id)
        if self._interrupt:
            raise KeyboardInterrupt
        if self._wait_for_approval and not self.decisions:
            task.status = TaskStatus.WAITING_APPROVAL
        else:
            task.status = TaskStatus.COMPLETED
        return task

    def resolve_approval(self, task_id: UUID, action_hash: str, decision: str) -> bool:
        self.decisions.append((task_id, action_hash, decision))
        if self._consume_reject and decision == "reject":
            self.task(task_id).status = TaskStatus.BLOCKED
            return False
        return True

    def cancel(self, task_id: UUID) -> TaskState:
        task = self.task(task_id)
        task.status = TaskStatus.CANCELLED
        self.cancelled.append(task_id)
        return task

    def task(self, task_id: UUID) -> TaskState:
        return next(task for task in self.created if task.id == task_id)

    def tasks(self) -> list[TaskState]:
        return self.created

    def events(self, task_id: UUID) -> list[StoredRunEvent]:
        return [
            StoredRunEvent(
                task_id=task_id,
                task_status=self.task(task_id).status,
                action_hash="bound-approval-hash",
                action_projection="删除项目内文件",
                policy_verdict=PolicyVerdict.APPROVAL_REQUIRED,
                feedback_kind=FeedbackKind.ASSERTION_FAILURE,
                feedback_node_id="tests/test_value.py::test_value",
                stop_reason=StopReason.COMPLETED,
            )
        ]

    def memory_proposals(self) -> list[object]:
        return []

    def memories(self) -> list[object]:
        return []

    def command_rules(self) -> list[object]:
        return []

    def update_credential(self, api_key: str) -> None:
        self.configured = bool(api_key)

    def clear_credential(self) -> None:
        self.configured = False


def test_repl_refuses_credential_update_secret_entry_from_non_tty_without_getpass(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Catches a piped credential update reading a secret through getpass fallback."""
    from guardedpy.cli import run_repl

    runtime = FakeRuntime()
    output = StringIO()
    monkeypatch.setattr(
        "guardedpy.cli.getpass", lambda _: pytest.fail("non-TTY update must not call getpass")
    )

    code = run_repl(runtime, StringIO("/credentials\nupdate\n/exit\n"), output, lambda: False)

    assert code == 0
    assert runtime.configured is False
    assert output.getvalue().endswith("非交互终端不能录入凭据。\n")


def test_one_shot_bugfix_requires_an_explicit_pytest_node() -> None:
    """Catches a bugfix one-shot creating a task without its required node."""
    from guardedpy.cli import main

    assert main(["--prompt", "repair", "--mode", "bugfix"], runtime_factory=FakeRuntime) == 2


def test_one_shot_bugfix_rejects_a_whitespace_only_pytest_node_before_composition() -> None:
    """Catches a whitespace target constructing a runtime or creating a malformed task."""
    from guardedpy.cli import main

    composed: list[FakeRuntime] = []

    def runtime_factory() -> FakeRuntime:
        runtime = FakeRuntime()
        composed.append(runtime)
        return runtime

    assert (
        main(
            ["--prompt", "repair", "--mode", "bugfix", "--target", "   "],
            runtime_factory=runtime_factory,
        )
        == 2
    )
    assert composed == []


@pytest.mark.parametrize("decision", ("reject", "once", "always"))
def test_one_shot_consumes_injected_approval_input_until_the_task_finishes(
    decision: str,
) -> None:
    """Catches a waiting one-shot ignoring stdin and repeatedly reading an empty approval."""
    from guardedpy.cli import main

    class ApprovalOutput(StringIO):
        def write(self, text: str) -> int:
            if text == "审批输入无效。\n":
                raise AssertionError("one-shot ignored its supplied approval input")
            return super().write(text)

    runtime = FakeRuntime(wait_for_approval=True)
    output = ApprovalOutput()

    code = main(
        ["--prompt", "dangerous task"],
        runtime_factory=lambda: runtime,
        stdin=StringIO(f"{decision}\n"),
        stdout=output,
    )

    assert code == 0
    assert runtime.decisions == [(runtime.created[0].id, "bound-approval-hash", decision)]
    assert runtime.created[0].status is TaskStatus.COMPLETED


def test_repl_runs_ordinary_feature_text_and_renders_only_safe_progress() -> None:
    """Catches a terminal task omitting its safe lifecycle projection or leaking raw details."""
    from guardedpy.cli import run_repl

    runtime = FakeRuntime()
    output = StringIO()

    code = run_repl(runtime, StringIO("implement safely\n/exit\n"), output, lambda: False)

    rendered = output.getvalue()
    assert code == 0
    assert runtime.created[0].description == "implement safely"
    assert runtime.created[0].mode is TaskMode.FEATURE
    assert str(runtime.created[0].id) in rendered
    assert "completed" in rendered
    assert "删除项目内文件" in rendered
    assert "approval_required" in rendered
    assert "assertion_failure" in rendered
    assert "tests/test_value.py::test_value" in rendered
    assert "completed" in rendered
    assert "bound-approval-hash" not in rendered
    assert "\x1b[" not in rendered


def test_repl_resolves_only_an_exact_approval_decision_before_continuing() -> None:
    """Catches accepting an unrecognized approval value or losing the pending task state."""
    from guardedpy.cli import run_repl

    runtime = FakeRuntime(wait_for_approval=True)
    output = StringIO()

    code = run_repl(runtime, StringIO("dangerous task\nyes\nonce\n/exit\n"), output, lambda: False)

    assert code == 0
    assert runtime.decisions == [(runtime.created[0].id, "bound-approval-hash", "once")]
    assert "审批输入无效。" in output.getvalue()
    assert runtime.created[0].status is TaskStatus.COMPLETED


def test_repl_renders_a_consumed_reject_approval_as_blocked_not_stale() -> None:
    """Catches the terminal reporting a valid rejected approval as stale."""
    from guardedpy.cli import run_repl

    runtime = FakeRuntime(wait_for_approval=True, consume_reject=True)
    output = StringIO()

    code = run_repl(runtime, StringIO("dangerous task\nreject\n/exit\n"), output, lambda: False)

    assert code == 0
    assert runtime.created[0].status is TaskStatus.BLOCKED
    assert "blocked" in output.getvalue()
    assert "审批请求已失效。" not in output.getvalue()


def test_repl_rejects_unknown_slash_commands_without_creating_a_task() -> None:
    """Catches unsupported slash commands acquiring a runtime mutation path."""
    from guardedpy.cli import run_repl

    runtime = FakeRuntime()
    output = StringIO()

    code = run_repl(runtime, StringIO("/shell rm -rf .\n/exit\n"), output, lambda: False)

    assert code == 0
    assert runtime.created == []
    assert output.getvalue() == "未知命令。\n"


def test_repl_cancels_the_active_task_after_ctrl_c() -> None:
    """Catches Ctrl-C exiting while a live task remains active in LocalRuntime."""
    from guardedpy.cli import run_repl

    runtime = FakeRuntime(interrupt=True)
    output = StringIO()

    code = run_repl(runtime, StringIO("long task\n"), output, lambda: False)

    assert code == 0
    assert runtime.cancelled == [runtime.created[0].id]
    assert "cancelled" in output.getvalue()


def test_repl_cancels_the_waiting_task_when_ctrl_c_interrupts_the_approval_prompt() -> None:
    """Catches Ctrl-C at an approval prompt leaving its active runtime task alive."""
    from guardedpy.cli import run_repl

    class InterruptAtApproval:
        def __init__(self) -> None:
            self._lines = iter(("dangerous task\n", KeyboardInterrupt(), "/exit\n"))

        def readline(self) -> str:
            value = next(self._lines)
            if isinstance(value, KeyboardInterrupt):
                raise value
            return value

    runtime = FakeRuntime(wait_for_approval=True)
    output = StringIO()

    code = run_repl(runtime, InterruptAtApproval(), output, lambda: False)

    assert code == 0
    assert runtime.cancelled == [runtime.created[0].id]
    assert "cancelled" in output.getvalue()


def test_task_command_requires_target_for_bugfix_before_runtime_mutation() -> None:
    """Catches `/task` forwarding a malformed bugfix request to the runtime."""
    from guardedpy.cli import run_repl

    runtime = FakeRuntime()
    output = StringIO()

    code = run_repl(runtime, StringIO("/task\nbugfix\nrepair\n \n/exit\n"), output, lambda: False)

    assert code == 0
    assert runtime.created == []
    assert "缺陷修复任务必须提供 pytest node。" in output.getvalue()


def test_repl_help_omits_retired_manual_init() -> None:
    """Catches interactive help retaining the retired manual setup workflow."""
    from guardedpy.cli import run_repl

    output = StringIO()

    assert run_repl(FakeRuntime(), StringIO("/help\n/exit\n"), output, lambda: False) == 0
    assert "/init" not in output.getvalue()
    assert "/task" in output.getvalue()


def test_help_exposes_only_terminal_options_and_no_retired_surface(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Catches the sole CLI advertising a server, Web/API surface, or setup command."""
    from guardedpy.cli import main

    assert main(["--help"], runtime_factory=lambda: pytest.fail("help composed runtime")) == 0

    help_text = capsys.readouterr().out.lower()
    for retired in ("serve", "server", "webui", "api", "/init"):
        assert retired not in help_text
