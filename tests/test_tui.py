"""Textual contracts for the safe GuardedPy terminal session."""

from __future__ import annotations

import asyncio
from pathlib import Path
from threading import Event, Timer

import pytest
from textual.css.query import NoMatches
from guardedpy.config import HarnessConfig
from guardedpy.credentials import CredentialStatus
from guardedpy.discovery import ProjectProfile
from guardedpy.domain import TaskIntent, TaskState, TaskStatus
from guardedpy.events import StoredRunEvent
from guardedpy.runtime import LocalRuntime, RuntimeServices


class _Runtime:
    """The smallest real-session boundary needed to exercise Textual widgets."""

    def __init__(self, profile: ProjectProfile) -> None:
        self.project_root = profile.root
        self.config = HarnessConfig(profile=profile)
        self.created: list[TaskState] = []
        self.cancelled: list[object] = []
        self.history_reads = 0
        self.configured = True

    def events(self, task_id: object) -> list[StoredRunEvent]:
        del task_id
        return []

    def update_future_defaults(self, **changes: str) -> HarnessConfig:
        self.updated = changes
        self.config = self.config.model_copy(update=changes)
        return self.config

    def tasks(self) -> list[TaskState]:
        self.history_reads += 1
        return []

    def credential_status(self) -> CredentialStatus:
        return CredentialStatus(configured=self.configured)

    def update_credential(self, key: str) -> None:
        assert key
        self.configured = True

    def create_task(self, description: str, intent: object, review_path: str | None = None) -> TaskState:
        task = TaskState(description=description, config=self.config, intent=intent, review_path=review_path)
        self.created.append(task)
        return task

    def run(self, task_id: object) -> TaskState:
        task = next(task for task in self.created if task.id == task_id)
        task.status = TaskStatus.WAITING_APPROVAL
        return task

    def cancel(self, task_id: object) -> TaskState:
        task = next(task for task in self.created if task.id == task_id)
        task.status = TaskStatus.CANCELLED
        self.cancelled.append(task_id)
        return task


class _Credentials:
    """A credential port for tests that exercise the real LocalRuntime."""

    def status(self) -> CredentialStatus:
        return CredentialStatus(configured=True)

    def set_key(self, key: str) -> None:
        del key

    def clear_key(self) -> None:
        return None


def _profile(tmp_path: Path) -> ProjectProfile:
    from guardedpy.discovery import discover_project

    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "app.py").write_text("VALUE = 1\n")
    (tmp_path / "tests").mkdir()
    return discover_project(tmp_path)


def test_composer_enter_submits_help_and_shift_enter_or_ctrl_j_adds_newline(tmp_path: Path) -> None:
    """Catches TextArea consuming Enter instead of delivering the submitted command."""
    from guardedpy.tui import Composer, GuardedPyApp
    from textual.widgets import RichLog

    profile = _profile(tmp_path)
    app = GuardedPyApp(_Runtime(profile), profile)

    async def check() -> None:
        async with app.run_test() as pilot:
            await pilot.click("#composer")
            await pilot.press(*"/help", "enter")
            transcript = app.query_one("#transcript", RichLog)
            assert any("会话与对话" in line.text for line in transcript.lines)
            composer = app.query_one("#composer", Composer)
            await pilot.press("shift+enter")
            assert "\n" in composer.text
            composer.text = ""
            await pilot.press("ctrl+j")
            assert composer.text == "\n"

    asyncio.run(check())


def test_tui_mounts_safe_session_widgets_and_filters_exact_command_palette(tmp_path: Path) -> None:
    """Catches the interactive session losing its composer, status, or safe command UI."""
    from guardedpy.tui import COMMANDS, Composer, GuardedPyApp

    profile = _profile(tmp_path)
    app = GuardedPyApp(_Runtime(profile), profile)

    async def check() -> None:
        async with app.run_test() as pilot:
            assert "项目：" in str(app.query_one("#status").render())
            assert isinstance(app.query_one("#composer"), Composer)
            await pilot.click("#composer")
            await pilot.press("/")
            assert app.query_one("#command-palette").display is True
            await pilot.press("h")
            assert [
                item.query_one("Static").render()
                for item in app.query("#command-palette ListItem")
                if item.display
            ] == ["/history", "/help"]
            await pilot.click("#command-history")
            assert app.query_one("#composer", Composer).text == "/history"
            assert tuple(COMMANDS) == (
                "/history", "/new", "/clear", "/exit", "/plan", "/review",
                "/tests", "/diff", "/permissions", "/credentials", "/memory",
                "/model", "/effort", "/doctor", "/help",
            )

    asyncio.run(check())


def test_tui_lifecycle_tracks_baseline_task_and_safe_cancel_exit(tmp_path: Path) -> None:
    """Catches a session hiding active state or abandoning an active task on Ctrl+C/exit."""
    from guardedpy.tui import GuardedPyApp

    profile = _profile(tmp_path)
    runtime = _Runtime(profile)
    app = GuardedPyApp(runtime, profile)

    async def check() -> None:
        async with app.run_test() as pilot:
            app.submit("repair value")
            await pilot.pause()
            await app.workers.wait_for_complete()
            assert "基线：baseline_pending · 任务：waiting_approval" in str(
                app.query_one("#status").render()
            )
            await pilot.press("ctrl+c")
            assert runtime.cancelled == [runtime.created[0].id]
            app.submit("repair value")
            await pilot.pause()
            await app.workers.wait_for_complete()
            app.submit("/exit")
            await pilot.pause()
            assert app.screen.query_one("#exit-confirm")

    asyncio.run(check())


def test_tui_transcript_records_the_submitted_user_request_before_lifecycle(tmp_path: Path) -> None:
    """Catches a task starting without an auditable, safe user-message projection."""
    from guardedpy.tui import GuardedPyApp
    from textual.widgets import RichLog

    profile = _profile(tmp_path)
    app = GuardedPyApp(_Runtime(profile), profile)

    async def check() -> None:
        async with app.run_test() as pilot:
            app.submit("repair value")
            await pilot.pause()
            await app.workers.wait_for_complete()
            transcript = app.query_one("#transcript", RichLog)
            assert transcript.lines[0].text == "用户：repair value"

    asyncio.run(check())


def test_tui_cancels_a_real_blocking_local_runtime_without_freezing_the_session(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Catches Ctrl+C waiting on LocalRuntime's lifecycle lock while a run is blocked."""
    from guardedpy.tui import GuardedPyApp

    class BlockingOrchestrator:
        def __init__(self) -> None:
            self.task: TaskState | None = None
            self.started = Event()
            self.release = Event()
            self.cancel_before_release = False

        def submit(self, task: TaskState) -> TaskState:
            self.task = task
            return task

        def run(self, task: TaskState) -> TaskState:
            task.status = TaskStatus.RUNNING
            self.started.set()
            self.release.wait()
            return task

        def cancel(self, task_id: object) -> TaskState:
            assert self.task is not None
            assert task_id == self.task.id
            self.cancel_before_release = not self.release.is_set()
            self.task.status = TaskStatus.CANCELLED
            self.release.set()
            return self.task

        def resolve_approval(self, task_id: object, action_hash: str, *, decision: object) -> bool:
            del task_id, action_hash, decision
            return False

    profile = _profile(tmp_path)
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    orchestrator = BlockingOrchestrator()
    runtime = LocalRuntime(
        RuntimeServices(
            credentials=_Credentials(),
            orchestrator_factory=lambda root, config, memory: orchestrator,
        )
    )
    runtime.setup(profile, api_key=None)
    app = GuardedPyApp(runtime, profile)

    async def check() -> None:
        timer = Timer(0.3, orchestrator.release.set)
        timer.start()
        try:
            async with app.run_test() as pilot:
                app.submit("long repair")
                await pilot.pause()
                assert orchestrator.started.is_set()
                assert orchestrator.release.is_set() is False
                await pilot.press("ctrl+c")
                assert orchestrator.cancel_before_release is True
                app.submit("/help")
                await pilot.pause()
                assert "项目：" in str(app.query_one("#status").render())
        finally:
            timer.cancel()
            orchestrator.release.set()

    asyncio.run(check())


def test_tui_runs_blocking_tests_command_off_loop_without_an_active_task(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Catches `/tests` freezing Textual while its bounded subprocess is still running."""
    from guardedpy.tui import GuardedPyApp
    from textual.widgets import RichLog

    entered = Event()
    release = Event()

    class CompletedRun:
        returncode = 0
        stdout = ""

    def blocking_run(*args: object, **kwargs: object) -> CompletedRun:
        del args, kwargs
        entered.set()
        release.wait()
        return CompletedRun()

    monkeypatch.setattr("guardedpy.terminal.subprocess.run", blocking_run)
    profile = _profile(tmp_path)
    app = GuardedPyApp(_Runtime(profile), profile)

    async def check() -> None:
        timer = Timer(1, release.set)
        timer.start()
        try:
            async with app.run_test() as pilot:
                app.submit("/tests")
                await pilot.pause()
                assert entered.is_set()
                assert release.is_set() is False

                await pilot.press("ctrl+c")
                app.submit("/help")
                await pilot.pause()
                assert "项目：" in str(app.query_one("#status").render())

                release.set()
                await pilot.pause()
                transcript = app.query_one("#transcript", RichLog)
                assert any("完整测试：passed" in line.text for line in transcript.lines)
        finally:
            timer.cancel()
            release.set()

    asyncio.run(check())


def test_tui_unmount_cancels_its_active_worker_and_discards_late_completion(
    tmp_path: Path,
) -> None:
    """Catches session teardown leaving a daemon worker or repainting from its late result."""
    from guardedpy.tui import GuardedPyApp

    class LateCompletionRuntime(_Runtime):
        def __init__(self, profile: ProjectProfile) -> None:
            super().__init__(profile)
            self.started = Event()
            self.release = Event()

        def run(self, task_id: object) -> TaskState:
            task = next(task for task in self.created if task.id == task_id)
            self.started.set()
            self.release.wait(timeout=1)
            return task.model_copy(update={"status": TaskStatus.WAITING_APPROVAL})

        def cancel(self, task_id: object) -> TaskState:
            self.release.set()
            return super().cancel(task_id)

    profile = _profile(tmp_path)
    runtime = LateCompletionRuntime(profile)
    app = GuardedPyApp(runtime, profile)

    async def check() -> None:
        async with app.run_test() as pilot:
            app.submit("long repair")
            await pilot.pause()
            assert runtime.started.is_set()

            app.on_unmount()
            await pilot.pause()

            assert runtime.cancelled == [runtime.created[0].id]
            assert app._active_task is None
            assert app._run_threads == {}

    asyncio.run(check())


def test_tui_status_is_unknown_and_idle_status_explains_first_task_test_scope(tmp_path: Path) -> None:
    """Catches the retired status command or an opaque idle status returning."""
    from guardedpy.tui import GuardedPyApp
    from textual.widgets import RichLog

    profile = _profile(tmp_path)
    app = GuardedPyApp(_Runtime(profile), profile)

    async def check() -> None:
        async with app.run_test() as pilot:
            app.submit("/status")
            await pilot.pause()
            assert "已就绪 · 尚未提交任务 · 首个任务将运行完整测试" in str(
                app.query_one("#status").render()
            )
            assert app.query_one("#transcript", RichLog).lines[-1].text == "未知命令。"

    asyncio.run(check())


@pytest.mark.parametrize(
    "command",
    ("/new unexpected", "/credentials unexpected", "/help unexpected"),
)
def test_tui_rejects_trailing_arguments_for_no_argument_commands(
    tmp_path: Path, command: str
) -> None:
    """Catches a typo after a no-argument command changing the interactive session."""
    from guardedpy.tui import GuardedPyApp
    from textual.widgets import RichLog

    profile = _profile(tmp_path)
    app = GuardedPyApp(_Runtime(profile), profile)

    async def check() -> None:
        async with app.run_test() as pilot:
            transcript = app.query_one("#transcript", RichLog)
            transcript.write("保留的会话记录")
            app.submit(command)
            await pilot.pause()

            assert transcript.lines[0].text == "保留的会话记录"
            assert transcript.lines[-1].text == "未知命令。"

    asyncio.run(check())


def test_tui_approval_modal_shows_only_safe_projection(tmp_path: Path) -> None:
    """Catches the approval dialog exposing an action hash or raw model payload."""
    from guardedpy.tui import GuardedPyApp

    profile = _profile(tmp_path)
    app = GuardedPyApp(_Runtime(profile), profile)
    task = TaskState(description="remove generated output", config=HarnessConfig(profile=profile))

    async def check() -> None:
        async with app.run_test() as pilot:
            app.request_approval(
                task,
                action_projection="删除项目内文件",
                rule_id="delete.approval_required",
                raw_action_hash="must-not-render",
            )
            await pilot.pause()
            projection = app.screen.query_one("#approval-projection")
            rule = app.screen.query_one("#approval-rule")
            assert "删除项目内文件" in str(projection.render())
            assert "delete.approval_required" in str(rule.render())
            assert "must-not-render" not in str(projection.render())
            with pytest.raises(NoMatches, match="approval-always"):
                app.screen.query_one("#approval-always")

    asyncio.run(check())


def test_tui_rejected_approval_closes_modal_and_renders_the_blocked_task(tmp_path: Path) -> None:
    """Catches a valid rejection being treated like an invalid approval response."""
    from guardedpy.tui import GuardedPyApp

    class RejectingRuntime(_Runtime):
        def resolve_approval(self, task_id: object, action_hash: str, decision: object) -> bool:
            assert task_id == task.id
            assert action_hash == "approval-hash"
            assert decision == "reject"
            task.status = TaskStatus.BLOCKED
            return False

    profile = _profile(tmp_path)
    task = TaskState(description="reject delete", config=HarnessConfig(profile=profile))
    task.status = TaskStatus.WAITING_APPROVAL
    app = GuardedPyApp(RejectingRuntime(profile), profile)

    async def check() -> None:
        async with app.run_test() as pilot:
            app._active_task = task
            app.request_approval(
                task,
                action_projection="删除项目内文件",
                rule_id="delete.approval_required",
                raw_action_hash="approval-hash",
            )
            await pilot.pause()
            await pilot.click("#approval-reject")
            await pilot.pause()

            assert app._active_task is None
            assert "任务：blocked" in str(app.query_one("#status").render())

    asyncio.run(check())


def test_tui_runs_blocking_approval_continuation_off_loop_and_keeps_cancel_reachable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Catches approval resolution or its resumed run freezing the Textual event loop."""
    from guardedpy.tui import GuardedPyApp

    class BlockingApprovalOrchestrator:
        def __init__(self) -> None:
            self.task: TaskState | None = None
            self.resolution_started = Event()
            self.release = Event()
            self.cancel_before_release = False

        def submit(self, task: TaskState) -> TaskState:
            self.task = task
            return task

        def run(self, task: TaskState) -> TaskState:
            self.release.wait()
            return task

        def cancel(self, task_id: object) -> TaskState:
            assert self.task is not None
            assert task_id == self.task.id
            self.cancel_before_release = not self.release.is_set()
            self.task.status = TaskStatus.CANCELLED
            self.release.set()
            return self.task

        def resolve_approval(self, task_id: object, action_hash: str, *, decision: object) -> bool:
            assert self.task is not None
            assert task_id == self.task.id
            assert action_hash == "approval-hash"
            assert decision == "once"
            self.task.status = TaskStatus.RUNNING
            self.resolution_started.set()
            self.release.wait()
            return True

    profile = _profile(tmp_path)
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    orchestrator = BlockingApprovalOrchestrator()
    runtime = LocalRuntime(
        RuntimeServices(
            credentials=_Credentials(),
            orchestrator_factory=lambda root, config, memory: orchestrator,
        )
    )
    runtime.setup(profile, api_key=None)
    task = runtime.create_task("approve command", TaskIntent.CODING)
    task.status = TaskStatus.WAITING_APPROVAL
    app = GuardedPyApp(runtime, profile)

    async def check() -> None:
        timer = Timer(1, orchestrator.release.set)
        timer.start()
        try:
            async with app.run_test() as pilot:
                app._active_task = task
                app.request_approval(
                    task,
                    action_projection="运行已批准命令",
                    rule_id="command.approval_required",
                    raw_action_hash="approval-hash",
                    permanent_eligible=True,
                )
                await pilot.pause()
                await pilot.click("#approval-once")
                await pilot.pause()
                assert orchestrator.resolution_started.is_set()
                assert orchestrator.release.is_set() is False

                await pilot.press("ctrl+c")
                assert orchestrator.cancel_before_release is True
                app.submit("/help")
                await pilot.pause()
                assert "项目：" in str(app.query_one("#status").render())
        finally:
            timer.cancel()
            orchestrator.release.set()

    asyncio.run(check())


def test_tui_palette_selection_fills_then_second_enter_executes_and_model_picker_updates_defaults(
    tmp_path: Path,
) -> None:
    """Catches palette selection executing early or settings ignoring keyboard selection."""
    from guardedpy.tui import Composer, GuardedPyApp, SettingsScreen

    profile = _profile(tmp_path)
    runtime = _Runtime(profile)
    app = GuardedPyApp(runtime, profile)

    async def check() -> None:
        async with app.run_test() as pilot:
            await pilot.click("#composer")
            await pilot.press("/", "down", "enter")
            composer = app.query_one("#composer", Composer)
            assert composer.text == "/history"
            assert runtime.history_reads == 0
            await pilot.press("enter")
            assert runtime.history_reads == 1

            app.submit("/model")
            await pilot.pause()
            await pilot.press("down", "enter")
            await pilot.pause()
            assert runtime.updated == {"model": "deepseek-v4-pro"}
            assert composer.text == ""
            assert not isinstance(app.screen, SettingsScreen)

    asyncio.run(check())


def test_tui_has_grouped_help_without_footer(tmp_path: Path) -> None:
    """Catches the accepted help surface retaining framework Footer chrome."""
    from guardedpy.tui import GuardedPyApp
    from textual.css.query import NoMatches
    from textual.widgets import Footer, RichLog

    profile = _profile(tmp_path)
    app = GuardedPyApp(_Runtime(profile), profile)

    async def check() -> None:
        async with app.run_test() as pilot:
            with pytest.raises(NoMatches):
                app.query_one(Footer)
            app.submit("/help")
            await pilot.pause()
            rendered = "\n".join(line.text for line in app.query_one("#transcript", RichLog).lines)
            for group in ("会话与对话", "任务与检查", "设置与安全"):
                assert group in rendered
            assert "/status" not in rendered

    asyncio.run(check())


def test_tui_credentials_command_uses_a_masked_modal_input(tmp_path: Path) -> None:
    """Catches the interactive credential path falling back to visible transcript text."""
    from guardedpy.tui import GuardedPyApp
    from textual.widgets import Input

    profile = _profile(tmp_path)
    app = GuardedPyApp(_Runtime(profile), profile)

    async def check() -> None:
        async with app.run_test() as pilot:
            app.submit("/credentials")
            await pilot.pause()
            field = app.screen.query_one("#credential-value", Input)
            assert field.password is True

    asyncio.run(check())


def test_tui_requires_masked_credential_before_creating_llm_task(tmp_path: Path) -> None:
    """Catches a provider task being registered before its interactive credential exists."""
    from guardedpy.tui import GuardedPyApp
    from textual.widgets import Input, RichLog

    profile = _profile(tmp_path)
    runtime = _Runtime(profile)
    runtime.configured = False
    app = GuardedPyApp(runtime, profile)

    async def check() -> None:
        async with app.run_test() as pilot:
            app.submit("repair failing test")
            await pilot.pause()
            assert runtime.created == []
            assert app.screen.query_one("#credential-value", Input).password is True
            field = app.screen.query_one("#credential-value", Input)
            field.value = "test-key"
            await pilot.click("#credential-update")
            await pilot.pause()
            assert [task.description for task in runtime.created] == ["repair failing test"]

            runtime.configured = False
            app.submit("/plan inspect")
            await pilot.pause()
            await pilot.click("#credential-cancel")
            await pilot.pause()
            transcript = app.query_one("#transcript", RichLog)
            assert transcript.lines[-1].text == "未配置凭据，任务未开始。"

    asyncio.run(check())


def test_tui_credential_clear_requires_a_second_confirmation(tmp_path: Path) -> None:
    """Catches one accidental clear button press deleting the stored credential."""
    from guardedpy.tui import GuardedPyApp

    profile = _profile(tmp_path)
    app = GuardedPyApp(_Runtime(profile), profile)

    async def check() -> None:
        async with app.run_test() as pilot:
            app.submit("/credentials")
            await pilot.pause()
            await pilot.click("#credential-clear")
            await pilot.pause()
            assert app.screen.query_one("#credential-clear-confirm")

    asyncio.run(check())


def test_demo_selector_presents_fixed_request_then_runs_selected_scenario(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Catches the demo hiding its fixed request or ignoring the selected Task 16 scenario."""
    from guardedpy.mechanism_demo import ScenarioResult
    from guardedpy.tui import DemoApp

    calls: list[str] = []

    def runner(name: str) -> ScenarioResult:
        calls.append(name)
        return ScenarioResult(name, "completed", None, None, False, (), "fixed")

    monkeypatch.setattr("guardedpy.tui.run_scenario", runner)
    app = DemoApp()

    async def check() -> None:
        async with app.run_test() as pilot:
            await pilot.press("down")
            assert "Correct the selected assertion failure." in str(
                app.query_one("#demo-request").render()
            )
            await pilot.press("enter")
            assert calls == ["failure_feedback_corrects"]

    asyncio.run(check())
