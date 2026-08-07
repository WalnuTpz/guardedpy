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
from guardedpy.domain import FeedbackKind, TaskIntent, TaskState, TaskStatus
from guardedpy.events import StoredRunEvent
from guardedpy.runtime import LocalRuntime, RuntimeServices


@pytest.fixture(autouse=True)
def _isolated_state_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep Textual sessions that create a safe conversation index out of the host state home."""
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))


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

    def task(self, task_id: object) -> TaskState:
        return next(task for task in self.created if task.id == task_id)

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

    def create_task(
        self, description: str, intent: object, review_path: str | None = None,
        session_goal: str | None = None,
    ) -> TaskState:
        task = TaskState(
            description=description, config=self.config, intent=intent,
            review_path=review_path, session_goal=session_goal,
        )
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
    from textual.widgets import Log, Log

    profile = _profile(tmp_path)
    app = GuardedPyApp(_Runtime(profile), profile)

    async def check() -> None:
        async with app.run_test() as pilot:
            await pilot.click("#composer")
            await pilot.press(*"/help", "enter")
            assert "会话与对话" in "\n".join(app.screen.query_one("#help-content", Log).lines)
            await pilot.click("#help-close")
            await pilot.click("#composer")
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
                "/history", "/conversations", "/new", "/clear", "/exit", "/plan", "/review",
                "/tests", "/diff", "/permissions", "/credentials", "/memory",
                "/model", "/effort", "/goal", "/doctor", "/help",
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
            status = str(app.query_one("#status").render())
            assert str(profile.root) in status
            assert "baseline" not in status
            assert "waiting_approval" not in status
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
    from textual.widgets import Log

    profile = _profile(tmp_path)
    app = GuardedPyApp(_Runtime(profile), profile)

    async def check() -> None:
        async with app.run_test() as pilot:
            app._write("first safe projection")
            app._write("second safe projection")
            transcript = app.query_one("#transcript", Log)
            assert transcript.lines[:2] == ["first safe projection", "second safe projection"]
            app.submit("repair value")
            await pilot.pause()
            await app.workers.wait_for_complete()
            assert transcript.lines[2] == "› repair value"

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
        timer = Timer(1, orchestrator.release.set)
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
    from textual.widgets import Log

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
                transcript = app.query_one("#transcript", Log)
                assert any("完整测试：passed" in line for line in transcript.lines)
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
    from textual.widgets import Log

    profile = _profile(tmp_path)
    app = GuardedPyApp(_Runtime(profile), profile)

    async def check() -> None:
        async with app.run_test() as pilot:
            app.submit("/status")
            await pilot.pause()
            assert "首个任务将运行完整测试" not in str(
                app.query_one("#status").render()
            )
            assert app.query_one("#transcript", Log).lines[-2] == "未知命令。"

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
    from textual.widgets import Log

    profile = _profile(tmp_path)
    app = GuardedPyApp(_Runtime(profile), profile)

    async def check() -> None:
        async with app.run_test() as pilot:
            transcript = app.query_one("#transcript", Log)
            transcript.write("保留的会话记录")
            app.submit(command)
            await pilot.pause()

            assert transcript.lines[0] == "保留的会话记录"
            assert transcript.lines[-2] == "未知命令。"

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
    from textual.widgets import Log

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
            assert "GuardedPy：任务已阻止。" in "\n".join(app.query_one("#transcript", Log).lines)

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


def test_tui_message_flow_renders_safe_incremental_history_and_one_live_task_status(
    tmp_path: Path,
) -> None:
    """Catches a delayed lifecycle dump, duplicate events, or raw audit leakage."""
    from guardedpy.tui import GuardedPyApp
    from textual.widgets import Log, Static

    class BlockingLiveRuntime(_Runtime):
        def __init__(self, profile: ProjectProfile) -> None:
            super().__init__(profile)
            self.started = Event()
            self.release = Event()
            self._events: list[StoredRunEvent] = []

        def events(self, task_id: object) -> list[StoredRunEvent]:
            assert self.created and task_id == self.created[0].id
            return list(self._events)

        def run(self, task_id: object) -> TaskState:
            task = next(task for task in self.created if task.id == task_id)
            task.status = TaskStatus.RUNNING
            self.started.set()
            self.release.wait(timeout=2)
            task.status = TaskStatus.COMPLETED
            return task

    profile = _profile(tmp_path)
    runtime = BlockingLiveRuntime(profile)
    app = GuardedPyApp(runtime, profile)

    async def check() -> None:
        async with app.run_test() as pilot:
            app.submit("repair value")
            await pilot.pause()
            assert runtime.started.is_set()
            transcript = app.query_one("#transcript", Log)
            live_status = app.query_one("#live-task-status", Static)
            assert "› repair value" in transcript.lines
            assert live_status.display is True
            initial_live_status = str(live_status.render())

            task = runtime.created[0]
            runtime._events.append(
                StoredRunEvent(
                    id=1,
                    task_id=task.id,
                    task_status=TaskStatus.RUNNING,
                    action_summary="run configured tests",
                    feedback_kind=FeedbackKind.PASSED,
                    action_projection="raw-secret-marker",
                )
            )
            await asyncio.sleep(0.2)
            history = "\n".join(transcript.lines)
            assert history.count("运行配置测试") == 1
            assert history.count("pytest passed") == 1
            assert "raw-secret-marker" not in history
            assert str(live_status.render()) != initial_live_status
            assert live_status.display is True

            runtime.release.set()
            await app.workers.wait_for_complete()
            await pilot.pause()
            assert live_status.display is False
            assert str(live_status.render()) == ""
            assert "GuardedPy：任务完成。" in "\n".join(transcript.lines)
            status = str(app.query_one("#status", Static).render())
            assert str(profile.root) in status
            for removed_detail in ("模型", "effort", "基线", "任务"):
                assert removed_detail not in status

    asyncio.run(check())


def test_tui_transcript_history_replays_safe_completed_messages_without_live_task_status(
    tmp_path: Path,
) -> None:
    """Catches restored conversations rebuilding a lifecycle dump or stale live row."""
    from guardedpy.conversations import ConversationStore
    from guardedpy.tui import GuardedPyApp
    from textual.widgets import Log, Static

    class HistoryRuntime(_Runtime):
        def __init__(self, profile: ProjectProfile) -> None:
            super().__init__(profile)
            self.completed = TaskState(description="repair value", config=self.config)
            self.completed.status = TaskStatus.COMPLETED
            self._events = [
                StoredRunEvent(
                    id=1,
                    task_id=self.completed.id,
                    task_status=TaskStatus.COMPLETED,
                    action_summary="run configured tests",
                    feedback_kind=FeedbackKind.PASSED,
                    action_projection="raw-secret-marker",
                )
            ]

        def task(self, task_id: object) -> TaskState:
            assert task_id == self.completed.id
            return self.completed

        def events(self, task_id: object) -> list[StoredRunEvent]:
            assert task_id == self.completed.id
            return list(self._events)

    profile = _profile(tmp_path)
    runtime = HistoryRuntime(profile)
    store = ConversationStore(profile.root)
    conversation = store.create()
    store.attach_task(conversation.id, runtime.completed.id)
    app = GuardedPyApp(runtime, profile)

    async def check() -> None:
        async with app.run_test():
            app._conversation_selected(conversation.id)
            transcript = app.query_one("#transcript", Log)
            history = "\n".join(transcript.lines)
            assert "› repair value" in history
            assert history.count("运行配置测试") == 1
            assert history.count("pytest passed") == 1
            assert "GuardedPy：任务完成。" in history
            assert "raw-secret-marker" not in history
            live_status = app.query_one("#live-task-status", Static)
            assert live_status.display is False
            assert str(live_status.render()) == ""

    asyncio.run(check())


def test_tui_palette_wheel_and_click_are_fill_first_before_enter(tmp_path: Path) -> None:
    """Catches pointer palette selection executing a command or ignoring wheel navigation."""
    from guardedpy.tui import Composer, GuardedPyApp
    from textual import events

    profile = _profile(tmp_path)
    runtime = _Runtime(profile)
    app = GuardedPyApp(runtime, profile)

    async def check() -> None:
        async with app.run_test() as pilot:
            composer = app.query_one("#composer", Composer)
            await pilot.click("#composer")
            await pilot.press("/")
            composer.post_message(events.MouseScrollDown(composer, 0, 0, 0, 1, 0, False, False, False))
            await pilot.pause()
            assert app.query_one("#command-palette").index == 0
            composer.post_message(events.MouseScrollUp(composer, 0, 0, 0, -1, 0, False, False, False))
            await pilot.pause()
            assert app.query_one("#command-palette").index == len(app.query("#command-palette ListItem")) - 1

            composer.text = "/h"
            await pilot.pause()
            await pilot.click("#command-history")
            await pilot.pause()
            assert composer.text == "/history"
            assert runtime.history_reads == 0
            await pilot.press("enter")
            assert runtime.history_reads == 1

    asyncio.run(check())


def test_tui_new_confirmation_keyboard_click_and_reject_control_cancellation(tmp_path: Path) -> None:
    """Catches /new cancelling without consent or failing to cancel the exact active task."""
    from guardedpy.tui import GuardedPyApp
    from textual.widgets import Log

    profile = _profile(tmp_path)
    runtime = _Runtime(profile)
    active = TaskState(description="active", config=runtime.config)
    runtime.created.append(active)
    app = GuardedPyApp(runtime, profile)

    async def check() -> None:
        async with app.run_test() as pilot:
            app._active_task = active
            app.query_one("#transcript", Log).write("old")
            app.submit("/new")
            await pilot.pause()
            await pilot.click("#new-cancel")
            await pilot.pause()
            assert runtime.cancelled == []

            app.submit("/new")
            await pilot.pause()
            await pilot.press("enter")
            await pilot.pause()
            assert runtime.cancelled == [active.id]
            assert app._conversation_id is None
            assert app.query_one("#transcript", Log).lines == []

            clicked_active = TaskState(description="clicked", config=runtime.config)
            runtime.created.append(clicked_active)
            app._active_task = clicked_active
            app.submit("/new")
            await pilot.pause()
            await pilot.click("#new-confirm-button")
            await pilot.pause()
            assert runtime.cancelled == [active.id, clicked_active.id]
            assert app._conversation_id is None
            assert app.query_one("#transcript", Log).lines == []

    asyncio.run(check())


def test_tui_effort_picker_supports_keyboard_and_click(tmp_path: Path) -> None:
    """Catches effort selection diverging from model picker behavior for keyboard or mouse."""
    from guardedpy.tui import Composer, GuardedPyApp, SettingsScreen

    profile = _profile(tmp_path)
    keyboard_runtime = _Runtime(profile)
    keyboard_app = GuardedPyApp(keyboard_runtime, profile)
    click_runtime = _Runtime(profile)
    click_app = GuardedPyApp(click_runtime, profile)

    async def check() -> None:
        async with keyboard_app.run_test() as pilot:
            keyboard_app.submit("/effort")
            await pilot.press("down", "enter")
            await pilot.pause()
            assert keyboard_runtime.updated == {"reasoning_effort": "max"}
            assert keyboard_app.query_one("#composer", Composer).text == ""
            assert not isinstance(keyboard_app.screen, SettingsScreen)
        async with click_app.run_test() as pilot:
            click_app.submit("/effort")
            await pilot.pause()
            await pilot.click("#setting-max")
            await pilot.pause()
            assert click_runtime.updated == {"reasoning_effort": "max"}
            assert click_app.query_one("#composer", Composer).text == ""
            assert not isinstance(click_app.screen, SettingsScreen)

    asyncio.run(check())


def test_tui_has_grouped_help_without_footer(tmp_path: Path) -> None:
    """Catches the accepted help surface retaining framework Footer chrome."""
    from guardedpy.tui import GuardedPyApp
    from textual.css.query import NoMatches
    from textual.widgets import Footer, Log

    profile = _profile(tmp_path)
    app = GuardedPyApp(_Runtime(profile), profile)

    async def check() -> None:
        async with app.run_test() as pilot:
            with pytest.raises(NoMatches):
                app.query_one(Footer)
            app.submit("/help")
            await pilot.pause()
            rendered = "\n".join(app.screen.query_one("#help-content", Log).lines)
            for group in ("会话与对话", "任务与检查", "设置与安全"):
                assert group in rendered
            assert "/status" not in rendered

    asyncio.run(check())


def test_tui_help_is_scrollable_and_send_button_matches_enter(tmp_path: Path) -> None:
    """Catches a terse help transcript or a pointer send path diverging from Enter."""
    from guardedpy.tui import GuardedPyApp
    from textual.widgets import Button, Log

    profile = _profile(tmp_path)
    runtime = _Runtime(profile)
    app = GuardedPyApp(runtime, profile)

    async def check() -> None:
        async with app.run_test() as pilot:
            send = app.query_one("#send", Button)
            assert send.disabled is True
            app.submit("/help")
            await pilot.pause()
            help_content = app.screen.query_one("#help-content", Log)
            assert help_content.allow_select is True
            assert "非交互" in "\n".join(help_content.lines)

            await pilot.click("#help-close")
            composer = app.query_one("#composer")
            composer.text = "send by click"
            await pilot.pause()
            assert send.disabled is False
            await pilot.click("#send")
            await pilot.pause()
            assert runtime.created[0].description == "send by click"
            assert "首个任务将运行完整测试" not in str(app.query_one("#status").render())

    asyncio.run(check())


def test_tui_send_button_is_inside_composer_and_matches_enter(tmp_path: Path) -> None:
    """Catches the send control consuming a row or falling outside the composer bottom-right."""
    from guardedpy.tui import Composer, GuardedPyApp
    from textual.widgets import Button

    profile = _profile(tmp_path)
    runtime = _Runtime(profile)
    app = GuardedPyApp(runtime, profile)

    async def check() -> None:
        async with app.run_test() as pilot:
            composer = app.query_one("#composer", Composer)
            send = app.query_one("#send", Button)
            shell = app.query_one("#composer-shell")
            assert composer.parent is shell and send.parent.parent is shell
            assert send.region.x >= composer.region.x + composer.region.width - send.region.width
            assert send.region.y >= composer.region.y + composer.region.height - send.region.height
            assert send.region.right <= composer.region.right
            assert send.region.bottom <= composer.region.bottom
            assert send.disabled is True
            composer.text = "click sends"
            await pilot.pause()
            assert send.disabled is False
            await pilot.click("#send")
            await pilot.pause()
            assert runtime.created[0].description == "click sends"

    asyncio.run(check())


def test_tui_composer_modes_dispatch_once_and_keep_goal_ephemeral(tmp_path: Path) -> None:
    """Catches a decorative mode picker or Goal reaching task, credential, or transcript state."""
    from guardedpy.tui import Composer, GuardedPyApp
    from textual.css.query import NoMatches
    from textual.widgets import Button, ListItem, Static

    profile = _profile(tmp_path)
    runtime = _Runtime(profile)
    app = GuardedPyApp(runtime, profile)

    async def check() -> None:
        async with app.run_test() as pilot:
            shell = app.query_one("#composer-shell")
            assert shell.styles.height.value == 6
            assert app.query_one("#mode-chip", Static).display is False
            mode_picker = app.query_one("#mode-picker", Button)
            assert mode_picker.styles.height.value == 1
            assert mode_picker.styles.color is not None
            assert runtime.config.model in str(app.query_one("#composer-model", Button).render())
            assert runtime.config.reasoning_effort in str(app.query_one("#composer-effort", Button).render())

            await pilot.click("#mode-picker")
            await pilot.pause()
            assert [str(item.query_one(Static).render()) for item in app.screen.query(ListItem)] == [
                "计划", "审查", "目标"
            ]
            await pilot.press("enter")
            await pilot.pause()
            assert str(app.query_one("#mode-chip", Static).render()) == "[计划]"
            app.submit("draft migration")
            await pilot.pause()
            assert runtime.created[-1].intent is TaskIntent.PLAN
            assert app.query_one("#mode-chip", Static).display is False
            app._cancel_active_task()

            await pilot.click("#mode-picker")
            await pilot.pause()
            await pilot.click("#mode-review")
            await pilot.pause()
            app.submit("src/app.py")
            await pilot.pause()
            assert runtime.created[-1].intent is TaskIntent.REVIEW
            assert runtime.created[-1].review_path == "src/app.py"
            app._cancel_active_task()

            runtime.configured = False
            await pilot.click("#mode-picker")
            await pilot.pause()
            await pilot.press("down", "down", "enter")
            await pilot.pause()
            assert str(app.query_one("#mode-chip", Static).render()) == "[目标]"
            before_goal = len(runtime.created)
            app.submit("release checklist")
            await pilot.pause()
            assert len(runtime.created) == before_goal
            assert "release checklist" not in "\n".join(app.query_one("#transcript").lines)
            with pytest.raises(NoMatches):
                app.screen.query_one("#credential-value")
            runtime.configured = True
            app.submit("repair value")
            await pilot.pause()
            assert runtime.created[-1].session_goal == "release checklist"
            app._cancel_active_task()

            app.submit("/goal clear")
            await pilot.pause()
            assert app.query_one("#mode-chip", Static).display is False
            app.submit("/goal release checklist")
            await pilot.pause()
            assert str(app.query_one("#mode-chip", Static).render()) == "[目标]"
            app.submit("/new")
            await pilot.pause()
            assert app.query_one("#mode-chip", Static).display is False
            assert isinstance(app.query_one("#composer"), Composer)

    asyncio.run(check())


def test_tui_transcript_is_selectable_safe_log_and_unavailable_keyring_never_opens_input(tmp_path: Path) -> None:
    """Catches an unselectable transcript or an unsafe credential prompt without secure storage."""
    from guardedpy.credentials import CredentialBackendUnavailableError
    from guardedpy.tui import GuardedPyApp
    from textual.widgets import Input, Log
    from textual.css.query import NoMatches

    class UnavailableRuntime(_Runtime):
        def credential_status(self) -> CredentialStatus:
            raise CredentialBackendUnavailableError("keyring backend is unavailable")

    profile = _profile(tmp_path)
    app = GuardedPyApp(UnavailableRuntime(profile), profile)

    async def check() -> None:
        async with app.run_test() as pilot:
            transcript = app.query_one("#transcript", Log)
            assert transcript.allow_select is True
            app._write("safe projection")
            assert "raw-marker" not in "\n".join(transcript.lines)
            app.submit("/credentials")
            await pilot.pause()
            assert "安全系统密钥环" in str(app.screen.query_one("#credential-backend-unavailable").render())
            with pytest.raises(NoMatches):
                app.screen.query_one("#credential-value", Input)

    asyncio.run(check())


def test_tui_conversation_selection_restores_cli_history_and_new_confirms_active_cancel(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Catches history losing its safe user entry or /new abandoning active work."""
    from guardedpy.conversations import ConversationStore
    from guardedpy.tui import Composer, GuardedPyApp
    from textual.widgets import Log

    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    profile = _profile(tmp_path)
    runtime = _Runtime(profile)
    task = TaskState(description="repair value", config=runtime.config)
    task.status = TaskStatus.COMPLETED
    runtime.created.append(task)
    conversation = ConversationStore(profile.root).create()
    ConversationStore(profile.root).attach_task(conversation.id, task.id)
    app = GuardedPyApp(runtime, profile)

    async def check() -> None:
        async with app.run_test() as pilot:
            app.submit("/conversations")
            await pilot.pause()
            await pilot.press("enter")
            await pilot.pause()
            transcript = app.query_one("#transcript", Log)
            rendered = "\n".join(line for line in transcript.lines)
            assert "› repair value" in rendered
            assert "GuardedPy：任务完成。" in rendered
            composer = app.query_one("#composer", Composer)
            assert app.focused is composer
            assert composer.cursor_location == composer.document.end

            app._active_task = TaskState(description="active", config=runtime.config)
            app.submit("/new")
            await pilot.pause()
            assert app.screen.query_one("#new-confirm")

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
    from textual.widgets import Input, Log

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
            transcript = app.query_one("#transcript", Log)
            assert transcript.lines[-2] == "未配置凭据，任务未开始。"

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
