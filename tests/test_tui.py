"""Textual contracts for the safe GuardedPy terminal session."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
from textual.css.query import NoMatches
from guardedpy.config import HarnessConfig
from guardedpy.credentials import CredentialStatus
from guardedpy.discovery import ProjectProfile
from guardedpy.domain import TaskState, TaskStatus
from guardedpy.events import StoredRunEvent


class _Runtime:
    """The smallest real-session boundary needed to exercise Textual widgets."""

    def __init__(self, profile: ProjectProfile) -> None:
        self.project_root = profile.root
        self.config = HarnessConfig(profile=profile)
        self.created: list[TaskState] = []
        self.cancelled: list[object] = []

    def events(self, task_id: object) -> list[StoredRunEvent]:
        del task_id
        return []

    def update_future_defaults(self, **changes: str) -> HarnessConfig:
        self.updated = changes
        self.config = self.config.model_copy(update=changes)
        return self.config

    def credential_status(self) -> CredentialStatus:
        return CredentialStatus(configured=False)

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


def _profile(tmp_path: Path) -> ProjectProfile:
    from guardedpy.discovery import discover_project

    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "app.py").write_text("VALUE = 1\n")
    (tmp_path / "tests").mkdir()
    return discover_project(tmp_path)


def test_tui_mounts_safe_session_widgets_and_filters_exact_command_palette(tmp_path: Path) -> None:
    """Catches the interactive session losing its composer, status, or safe command UI."""
    from guardedpy.tui import COMMANDS, GuardedPyApp
    from textual.widgets import TextArea

    profile = _profile(tmp_path)
    app = GuardedPyApp(_Runtime(profile), profile)

    async def check() -> None:
        async with app.run_test() as pilot:
            assert "项目：" in str(app.query_one("#status").render())
            assert isinstance(app.query_one("#composer"), TextArea)
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
            assert app.query_one("#composer", TextArea).text == "/history"
            assert tuple(COMMANDS) == (
                "/new", "/clear", "/history", "/exit", "/plan", "/review",
                "/tests", "/diff", "/permissions", "/credentials", "/memory",
                "/model", "/effort", "/status", "/doctor", "/help",
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
            assert "基线：baseline_pending · 任务：waiting_approval" in str(
                app.query_one("#status").render()
            )
            await pilot.press("ctrl+c")
            assert runtime.cancelled == [runtime.created[0].id]
            app.submit("repair value")
            await pilot.pause()
            app.submit("/exit")
            await pilot.pause()
            assert app.screen.query_one("#exit-confirm")

    asyncio.run(check())


def test_tui_status_command_renders_without_using_a_removed_widget_property(tmp_path: Path) -> None:
    """Catches `/status` crashing after Textual removes an internal Static attribute."""
    from guardedpy.tui import GuardedPyApp

    profile = _profile(tmp_path)
    app = GuardedPyApp(_Runtime(profile), profile)

    async def check() -> None:
        async with app.run_test() as pilot:
            app.submit("/status")
            await pilot.pause()

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


def test_tui_model_command_updates_only_future_defaults(tmp_path: Path) -> None:
    """Catches the visible model picker becoming a nonfunctional status message."""
    from guardedpy.tui import GuardedPyApp

    profile = _profile(tmp_path)
    runtime = _Runtime(profile)
    app = GuardedPyApp(runtime, profile)

    async def check() -> None:
        async with app.run_test() as pilot:
            app.submit("/model deepseek-v4-pro")
            await pilot.pause()
            assert runtime.updated == {"model": "deepseek-v4-pro"}

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
