"""Textual contracts for the safe GuardedPy terminal session."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
from textual.css.query import NoMatches
from guardedpy.config import HarnessConfig
from guardedpy.credentials import CredentialStatus
from guardedpy.discovery import ProjectProfile
from guardedpy.domain import TaskState
from guardedpy.events import StoredRunEvent


class _Runtime:
    """The smallest real-session boundary needed to exercise Textual widgets."""

    def __init__(self, profile: ProjectProfile) -> None:
        self.project_root = profile.root
        self.config = HarnessConfig(profile=profile)

    def events(self, task_id: object) -> list[StoredRunEvent]:
        del task_id
        return []

    def update_future_defaults(self, **changes: str) -> HarnessConfig:
        self.updated = changes
        self.config = self.config.model_copy(update=changes)
        return self.config

    def credential_status(self) -> CredentialStatus:
        return CredentialStatus(configured=False)


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
            assert tuple(COMMANDS) == (
                "/new", "/clear", "/history", "/exit", "/plan", "/review",
                "/tests", "/diff", "/permissions", "/credentials", "/memory",
                "/model", "/effort", "/status", "/doctor", "/help",
            )

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
