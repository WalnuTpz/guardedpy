"""Focused contracts for the retained GuardedPy continuous TUI."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from pathlib import Path
from uuid import UUID, uuid4

import pytest
from textual.css.query import NoMatches
from textual.widgets import Button, Input, Log, Static

from guardedpy.config import HarnessConfig
from guardedpy.credentials import CredentialStatus
from guardedpy.discovery import ProjectProfile, discover_project


@pytest.fixture(autouse=True)
def _isolated_state_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))


class _Runtime:
    def __init__(self, profile: ProjectProfile) -> None:
        self.project_root = profile.root
        self.config = HarnessConfig(profile=profile)
        self.configured = True

    def credential_status(self) -> CredentialStatus:
        return CredentialStatus(configured=self.configured)

    def update_credential(self, key: str) -> None:
        assert key
        self.configured = True

    def clear_credential(self) -> None:
        self.configured = False

    def update_future_defaults(self, **changes: str) -> HarnessConfig:
        self.updated = changes
        self.config = self.config.model_copy(update=changes)
        return self.config


def _profile(tmp_path: Path) -> ProjectProfile:
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "app.py").write_text("VALUE = 1\n")
    (tmp_path / "tests").mkdir()
    return discover_project(tmp_path)


def test_composer_keeps_enter_submit_multiline_and_command_palette(tmp_path: Path) -> None:
    from guardedpy.tui import COMMANDS, Composer, GuardedPyApp

    profile = _profile(tmp_path)
    app = GuardedPyApp(_Runtime(profile), profile)

    async def check() -> None:
        async with app.run_test() as pilot:
            composer = app.query_one("#composer", Composer)
            await pilot.click("#composer")
            await pilot.press(*"/help", "enter")
            assert "会话与对话" in "\n".join(app.screen.query_one("#help-content", Log).lines)
            await pilot.click("#help-close")
            await pilot.click("#composer")
            await pilot.press("shift+enter")
            assert "\n" in composer.text
            composer.text = ""
            await pilot.press("ctrl+j")
            assert composer.text == "\n"
            composer.text = "/h"
            await pilot.pause()
            assert [
                item.query_one(Static).render()
                for item in app.query("#command-palette ListItem")
                if item.display
            ] == ["/history", "/help"]
            await pilot.click("#command-history")
            assert composer.text == "/history"
            assert tuple(COMMANDS) == (
                "/history", "/conversations", "/new", "/clear", "/exit", "/plan", "/review",
                "/credentials", "/model", "/effort", "/goal", "/help", "/stop", "/queue",
            )

    asyncio.run(check())


def test_tui_settings_and_credentials_remain_keyboard_and_secret_safe(tmp_path: Path) -> None:
    from guardedpy.credentials import CredentialBackendUnavailableError
    from guardedpy.tui import GuardedPyApp

    profile = _profile(tmp_path)
    runtime = _Runtime(profile)
    app = GuardedPyApp(runtime, profile)

    async def check() -> None:
        async with app.run_test() as pilot:
            app.submit("/effort")
            await pilot.press("down", "enter")
            await pilot.pause()
            assert runtime.updated == {"reasoning_effort": "max"}
            app.submit("/credentials")
            await pilot.pause()
            assert app.screen.query_one("#credential-value", Input).password is True

    asyncio.run(check())

    class UnavailableRuntime(_Runtime):
        def credential_status(self) -> CredentialStatus:
            raise CredentialBackendUnavailableError("unavailable")

    blocked = GuardedPyApp(UnavailableRuntime(profile), profile)

    async def blocked_check() -> None:
        async with blocked.run_test() as pilot:
            blocked.submit("/credentials")
            await pilot.pause()
            assert "安全系统密钥环" in str(blocked.screen.query_one("#credential-backend-unavailable").render())
            with pytest.raises(NoMatches):
                blocked.screen.query_one("#credential-value", Input)

    asyncio.run(blocked_check())


def test_transcript_presenter_coalesces_streaming_text_and_hides_tool_payloads() -> None:
    from guardedpy.conversation import SessionEvent
    from guardedpy.tui import TranscriptPresenter

    session_id, turn_id, user_item_id, item_id = uuid4(), uuid4(), uuid4(), uuid4()
    presenter = TranscriptPresenter()
    first = presenter.present(SessionEvent(session_id, turn_id, 1, "assistant_text_delta", item_id, "one "))
    second = presenter.present(SessionEvent(session_id, turn_id, 2, "assistant_text_delta", item_id, "two"))
    tool = presenter.present(
        SessionEvent(session_id, turn_id, 3, "tool_output", item_id, "secret", {"detail": "secret"})
    )

    assert (first.text, first.replace) == ("助手：one ", True)
    assert (second.text, second.replace) == ("助手：one two", True)
    assert tool.text == "工具已返回受限结果。"
    assert "secret" not in tool.text


def test_tui_renders_continuous_turns_immediately_and_streams_updates(tmp_path: Path) -> None:
    from guardedpy.conversation import SessionEvent
    from guardedpy.tui import GuardedPyApp

    session_id, turn_id, user_item_id, item_id = uuid4(), uuid4(), uuid4(), uuid4()

    class Conversation:
        def create_session(self, title: str) -> UUID:
            assert title == str(profile.root)
            return session_id

        def begin_turn(self, session: UUID, text: str, mode: str) -> tuple[UUID, SessionEvent]:
            assert (session, text, mode) == (session_id, "hello", "normal")
            return turn_id, SessionEvent(session_id, turn_id, 1, "user_message", user_item_id, "hello")

        def run_turn(self, session: UUID, turn: UUID) -> tuple[SessionEvent, ...]:
            assert (session, turn) == (session_id, turn_id)
            return (
                SessionEvent(session_id, turn_id, 2, "assistant_text_delta", item_id, "hello "),
                SessionEvent(session_id, turn_id, 3, "assistant_text_delta", item_id, "there"),
                SessionEvent(session_id, turn_id, 4, "turn_completed"),
            )

        def interrupt(self, session: UUID, turn: UUID) -> SessionEvent:
            return SessionEvent(session, turn, 5, "turn_interrupted")

    profile = _profile(tmp_path)
    app = GuardedPyApp(_Runtime(profile), profile, conversation=Conversation())

    async def check() -> None:
        async with app.run_test() as pilot:
            app.submit("hello")
            assert app.query_one("#transcript", Log).lines[0] == "› hello"
            await pilot.pause()
            await app.workers.wait_for_complete()
            assert app.query_one("#transcript", Log).lines[:-1] == [
                "› hello", "助手：hello there", "本轮回复已完成。"
            ]

    asyncio.run(check())


def test_tui_controls_approval_queue_stop_and_new_session(tmp_path: Path) -> None:
    from guardedpy.conversation import SessionEvent
    from guardedpy.tui import ApprovalScreen, GuardedPyApp

    session_id, next_session, turn_id, approval_id, queued_id = uuid4(), uuid4(), uuid4(), uuid4(), uuid4()

    class Conversation:
        def __init__(self) -> None:
            self.created = 0
            self.resolutions: list[bool] = []
            self.queues: list[str] = []
            self.interrupts: list[tuple[UUID, UUID]] = []

        def create_session(self, title: str, summary_id: UUID | None = None) -> UUID:
            del title, summary_id
            self.created += 1
            return session_id if self.created == 1 else next_session

        def begin_turn(self, session: UUID, text: str, mode: str) -> tuple[UUID, SessionEvent]:
            return turn_id, SessionEvent(session, turn_id, 1, "user_message", uuid4(), text)

        def run_turn(self, session: UUID, turn: UUID) -> tuple[SessionEvent, ...]:
            return (SessionEvent(session, turn, 2, "approval_requested", uuid4(), data={
                "approval_id": str(approval_id), "tool": "delete_path", "rule_id": "delete.approval"
            }),)

        def resolve_approval(self, session: UUID, turn: UUID, approval: UUID, accepted: bool) -> tuple[SessionEvent, ...]:
            self.resolutions.append(accepted)
            return (SessionEvent(session, turn, 3, "turn_completed"),)

        def queue(self, session: UUID, text: str, mode: str) -> tuple[UUID, SessionEvent]:
            self.queues.append(text)
            return queued_id, SessionEvent(session, queued_id, 1, "user_message", uuid4(), text)

        def interrupt(self, session: UUID, turn: UUID) -> SessionEvent:
            self.interrupts.append((session, turn))
            return SessionEvent(session, turn, 4, "turn_interrupted")

    profile = _profile(tmp_path)
    conversation = Conversation()
    app = GuardedPyApp(_Runtime(profile), profile, conversation=conversation)

    async def check() -> None:
        async with app.run_test() as pilot:
            app.submit("remove old file")
            await pilot.pause()
            assert isinstance(app.screen, ApprovalScreen)
            await pilot.click("#approval-reject")
            await pilot.pause()
            assert conversation.resolutions == [False]
            app._continuous_session_id = session_id
            app._continuous_turn_id = turn_id
            app.submit("/queue later")
            assert conversation.queues == ["later"]
            app.submit("/new")
            await pilot.pause()
            await pilot.click("#new-confirm-button")
            await pilot.pause()
            assert conversation.interrupts == [(session_id, turn_id)]
            assert app._continuous_session_id == next_session

    asyncio.run(check())


def test_conversation_picker_restores_only_safe_continuous_summary(tmp_path: Path) -> None:
    from guardedpy.conversation import ConversationSummary, SafeTurnSummary
    from guardedpy.tui import GuardedPyApp

    summary_id, session_id = uuid4(), uuid4()
    summary = ConversationSummary(
        id=summary_id, project_title="project", created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc), turns=(SafeTurnSummary(
            terminal_status="completed", changed_paths=("src/private.py",), pytest_outcome="passed",
            approval_outcome="none", final_text="已完成安全摘要",
        ),),
    )

    class Store:
        def summaries(self) -> tuple[object, ...]:
            return (summary,)

    class Conversation:
        store = Store()

        def create_session(self, title: str, selected: UUID | None = None) -> UUID:
            assert title == str(profile.root)
            assert selected == summary_id
            return session_id

        def summary(self, received: UUID) -> object:
            assert received == session_id
            return summary

    profile = _profile(tmp_path)
    app = GuardedPyApp(_Runtime(profile), profile, conversation=Conversation())

    async def check() -> None:
        async with app.run_test() as pilot:
            app.submit("/conversations")
            await pilot.press("enter")
            await pilot.pause()
            rendered = "\n".join(app.query_one("#transcript", Log).lines)
            assert "助手：已完成安全摘要" in rendered
            assert "本轮回复已完成。" in rendered
            assert "src/private.py" not in rendered

    asyncio.run(check())
