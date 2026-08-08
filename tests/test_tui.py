"""Focused contracts for the retained GuardedPy continuous TUI."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from pathlib import Path
from uuid import UUID, uuid4

import pytest
from textual.css.query import NoMatches
from textual.widgets import Button, Input, Static, TextArea

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


def _transcript_lines(app: object) -> tuple[str, ...]:
    from guardedpy.tui import TranscriptLog

    return app.query_one("#transcript", TranscriptLog).text_entries  # type: ignore[attr-defined]


def test_composer_keeps_enter_submit_multiline_and_command_palette(tmp_path: Path) -> None:
    from guardedpy.tui import COMMANDS, Composer, GuardedPyApp, _HELP_LINES

    profile = _profile(tmp_path)
    app = GuardedPyApp(_Runtime(profile), profile)

    async def check() -> None:
        async with app.run_test() as pilot:
            composer = app.query_one("#composer", Composer)
            await pilot.click("#composer")
            await pilot.press(*"/help", "enter")
            help_content = app.screen.query_one("#help-content", Static)
            assert str(help_content.render()) == "\n\n".join(_HELP_LINES)
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
            ] == ["/help"]
            await pilot.click("#command-help")
            assert composer.text == "/help"
            await pilot.click("#mode-picker")
            assert composer.text == "/"
            assert app.query_one("#command-palette").display is True
            assert str(app.query_one("#mode-picker", Button).label) == "[+]"
            assert str(app.query_one("#send", Button).label) == "[发送]"
            assert tuple(COMMANDS) == (
                "/conversations", "/new", "/delete", "/exit", "/plan", "/review",
                "/tests", "/diff", "/permissions", "/credentials", "/model", "/effort",
                "/doctor", "/goal", "/help", "/stop", "/queue",
            )

    asyncio.run(check())


def test_composer_layout_keeps_status_left_and_reserves_no_right_input_gutter(tmp_path: Path) -> None:
    from guardedpy.tui import Composer, GuardedPyApp

    profile = _profile(tmp_path)
    app = GuardedPyApp(_Runtime(profile), profile)

    async def check() -> None:
        async with app.run_test(size=(120, 24)) as pilot:
            composer = app.query_one("#composer", Composer)
            plus = app.query_one("#mode-picker", Button)
            model = app.query_one("#composer-model", Button)
            await pilot.pause()
            assert plus.region.x < app.size.width // 2 < model.region.x
            assert composer.styles.padding.right == 1

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


def test_tui_settings_picker_starts_at_the_current_value(tmp_path: Path) -> None:
    from textual.widgets import ListView

    from guardedpy.tui import GuardedPyApp

    profile = _profile(tmp_path)
    runtime = _Runtime(profile)
    runtime.config = runtime.config.model_copy(update={"reasoning_effort": "max"})
    app = GuardedPyApp(runtime, profile)

    async def check() -> None:
        async with app.run_test() as pilot:
            app.submit("/effort")
            await pilot.pause()
            assert app.screen.query_one("#settings-picker", ListView).index == 1

    asyncio.run(check())


def test_first_request_resumes_after_masked_credential_entry(tmp_path: Path) -> None:
    from guardedpy.conversation import SessionEvent
    from guardedpy.tui import GuardedPyApp

    profile = _profile(tmp_path)
    runtime = _Runtime(profile)
    runtime.configured = False
    session_id, turn_id = uuid4(), uuid4()

    class Conversation:
        def create_session(self, title: str) -> UUID:
            return session_id

        def begin_turn(self, session: UUID, text: str, mode: str) -> tuple[UUID, SessionEvent]:
            assert (session, text, mode) == (session_id, "hello", "normal")
            return turn_id, SessionEvent(session, turn_id, 1, "user_message", uuid4(), text)

        def run_turn(self, session: UUID, turn: UUID) -> tuple[SessionEvent, ...]:
            return (SessionEvent(session, turn, 2, "turn_completed"),)

    app = GuardedPyApp(runtime, profile, conversation=Conversation())

    async def check() -> None:
        async with app.run_test() as pilot:
            app.submit("hello")
            await pilot.pause()
            await pilot.click("#credential-value")
            await pilot.press(*"secret")
            await pilot.click("#credential-update")
            await pilot.pause()
            await app.workers.wait_for_complete()
            assert runtime.configured is True
            assert "› hello" in _transcript_lines(app)

    asyncio.run(check())


def test_first_request_explains_an_unavailable_keyring_without_starting_a_turn(tmp_path: Path) -> None:
    from guardedpy.credentials import CredentialBackendUnavailableError
    from guardedpy.tui import GuardedPyApp

    class UnavailableRuntime(_Runtime):
        def credential_status(self) -> CredentialStatus:
            raise CredentialBackendUnavailableError("unavailable")

    profile = _profile(tmp_path)
    app = GuardedPyApp(UnavailableRuntime(profile), profile)

    async def check() -> None:
        async with app.run_test() as pilot:
            app.submit("修复项目")
            await pilot.pause()
            assert "安全系统密钥环" in str(
                app.screen.query_one("#credential-backend-unavailable").render()
            )
            assert "无法启动会话。" not in _transcript_lines(app)

    asyncio.run(check())


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

    assert (first.text, first.replace) == ("one ", True)
    assert (second.text, second.replace) == ("one two", True)
    assert tool is None


def test_transcript_presenter_projects_safe_changed_path_and_pytest_feedback() -> None:
    from guardedpy.conversation import SessionEvent
    from guardedpy.tui import TranscriptPresenter

    presenter = TranscriptPresenter()
    session_id, turn_id = uuid4(), uuid4()
    changed = presenter.present(SessionEvent(
        session_id, turn_id, 1, "tool_item_completed", uuid4(),
        data={"tool": "apply_patch", "changed_paths": "[\"src/calc.py\"]"},
    ))
    tested = presenter.present(SessionEvent(
        session_id, turn_id, 2, "tool_item_completed", uuid4(),
        data={"tool": "run_pytest", "pytest_outcome": "passed"},
    ))

    assert changed is not None and changed.text == "已修改 src/calc.py。"
    assert tested is not None and tested.text == "pytest：通过。"


def test_transcript_presenter_projects_a_governed_tool_target_without_tool_payload() -> None:
    from guardedpy.conversation import SessionEvent
    from guardedpy.tui import TranscriptPresenter

    presenter = TranscriptPresenter()
    event = SessionEvent(
        uuid4(), uuid4(), 1, "tool_item_started", uuid4(),
        data={"tool": "read_file", "path": "src/calc.py"},
    )

    update = presenter.present(event)

    assert update is not None and update.text == "正在查看项目文件…"


def test_transcript_presenter_explains_safe_tool_completion_and_denial() -> None:
    from guardedpy.conversation import SessionEvent
    from guardedpy.tui import TranscriptPresenter

    presenter = TranscriptPresenter()
    session_id, turn_id = uuid4(), uuid4()
    read = presenter.present(SessionEvent(
        session_id, turn_id, 1, "tool_item_completed", uuid4(),
        data={"tool": "read_file", "path": "src/calc.py", "code": "ok"},
    ))
    denied = presenter.present(SessionEvent(
        session_id, turn_id, 2, "tool_item_completed", uuid4(),
        data={"tool": "apply_patch", "code": "read_required", "verdict": "deny"},
    ))
    missing = presenter.present(SessionEvent(
        session_id, turn_id, 3, "tool_item_completed", uuid4(),
        data={"tool": "apply_patch", "code": "read_required", "missing_paths": '["src/calc.py"]'},
    ))
    program = presenter.present(SessionEvent(
        session_id, turn_id, 4, "tool_item_completed", uuid4(),
        data={"tool": "run_python", "code": "ok", "program_output": "hello world\n"},
    ))
    failed = presenter.present(SessionEvent(session_id, turn_id, 5, "turn_failed"))

    assert read is not None and read.text == "已查看 1 个文件：src/calc.py。"
    assert denied is not None and denied.text == "无法修改：需先完整读取目标文件。"
    assert missing is not None and missing.text == "无法修改：需先完整读取 src/calc.py。"
    assert program is not None and program.text == "程序输出：\nhello world"
    assert failed is None


def test_transcript_presenter_groups_reads_flattens_finished_markdown_and_hides_success_terminal() -> None:
    from guardedpy.conversation import SessionEvent
    from guardedpy.tui import TranscriptPresenter

    presenter = TranscriptPresenter()
    session_id, turn_id = uuid4(), uuid4()
    first_read, second_read, assistant = uuid4(), uuid4(), uuid4()

    start = presenter.present(SessionEvent(
        session_id, turn_id, 1, "tool_item_started", first_read, data={"tool": "read_file", "path": "src/a.py"},
    ))
    first_done = presenter.present(SessionEvent(
        session_id, turn_id, 2, "tool_item_completed", first_read, data={"tool": "read_file", "path": "src/a.py", "code": "ok"},
    ))
    presenter.present(SessionEvent(
        session_id, turn_id, 3, "tool_item_started", second_read, data={"tool": "read_file", "path": "src/b.py"},
    ))
    second_done = presenter.present(SessionEvent(
        session_id, turn_id, 4, "tool_item_completed", second_read, data={"tool": "read_file", "path": "src/b.py", "code": "ok"},
    ))
    presenter.present(SessionEvent(session_id, turn_id, 5, "assistant_text_delta", assistant, "## 结果\\n**已修复**"))
    formatted = presenter.present(SessionEvent(session_id, turn_id, 6, "assistant_item_completed", assistant))
    completed = presenter.present(SessionEvent(session_id, turn_id, 7, "turn_completed"))

    assert start is not None and start.text == "正在查看项目文件…"
    assert first_done is not None and first_done.replace is True
    assert first_done.text == "已查看 1 个文件：src/a.py。"
    assert second_done is not None and second_done.replace is True
    assert second_done.text == "已查看 2 个文件：src/a.py、src/b.py。"
    assert formatted is not None and formatted.replace is True and formatted.text == "结果\\n已修复"
    assert completed is None


def test_transcript_inserts_one_blank_line_before_a_later_user_turn(tmp_path: Path) -> None:
    from guardedpy.conversation import SessionEvent
    from guardedpy.tui import GuardedPyApp

    profile = _profile(tmp_path)
    app = GuardedPyApp(_Runtime(profile), profile)
    session_id, turn_id = uuid4(), uuid4()

    async def check() -> None:
        async with app.run_test() as pilot:
            app._present_session_event(SessionEvent(session_id, turn_id, 1, "assistant_text_delta", uuid4(), "已修复。"))
            app._present_session_event(SessionEvent(session_id, uuid4(), 1, "user_message", uuid4(), "继续扩展"))
            await pilot.pause()
            assert _transcript_lines(app) == ("已修复。", "", "› 继续扩展")

    asyncio.run(check())


def test_transcript_is_read_only_selectable_text(tmp_path: Path) -> None:
    from guardedpy.tui import GuardedPyApp, TranscriptLog

    profile = _profile(tmp_path)
    app = GuardedPyApp(_Runtime(profile), profile)

    async def check() -> None:
        async with app.run_test() as pilot:
            transcript = app.query_one("#transcript", TranscriptLog)
            transcript.write("Agent 的安全回复。")
            await pilot.pause()
            assert transcript.read_only is True
            assert transcript.soft_wrap is True
            transcript.select_all()
            assert transcript.selected_text == "Agent 的安全回复。"

    asyncio.run(check())


def test_tui_reopens_the_most_recent_saved_conversation_on_startup(tmp_path: Path) -> None:
    from guardedpy.conversation import ConversationSummary, VisibleTranscriptEntry
    from guardedpy.tui import GuardedPyApp

    profile = _profile(tmp_path)
    session_id = uuid4()
    now = datetime.now(timezone.utc)
    summary = ConversationSummary(
        id=session_id, project_title=str(profile.root), created_at=now, updated_at=now,
        transcript=(VisibleTranscriptEntry(role="user", text="上次的问题"),), turns=(),
    )

    class Store:
        def summaries(self) -> tuple[object, ...]:
            return (summary,)

    class Conversation:
        store = Store()

        def create_session(self, title: str, selected: UUID | None = None) -> UUID:
            assert title == str(profile.root)
            assert selected == session_id
            return session_id

        def summary(self, received: UUID) -> object:
            assert received == session_id
            return summary

    app = GuardedPyApp(_Runtime(profile), profile, conversation=Conversation())

    async def check() -> None:
        async with app.run_test() as pilot:
            await pilot.pause()
            assert app._continuous_session_id == session_id
            assert _transcript_lines(app) == ("› 上次的问题",)

    asyncio.run(check())


def test_tui_can_copy_the_safe_transcript_with_a_shortcut(tmp_path: Path) -> None:
    from guardedpy.tui import GuardedPyApp

    profile = _profile(tmp_path)
    app = GuardedPyApp(_Runtime(profile), profile)

    async def check() -> None:
        async with app.run_test() as pilot:
            app._write("可复制的安全记录")
            await pilot.press("ctrl+shift+c")
            assert app.clipboard == "可复制的安全记录"

    asyncio.run(check())


def test_demo_surface_offers_only_a_read_only_fixed_mock_request() -> None:
    from guardedpy.mechanism_demo import scenario_request
    from guardedpy.tui import DemoApp, DemoRequest

    app = DemoApp()

    async def check() -> None:
        async with app.run_test() as pilot:
            request = app.query_one("#composer", DemoRequest)
            assert str(request.render()) == scenario_request("delete_requires_approval")
            assert str(app.query_one("#mode-picker", Button).label) == "[+]"
            assert str(app.query_one("#send", Button).label) == "[发送]"
            assert str(app.query_one("#demo-hint", Static).render()) == "按↑↓来切换场景"
            assert str(app.query_one("#status", Static).render()) == "项目：机制演示临时项目"
            assert str(app.query_one("#composer-model", Button).label) == "Mock LLM1"
            assert str(app.query_one("#composer-effort", Button).label) == "high"
            await pilot.click("#composer-model")
            await pilot.click("#setting-mock-llm2")
            assert str(app.query_one("#composer-model", Button).label) == "Mock LLM2"
            await pilot.click("#composer-effort")
            await pilot.click("#setting-max")
            assert str(app.query_one("#composer-effort", Button).label) == "max"
            await pilot.click("#mode-picker")
            assert app.query_one("#command-palette").display is True
            await pilot.click("#command-conversations")
            await pilot.pause()
            assert app.query_one("#command-palette").display is False
            assert str(request.render()) == scenario_request("delete_requires_approval")
            await pilot.press("down")
            assert str(request.render()) == scenario_request("feedback_repair")

    asyncio.run(check())


def test_demo_surface_replays_the_governed_mock_scenario_after_enter() -> None:
    from guardedpy.tui import ApprovalScreen, DemoApp, TranscriptLog

    app = DemoApp()

    async def check() -> None:
        async with app.run_test() as pilot:
            await pilot.press("enter")
            for _ in range(20):
                await pilot.pause(0.1)
                if isinstance(app.screen, ApprovalScreen):
                    break
            assert isinstance(app.screen, ApprovalScreen)
            await pilot.pause()
            await pilot.click("#approval-once")
            for _ in range(20):
                await pilot.pause(0.1)
                if not app._scenario_running:
                    break
            assert app._scenario_running is False
            lines = app.query_one("#transcript", TranscriptLog).text_entries
            assert lines[0] == "› 删除 src/value.py。"
            assert "需要批准：删除 src/value.py。" in lines
            assert "审批已批准。" in lines
            assert "已删除 src/value.py。" in lines
            assert "已根据你的审批决定完成处理。" in lines

    asyncio.run(check())


def test_tui_applies_goal_to_exactly_the_next_continuous_turn(tmp_path: Path) -> None:
    from guardedpy.conversation import ConversationAgent, ProviderMessage, ResponseFinished, ScriptedConversationModel, TextDelta
    from guardedpy.conversations import ConversationStore
    from guardedpy.runtime import ConversationRuntime
    from guardedpy.tui import GuardedPyApp

    profile = _profile(tmp_path)
    model = ScriptedConversationModel([[TextDelta("done"), ResponseFinished("stop")]])
    conversation = ConversationRuntime(ConversationAgent(model), ConversationStore(profile.root))
    app = GuardedPyApp(_Runtime(profile), profile, conversation=conversation)

    async def check() -> None:
        async with app.run_test() as pilot:
            app.submit("/goal 保持改动最小")
            app.submit("修复计算器")
            await app.workers.wait_for_complete()
            await pilot.pause()
            assert ProviderMessage(
                role="system", content="Current turn goal: 保持改动最小"
            ) in model.received_messages[0]

    asyncio.run(check())


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
            assert _transcript_lines(app)[0] == "› hello"
            await pilot.pause()
            await app.workers.wait_for_complete()
            assert _transcript_lines(app) == ("› hello", "hello there")

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
                "approval_id": str(approval_id), "tool": "delete_path", "path": "src/obsolete.py", "rule_id": "delete.approval"
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
            assert "src/obsolete.py" in str(app.screen.query_one("#approval-projection").render())
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


def test_tui_keeps_the_promoted_fifo_turn_controllable(tmp_path: Path) -> None:
    from guardedpy.conversation import SessionEvent
    from guardedpy.tui import GuardedPyApp, SessionEventReceived

    profile = _profile(tmp_path)
    first, promoted = uuid4(), uuid4()
    app = GuardedPyApp(_Runtime(profile), profile)
    app._continuous_turn_id = first

    async def check() -> None:
        async with app.run_test() as pilot:
            app.on_session_event_received(SessionEventReceived(SessionEvent(uuid4(), first, 1, "turn_completed")))
            app.on_session_event_received(SessionEventReceived(SessionEvent(uuid4(), promoted, 1, "turn_started")))
            await pilot.pause()
            assert app._continuous_turn_id == promoted

    asyncio.run(check())


def test_tui_projects_a_replayed_session_event_once(tmp_path: Path) -> None:
    from guardedpy.conversation import SessionEvent
    from guardedpy.tui import GuardedPyApp, SessionEventReceived

    profile = _profile(tmp_path)
    app = GuardedPyApp(_Runtime(profile), profile)
    event = SessionEvent(uuid4(), uuid4(), 1, "user_message", uuid4(), "hello")

    async def check() -> None:
        async with app.run_test() as pilot:
            app.on_session_event_received(SessionEventReceived(event))
            app.on_session_event_received(SessionEventReceived(event))
            await pilot.pause()
            assert _transcript_lines(app) == ("› hello",)

    asyncio.run(check())


def test_tui_explains_that_plan_needs_a_request(tmp_path: Path) -> None:
    from guardedpy.tui import GuardedPyApp

    profile = _profile(tmp_path)
    app = GuardedPyApp(_Runtime(profile), profile)

    async def check() -> None:
        async with app.run_test() as pilot:
            app.submit("/plan")
            await pilot.pause()
            assert "计划任务不能为空。" in _transcript_lines(app)

    asyncio.run(check())


def test_conversation_picker_replays_the_selected_visible_transcript(tmp_path: Path) -> None:
    from guardedpy.conversation import ConversationSummary, SafeTurnSummary, VisibleTranscriptEntry
    from guardedpy.tui import GuardedPyApp

    summary_id, session_id = uuid4(), uuid4()
    summary = ConversationSummary(
        id=summary_id, project_title="project", created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc), transcript=(
            VisibleTranscriptEntry(role="user", text="之前的问题"),
            VisibleTranscriptEntry(role="assistant", text="之前的回答"),
        ), turns=(SafeTurnSummary(
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
            rendered = "\n".join(_transcript_lines(app))
            assert "› 之前的问题" in rendered
            assert "之前的回答" in rendered
            assert "已完成安全摘要" not in rendered

    asyncio.run(check())


def test_delete_reopens_the_previous_conversation_and_keeps_the_last_one(tmp_path: Path) -> None:
    from guardedpy.conversation import ConversationSummary, VisibleTranscriptEntry
    from guardedpy.tui import GuardedPyApp

    previous_id, current_id = uuid4(), uuid4()
    now = datetime.now(timezone.utc)
    previous = ConversationSummary(
        id=previous_id, project_title="project", created_at=now, updated_at=now,
        transcript=(
            VisibleTranscriptEntry(role="user", text="previous"),
            VisibleTranscriptEntry(role="assistant", text="answer"),
        ), turns=(),
    )

    class Store:
        def summaries(self) -> tuple[object, ...]:
            return (previous,)

    class Conversation:
        store = Store()

        def delete_session(self, received: UUID) -> object | None:
            assert received == current_id
            return previous

        def create_session(self, title: str, selected: UUID | None = None) -> UUID:
            assert title == str(profile.root)
            assert selected == previous_id
            return previous_id

        def summary(self, received: UUID) -> object:
            assert received == previous_id
            return previous

    profile = _profile(tmp_path)
    app = GuardedPyApp(_Runtime(profile), profile, conversation=Conversation())

    async def check() -> None:
        async with app.run_test() as pilot:
            app._continuous_session_id = current_id
            app.submit("/delete")
            await pilot.pause()
            assert app._continuous_session_id == previous_id
            assert _transcript_lines(app) == ("› previous", "answer")

    asyncio.run(check())
