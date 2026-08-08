"""Focused non-TTY contracts for the continuous conversation path."""

from __future__ import annotations

from io import StringIO
from uuid import uuid4

from guardedpy.conversation import SessionEvent


def test_plain_conversation_renders_continuous_events_and_stops_for_approval() -> None:
    from guardedpy.terminal import run_plain_conversation

    session_id, turn_id, user_item, assistant_item, approval_item = uuid4(), uuid4(), uuid4(), uuid4(), uuid4()

    class Runtime:
        def create_session(self, title: str) -> object:
            assert title == "project"
            return session_id

        def begin_turn(self, session: object, text: str, mode: str) -> tuple[object, SessionEvent]:
            assert (session, text, mode) == (session_id, "repair", "normal")
            return turn_id, SessionEvent(session_id, turn_id, 1, "user_message", user_item, text)

        def run_turn(self, session: object, turn: object) -> tuple[SessionEvent, ...]:
            return (
                SessionEvent(session_id, turn_id, 2, "assistant_text_delta", assistant_item, "Checking."),
                SessionEvent(session_id, turn_id, 3, "approval_requested", approval_item, data={"approval_id": str(uuid4())}),
            )

    output = StringIO()
    assert run_plain_conversation(Runtime(), "project", StringIO("repair\n"), output) == 1
    assert output.getvalue().splitlines() == [
        "› repair", "Checking.", "需要精确审批，非交互模式已安全停止。"
    ]


def test_plain_conversation_projects_safe_tool_facts_and_failures() -> None:
    from guardedpy.terminal import _render_event

    session_id, turn_id = uuid4(), uuid4()
    output = StringIO()
    _render_event(SessionEvent(
        session_id, turn_id, 1, "tool_item_completed", uuid4(),
        data={"tool": "read_file", "path": "src/calc.py", "code": "ok"},
    ), output)
    _render_event(SessionEvent(
        session_id, turn_id, 2, "tool_item_completed", uuid4(),
        data={"tool": "apply_patch", "code": "stale_read", "verdict": "deny"},
    ), output)

    assert output.getvalue().splitlines() == [
        "已读取 src/calc.py。", "修改未执行：目标文件已变化，请重新读取。"
    ]


def test_plain_conversation_keeps_help_and_settings_local() -> None:
    from guardedpy.credentials import CredentialStatus
    from guardedpy.terminal import run_plain_conversation

    class LocalRuntime:
        def credential_status(self) -> CredentialStatus:
            return CredentialStatus(configured=False)

        def update_future_defaults(self, **changes: str) -> None:
            self.changes = changes

    class Runtime:
        def create_session(self, title: str) -> object:
            return uuid4()

        def begin_turn(self, *args: object) -> object:
            raise AssertionError("local commands must not reach the model")

    local = LocalRuntime()
    output = StringIO()
    assert run_plain_conversation(
        Runtime(), "project", StringIO("/help\n/credentials\n/model deepseek-v4-pro\n/exit\n"), output,
        local_runtime=local,
    ) == 0
    rendered = output.getvalue()
    assert "/conversations：列出已保存会话。" in rendered
    assert "凭据：未配置" in rendered
    assert local.changes == {"model": "deepseek-v4-pro"}


def test_plain_local_controls_are_deterministic_and_do_not_reach_the_model() -> None:
    from guardedpy.terminal import run_plain_conversation

    class LocalRuntime:
        def local_check(self, name: str) -> str:
            return {"tests": "pytest：passed", "diff": "Git diff：无变更", "doctor": "诊断：凭据已配置"}[name]

    class Runtime:
        def create_session(self, title: str) -> object:
            return "session"

        def begin_turn(self, *args: object) -> object:
            raise AssertionError("local controls must not reach the model")

        def summary(self, session: object) -> object:
            return type("Summary", (), {"turns": ()})()

    output = StringIO()
    assert run_plain_conversation(
        Runtime(), "project", StringIO("/tests\n/diff\n/permissions\n/doctor\n/exit\n"), output,
        local_runtime=LocalRuntime(),
    ) == 0
    assert output.getvalue().splitlines() == [
        "pytest：passed", "Git diff：无变更",
        "权限：项目内读取、补丁、pytest 与只读 Git 自动允许；删除须逐次审批。",
        "诊断：凭据已配置",
    ]
