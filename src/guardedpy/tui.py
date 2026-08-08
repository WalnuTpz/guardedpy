"""Textual session surfaces backed by GuardedPy's deterministic runtime."""

from __future__ import annotations

from dataclasses import dataclass, field
import re
from threading import Event as ThreadEvent, Thread
from time import sleep
from typing import Any, Literal
from uuid import UUID

from rich.text import Text
from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical
from textual.css.query import NoMatches
from textual import events
from textual.message import Message
from textual.screen import ModalScreen
from textual.widgets import Button, Input, ListItem, ListView, Static, TextArea

from guardedpy.conversation import SessionEvent, TurnNotActiveError, safe_event_message
from guardedpy.credentials import CredentialBackendUnavailableError
from guardedpy.mechanism_demo import ScenarioName, ScenarioResult, run_scenario, scenario_request


COMMANDS = (
    "/conversations", "/new", "/delete", "/exit", "/plan", "/review",
    "/tests", "/diff", "/permissions", "/credentials", "/model", "/effort",
    "/doctor", "/goal", "/help", "/stop", "/queue",
)

_HELP_LINES = (
    "GuardedPy 使用指南",
    "直接输入：与 Agent 对话，或描述要检查、修复、实现的项目任务。",
    "/new：新建会话。",
    "/conversations：选择并恢复历史对话。",
    "/delete：删除当前对话并回到上一条；最后一条不能删除。",
    "/exit：退出 GuardedPy。",
    "/plan <任务>：只读制定计划；提交后自动回到普通模式。",
    "/review <路径>：只读审查；提交后自动回到普通模式。",
    "/goal <目标>：只约束下一回合；/goal clear 取消。",
    "/stop：中断当前回合并取消排队任务。",
    "/queue <任务>：把下一项工作排到当前回合之后。",
    "/tests：运行配置的 pytest。",
    "/diff：查看当前 Git diff。",
    "/doctor：查看本地项目状态。",
    "/permissions：查看可自动执行与需审批的操作。",
    "/credentials：安全录入、更新或清除 API Key。",
    "/model：选择后续回合模型。",
    "/effort：选择后续回合思考强度。",
    "/help：打开本帮助面板。",
    "输入：Enter 提交；Shift+Enter 或 Ctrl+J 换行；输入 / 或点击 ＋ 打开命令选择；选择命令后再按 Enter 执行；Ctrl+Shift+C 复制安全对话记录。",
    "安全：读取、补丁、pytest 和只读 Git 在项目边界内自动执行；删除始终逐次请求批准；非交互终端会安全停止，不会代替你批准。",
)


_NO_ARGUMENT_COMMANDS = frozenset(
    {
        "/new", "/delete", "/conversations", "/exit", "/credentials",
        "/help", "/model", "/effort", "/stop", "/tests", "/diff", "/permissions",
        "/doctor",
    }
)


class ComposerSubmitted(Message):
    """Deliver a composer submission that TextArea would otherwise consume."""

    def __init__(self, text: str) -> None:
        self.text = text
        super().__init__()


@dataclass(frozen=True)
class TranscriptUpdate:
    """One safe, renderer-agnostic transcript change."""

    item_id: UUID | None
    text: str
    replace: bool = False


@dataclass
class TranscriptPresenter:
    """Project continuous-session events to text safe for a transcript."""

    _assistant_text: dict[UUID, str] = field(default_factory=dict)
    _read_batch_item_id: UUID | None = None
    _read_paths: list[str] = field(default_factory=list)

    def present(self, event: SessionEvent) -> TranscriptUpdate | None:
        """Return the visible update for one event without exposing tool payloads."""
        if event.kind == "user_message":
            self._finish_read_batch()
            return TranscriptUpdate(event.item_id, f"› {event.text}")
        if event.kind == "assistant_text_delta":
            self._finish_read_batch()
            if event.item_id is None:
                return None
            text = self._assistant_text.get(event.item_id, "") + event.text
            self._assistant_text[event.item_id] = text
            return TranscriptUpdate(event.item_id, text, replace=True)
        if event.kind == "assistant_item_completed":
            self._finish_read_batch()
            if event.item_id is None:
                return None
            text = self._assistant_text.get(event.item_id)
            return None if text is None else TranscriptUpdate(
                event.item_id, _display_assistant_text(text), replace=True
            )
        if event.data.get("tool") == "read_file":
            return self._present_read(event)
        self._finish_read_batch()
        status = safe_event_message(event)
        if status is None:
            return None
        return TranscriptUpdate(event.item_id, status)

    def _present_read(self, event: SessionEvent) -> TranscriptUpdate | None:
        if event.kind == "tool_item_started":
            if self._read_batch_item_id is None:
                self._read_batch_item_id = event.item_id
                return TranscriptUpdate(event.item_id, "正在查看项目文件…")
            return None
        if event.kind != "tool_item_completed" or event.data.get("code") != "ok":
            self._finish_read_batch()
            status = safe_event_message(event)
            return None if status is None else TranscriptUpdate(event.item_id, status)
        path = event.data.get("path")
        if isinstance(path, str) and path not in self._read_paths:
            self._read_paths.append(path)
        batch_id = self._read_batch_item_id or event.item_id
        count = len(self._read_paths)
        preview = "、".join(self._read_paths[:3])
        if count > 3:
            preview += " 等"
        return TranscriptUpdate(batch_id, f"已查看 {count} 个文件：{preview}。", replace=True)

    def _finish_read_batch(self) -> None:
        self._read_batch_item_id = None
        self._read_paths.clear()


def _display_assistant_text(text: str) -> str:
    """Keep completed assistant prose compact without rendering provider Markdown."""
    text = text.replace("\r\n", "\n")
    text = re.sub(r"(?m)^\s{0,3}#{1,6}\s+", "", text)
    text = re.sub(r"\*\*([^*]+)\*\*", r"\1", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


class TranscriptLog(TextArea):
    """Read-only, soft-wrapping transcript with native mouse selection."""

    def __init__(self, *args: object, **kwargs: object) -> None:
        super().__init__(
            "", *args, **kwargs, language=None, soft_wrap=True, read_only=True,
            show_cursor=False, show_line_numbers=False, highlight_cursor_line=False,
        )
        self._continuous_entries: list[tuple[UUID | None, str]] = []

    @property
    def text_entries(self) -> tuple[str, ...]:
        """Safe source text used by the transcript copy shortcut and UI tests."""
        return tuple(text for _item_id, text in self._continuous_entries)

    def write(self, data: str, scroll_end: bool | None = None) -> "TranscriptLog":
        self._continuous_entries.append((None, str(data).rstrip("\n")))
        self._render_entries()
        return self

    def apply_update(self, update: TranscriptUpdate) -> None:
        """Replace streamed text in place while preserving event order."""
        if update.replace:
            for index, (item_id, _text) in enumerate(self._continuous_entries):
                if item_id == update.item_id:
                    self._continuous_entries[index] = (item_id, update.text)
                    break
            else:
                self._continuous_entries.append((update.item_id, update.text))
        else:
            if update.text.startswith("› ") and self._continuous_entries and self._continuous_entries[-1][1]:
                self._continuous_entries.append((None, ""))
            self._continuous_entries.append((update.item_id, update.text))
        self._render_entries()

    def _render_entries(self) -> None:
        self.load_text("\n".join(text for _item_id, text in self._continuous_entries))
        self.scroll_end(animate=False)

    def reset_continuous(self) -> None:
        """Discard the current session's rendered event projection."""
        self._continuous_entries.clear()
        self.load_text("")


class Composer(TextArea):
    """TextArea with explicit submit and multiline key boundaries."""

    def on_key(self, event: events.Key) -> None:
        if event.key == "enter":
            event.prevent_default()
            event.stop()
            self.post_message(ComposerSubmitted(self.text))
        elif event.key in {"shift+enter", "ctrl+j"}:
            event.prevent_default()
            event.stop()
            self.insert("\n")
        elif event.key in {"up", "down"} and self.app._move_palette(event.key):
            event.prevent_default()
            event.stop()

    def on_mouse_scroll_down(self, event: events.MouseScrollDown) -> None:
        if self.app._move_palette("down"):
            event.stop()

    def on_mouse_scroll_up(self, event: events.MouseScrollUp) -> None:
        if self.app._move_palette("up"):
            event.stop()


class DemoRequest(Static):
    """An intentionally non-editable request that owns demo shortcuts."""

    can_focus = True

    def on_key(self, event: events.Key) -> None:
        if event.key in {"up", "down", "j", "k", "enter", "escape"}:
            event.prevent_default()
            event.stop()
            self.app._demo_key(event.key)


class SettingsScreen(ModalScreen[str | None]):
    """Choose a validated future-task setting with keyboard or mouse."""

    def __init__(
        self,
        field: Literal["model", "reasoning_effort"],
        values: tuple[str, ...],
        current: str,
    ) -> None:
        super().__init__()
        self._field = field
        self._values = values
        self._current = current

    def compose(self) -> ComposeResult:
        yield Vertical(
            Static("选择模型" if self._field.endswith("model") else "选择推理强度"),
            ListView(
                *(ListItem(Static(value), id=f"setting-{value.lower().replace(' ', '-')}") for value in self._values),
                id="settings-picker",
            ),
            id="settings-modal",
        )

    def on_mount(self) -> None:
        picker = self.query_one("#settings-picker", ListView)
        picker.index = self._values.index(self._current)
        picker.focus()

    def on_list_view_selected(self, event: ListView.Selected) -> None:
        self.dismiss(str(event.item.query_one(Static).render()))


class HelpScreen(ModalScreen[None]):
    """Scrollable safe help, kept outside the task transcript."""

    def compose(self) -> ComposeResult:
        yield Vertical(
            Static("\n\n".join(_HELP_LINES), id="help-content"),
            Button("关闭", id="help-close"),
            id="help-modal",
        )

    def on_button_pressed(self, event: Button.Pressed) -> None:
        self.dismiss(None)


class CredentialBackendUnavailableScreen(ModalScreen[None]):
    """Explain why a key cannot safely be accepted without a system keyring."""

    def compose(self) -> ComposeResult:
        yield Vertical(
            Static("安全系统密钥环不可用。请先安装或启动兼容的安全系统密钥环；GuardedPy 不会接受明文 Key。", id="credential-backend-unavailable"),
            Button("关闭", id="credential-backend-close"),
        )

    def on_button_pressed(self, event: Button.Pressed) -> None:
        self.dismiss(None)


class ConversationScreen(ModalScreen[UUID | None]):
    """Select a stored conversation without rendering task bodies in the selector."""

    def __init__(self, conversations: tuple[Any, ...]) -> None:
        super().__init__()
        self._conversations = conversations

    def compose(self) -> ComposeResult:
        yield Vertical(
            ListView(
                *(ListItem(Static(f"{item.id} {item.updated_at.isoformat()}"), id=f"conversation-{item.id}") for item in self._conversations),
                id="conversation-picker",
            ),
            id="conversation-modal",
        )

    def on_mount(self) -> None:
        picker = self.query_one("#conversation-picker", ListView)
        picker.index = 0
        picker.focus()

    def on_list_view_selected(self, event: ListView.Selected) -> None:
        self.dismiss(UUID((event.item.id or "").removeprefix("conversation-")))


class NewConversationScreen(ModalScreen[bool]):
    """Require a cancellation decision before discarding an active conversation view."""

    def compose(self) -> ComposeResult:
        yield Vertical(Static("活跃任务将被取消。确认新建会话？", id="new-confirm"), Button("取消任务并新建", id="new-confirm-button"), Button("继续会话", id="new-cancel"))

    def on_button_pressed(self, event: Button.Pressed) -> None:
        self.dismiss(event.button.id == "new-confirm-button")


class ApprovalScreen(ModalScreen[str | None]):
    """A focused decision surface that receives only a safe action projection."""

    def __init__(self, action_projection: str, rule_id: str, permanent_eligible: bool) -> None:
        super().__init__()
        self._action_projection = action_projection
        self._rule_id = rule_id
        self._permanent_eligible = permanent_eligible

    def compose(self) -> ComposeResult:
        controls = [
            Static("需要审批", id="approval-title"),
            Static(self._action_projection, id="approval-projection"),
            Static(self._rule_id, id="approval-rule"),
            Button("拒绝", id="approval-reject"),
            Button("本次允许", id="approval-once"),
        ]
        if self._permanent_eligible:
            controls.append(Button("保存规则", id="approval-always"))
        yield Vertical(*controls, id="approval-modal")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        self.dismiss({
            "approval-reject": "reject",
            "approval-once": "once",
            "approval-always": "always",
        }[event.button.id or "approval-reject"])


class CredentialScreen(ModalScreen[tuple[str, str | None] | None]):
    """Keep a keyring update in a masked modal instead of any transcript."""

    def __init__(self, configured: bool) -> None:
        super().__init__()
        self._configured = configured

    def compose(self) -> ComposeResult:
        yield Vertical(
            Static(f"凭据：{'已配置' if self._configured else '未配置'}"),
            Input(password=True, placeholder="DeepSeek API Key", id="credential-value"),
            Button("更新", id="credential-update"),
            Button("清除", id="credential-clear"),
            Button("取消", id="credential-cancel"),
            id="credential-modal",
        )

    def on_button_pressed(self, event: Button.Pressed) -> None:
        button_id = event.button.id
        if button_id == "credential-update":
            self.dismiss(("update", self.query_one("#credential-value", Input).value))
        elif button_id == "credential-clear":
            self.dismiss(("clear_requested", None))
        else:
            self.dismiss(None)


class ClearCredentialScreen(ModalScreen[bool]):
    """Require a second, focused decision before clearing a keyring entry."""

    def compose(self) -> ComposeResult:
        yield Vertical(
            Static("确认清除已保存的凭据？", id="credential-clear-confirm"),
            Button("确认清除", id="credential-clear-confirm-button"),
            Button("取消", id="credential-clear-cancel"),
        )

    def on_button_pressed(self, event: Button.Pressed) -> None:
        self.dismiss(event.button.id == "credential-clear-confirm-button")


class ExitScreen(ModalScreen[bool]):
    """Avoid abandoning an active governed task on an accidental exit command."""

    def compose(self) -> ComposeResult:
        yield Vertical(
            Static("活跃任务将被取消。确认退出？", id="exit-confirm"),
            Button("取消任务并退出", id="exit-confirm-button"),
            Button("继续会话", id="exit-cancel"),
        )

    def on_button_pressed(self, event: Button.Pressed) -> None:
        self.dismiss(event.button.id == "exit-confirm-button")


class SessionEventReceived(Message):
    """Deliver one continuous-session event to the Textual event loop."""

    def __init__(self, event: SessionEvent) -> None:
        self.event = event
        super().__init__()


class GuardedPyApp(App[None]):
    """The single-project interactive terminal session."""

    BINDINGS = [("ctrl+shift+c", "copy_transcript", "复制会话")]

    CSS = """
    #status { height: 1; padding: 0 1; background: $panel; }
    #transcript-shell { height: 1fr; border: round $primary; }
    #transcript { height: 1fr; overflow-x: hidden; overflow-y: scroll; }
    #composer-shell { height: 7; border: round $accent; }
    #composer { height: 1fr; border: none; padding: 0 1; }
    #composer-controls { height: 1; align: left middle; }
    #mode-picker, #mode-chip, #composer-model, #composer-effort, #send { height: 1; min-height: 1; color: $text-muted; }
    #mode-picker, #send { background: transparent; border: none; padding: 0 1; }
    #composer-model, #composer-effort { background: transparent; border: none; padding: 0 2; }
    #mode-picker { width: 7; min-width: 7; }
    #mode-chip { width: auto; }
    #composer-controls-spacer { width: 1fr; }
    #composer-model, #composer-effort { width: auto; min-width: 0; }
    #send { width: 12; min-width: 12; }
    #command-palette { display: none; height: auto; max-height: 12; border: round $secondary; }
    #help-modal { width: 88; max-height: 1fr; }
    #help-content { height: 1fr; overflow-y: auto; padding: 1 2; }
    """

    def __init__(
        self,
        runtime: Any,
        profile: Any,
        initial_task: str | None = None,
        conversation: Any | None = None,
    ) -> None:
        super().__init__()
        self.runtime = runtime
        self.profile = profile
        self.initial_task = initial_task
        self._palette_index = -1
        self._suppress_palette_once = False
        self._session_goal: str | None = None
        self._conversation_runtime = conversation
        self._continuous_session_id: UUID | None = None
        self._continuous_turn_id: UUID | None = None
        self._continuous_pending_approval: tuple[UUID, UUID, UUID] | None = None
        self._pending_submission: tuple[str, Literal["normal", "plan", "review"]] | None = None
        self._transcript_presenter = TranscriptPresenter()
        self._seen_events: set[tuple[UUID, UUID, int]] = set()

    def compose(self) -> ComposeResult:
        yield Static(self._status_text(), id="status")
        yield Vertical(
            TranscriptLog(id="transcript"),
            id="transcript-shell",
        )
        yield ListView(
            *(ListItem(Static(command), id=f"command-{command.removeprefix('/')}" ) for command in COMMANDS),
            id="command-palette",
        )
        yield Vertical(
            Composer(id="composer"),
            Horizontal(
                Button(Text("[+]", no_wrap=True), id="mode-picker"),
                Static("", id="mode-chip"),
                Static("", id="composer-controls-spacer"),
                Button(self.runtime.config.model, id="composer-model"),
                Button(self.runtime.config.reasoning_effort, id="composer-effort"),
                Button(Text("[发送]", no_wrap=True), id="send", disabled=True),
                id="composer-controls",
            ),
            id="composer-shell",
        )

    def on_mount(self) -> None:
        self._refresh_composer_controls()
        if self.initial_task:
            self.call_after_refresh(self.submit, self.initial_task)
            return
        store = getattr(self._conversation_runtime, "store", None)
        if store is None:
            return
        summaries = store.summaries()
        if summaries:
            self.call_after_refresh(self._conversation_selected, summaries[-1].id)
        else:
            self.call_after_refresh(self._new_conversation)

    def on_unmount(self) -> None:
        """Interrupt an active continuous turn before releasing the runtime lease."""
        self._session_goal = None
        self._interrupt_continuous()

    def on_text_area_changed(self, event: TextArea.Changed) -> None:
        if event.text_area.id == "composer":
            if self._suppress_palette_once:
                self._suppress_palette_once = False
                return
            prefix = event.text_area.text.strip()
            palette = self.query_one("#command-palette")
            palette.display = prefix.startswith("/")
            for item in palette.query(ListItem):
                command = str(item.query_one(Static).render())
                item.display = command.startswith(prefix)
            self._palette_index = -1
            palette.index = None
            palette.scroll_home(animate=False)
            self.query_one("#send", Button).disabled = not bool(event.text_area.text.strip())

    def on_key(self, event: Any) -> None:
        if event.key == "ctrl+c" and self._continuous_turn_id is not None:
            event.prevent_default()
            event.stop()
            self._interrupt_continuous()

    def on_composer_submitted(self, event: ComposerSubmitted) -> None:
        """Make the custom composer message the sole interactive submit path."""
        palette = self.query_one("#command-palette", ListView)
        if event.text.strip() in COMMANDS:
            self.submit(event.text)
            return
        if palette.display:
            items = self._visible_palette_items()
            if items:
                index = self._palette_index if self._palette_index >= 0 else 0
                self._accept_palette_item(items[index])
                return
        self.submit(event.text)

    def _visible_palette_items(self) -> list[ListItem]:
        palette = self.query_one("#command-palette", ListView)
        return [item for item in palette.query(ListItem) if item.display]

    def _move_palette(self, direction: str) -> bool:
        palette = self.query_one("#command-palette", ListView)
        if not palette.display:
            return False
        items = self._visible_palette_items()
        if not items:
            return True
        change = 1 if direction == "down" else -1
        self._palette_index = (self._palette_index + change) % len(items)
        palette.index = list(palette.query(ListItem)).index(items[self._palette_index])
        return True

    def _accept_palette_item(self, item: ListItem) -> None:
        composer = self.query_one("#composer", Composer)
        self._suppress_palette_once = True
        composer.text = str(item.query_one(Static).render())
        self.query_one("#command-palette", ListView).display = False
        composer.focus()
        composer.cursor_location = composer.document.end

    def submit(self, text: str) -> None:
        request = text.strip()
        composer = self.query_one("#composer", TextArea)
        composer.text = ""
        self.query_one("#command-palette").display = False
        if not request:
            return
        if request.startswith("/"):
            self._submit_command(request)
            return
        self._submit_continuous(request, "normal")

    def _submit_continuous(self, text: str, mode: Literal["normal", "plan", "review"]) -> None:
        """Start or steer one continuous session without the legacy task renderer."""
        try:
            if not self.runtime.credential_status().configured:
                self._pending_submission = (text, mode)
                self.push_screen(CredentialScreen(False), self._credential_resolved)
                return
            if self._continuous_session_id is None:
                self._continuous_session_id = self._conversation_runtime.create_session(str(self.profile.root))
            if self._continuous_turn_id is not None:
                event = self._conversation_runtime.steer(
                    self._continuous_session_id, self._continuous_turn_id, text
                )
                self._present_session_event(event)
                return
            goal = self._session_goal
            turn_id, event = self._conversation_runtime.begin_turn(
                self._continuous_session_id, text, mode, **({"goal": goal} if goal else {})
            )
            self._session_goal = None
            self._refresh_composer_controls()
        except CredentialBackendUnavailableError:
            self.push_screen(CredentialBackendUnavailableScreen())
            return
        except Exception:
            self._write("无法启动会话。")
            return
        self._continuous_turn_id = turn_id
        self._present_session_event(event)
        Thread(
            target=self._run_continuous_turn,
            args=(self._continuous_session_id, turn_id),
            daemon=True,
        ).start()

    def _run_continuous_turn(self, session_id: UUID, turn_id: UUID) -> None:
        try:
            for event in self._conversation_runtime.run_turn(session_id, turn_id):
                self.post_message(SessionEventReceived(event))
        except Exception:
            self.post_message(SessionEventReceived(
                SessionEvent(session_id, turn_id, 0, "turn_failed", data={"code": "surface_failure"})
            ))

    def on_session_event_received(self, message: SessionEventReceived) -> None:
        event_key = (
            message.event.session_id,
            message.event.turn_id,
            message.event.sequence,
        )
        if event_key in self._seen_events:
            return
        self._seen_events.add(event_key)
        self._present_session_event(message.event)
        if message.event.kind == "turn_started":
            self._continuous_turn_id = message.event.turn_id
        if message.event.kind == "approval_requested":
            approval_id = UUID(message.event.data["approval_id"])
            self._continuous_pending_approval = (
                message.event.session_id, message.event.turn_id, approval_id
            )
            tool = message.event.data.get("tool", "受控操作")
            path = message.event.data.get("path")
            projection = f"删除 {path}" if tool == "delete_path" and path else tool
            self.push_screen(
                ApprovalScreen(
                    projection,
                    message.event.data.get("rule_id", "approval.required"),
                    False,
                ),
                self._continuous_approval_resolved,
            )
        if message.event.kind in {"turn_completed", "turn_interrupted", "turn_failed"} and message.event.turn_id == self._continuous_turn_id:
            self._continuous_turn_id = None

    def _continuous_approval_resolved(self, decision: str | None) -> None:
        pending = self._continuous_pending_approval
        if decision is None or pending is None:
            return
        self._continuous_pending_approval = None
        Thread(
            target=self._resolve_continuous_approval,
            args=(*pending, decision != "reject"),
            daemon=True,
        ).start()

    def _resolve_continuous_approval(
        self, session_id: UUID, turn_id: UUID, approval_id: UUID, accepted: bool
    ) -> None:
        try:
            for event in self._conversation_runtime.resolve_approval(
                session_id, turn_id, approval_id, accepted
            ):
                self.post_message(SessionEventReceived(event))
        except Exception:
            self.post_message(SessionEventReceived(
                SessionEvent(session_id, turn_id, 0, "turn_failed", data={"code": "surface_failure"})
            ))

    def _interrupt_continuous(self) -> None:
        if self._continuous_session_id is None or self._continuous_turn_id is None:
            return
        try:
            event = self._conversation_runtime.interrupt(
                self._continuous_session_id, self._continuous_turn_id
            )
        except TurnNotActiveError:
            self._continuous_turn_id = None
            return
        if event is not None:
            self._present_session_event(event)
            self._continuous_turn_id = None

    def _present_session_event(self, event: SessionEvent) -> None:
        update = self._transcript_presenter.present(event)
        if update is not None:
            try:
                self.query_one("#transcript", TranscriptLog).apply_update(update)
            except NoMatches:
                return

    def _submit_command(self, command: str) -> None:
        name, _, argument = command.partition(" ")
        if name not in COMMANDS or (name in _NO_ARGUMENT_COMMANDS and argument):
            self._write("未知命令。")
            return
        if name == "/delete":
            if self._continuous_session_id is None:
                self._write("没有可删除的会话。")
            elif self._continuous_turn_id is not None:
                self._write("请先停止当前会话后再删除。")
            else:
                replacement = self._conversation_runtime.delete_session(self._continuous_session_id)
                if replacement is None:
                    self._write("至少保留一条会话，无法删除。")
                else:
                    self._conversation_selected(replacement.id)
            return
        if name == "/new":
            if self._continuous_turn_id is None:
                self._new_conversation()
            else:
                self.push_screen(NewConversationScreen(), self._new_conversation_resolved)
            return
        if name == "/conversations":
            conversations = self._conversation_runtime.store.summaries()
            if conversations:
                self.push_screen(ConversationScreen(conversations), self._conversation_selected)
            else:
                self._write("会话：0")
            return
        if name == "/permissions":
            self._write("权限：项目内读取、补丁、pytest 与只读 Git 自动允许；删除须逐次审批。")
            return
        if name in {"/tests", "/diff", "/doctor"}:
            try:
                self._write(self.runtime.local_check(name.removeprefix("/")))
            except Exception:
                self._write("本地检查不可用。")
            return
        if name == "/help":
            self.push_screen(HelpScreen())
            return
        if name == "/credentials":
            try:
                configured = self.runtime.credential_status().configured
            except CredentialBackendUnavailableError:
                self.push_screen(CredentialBackendUnavailableScreen())
                return
            except Exception:
                self._write("凭据状态不可用。")
                return
            self.push_screen(CredentialScreen(configured), self._credential_resolved)
            return
        if name == "/model":
            self._open_settings("model", ("deepseek-v4-flash", "deepseek-v4-pro"))
            return
        if name == "/effort":
            self._open_settings("reasoning_effort", ("high", "max"))
            return
        if name == "/goal":
            if argument == "clear":
                self._session_goal = None
                self._refresh_composer_controls()
            elif argument:
                self._set_session_goal(argument)
            else:
                self._write("目标不能为空。")
            return
        if name == "/stop":
            self._interrupt_continuous()
            return
        if name == "/queue":
            if not argument or self._continuous_session_id is None or self._continuous_turn_id is None:
                self._write("没有可排队的活跃会话。")
                return
            try:
                _, event = self._conversation_runtime.queue(
                    self._continuous_session_id, argument, "normal"
                )
            except Exception:
                self._write("无法排队下一轮。")
                return
            self._present_session_event(event)
            return
        if name == "/plan":
            if not argument:
                self._write("计划任务不能为空。")
                return
            self._submit_continuous(argument, "plan")
            return
        if name == "/review":
            self._submit_continuous(argument or "Review project", "review")
            return
        if name == "/exit":
            if self._continuous_turn_id is None:
                self._session_goal = None
                self.exit()
            else:
                self.push_screen(ExitScreen(), self._exit_resolved)
            return

    def _open_settings(
        self, field: Literal["model", "reasoning_effort"], values: tuple[str, ...]
    ) -> None:
        current = getattr(self.runtime.config, field)
        self.push_screen(
            SettingsScreen(field, values, current),
            lambda value: self._setting_resolved(field, value),
        )

    def _setting_resolved(
        self, field: Literal["model", "reasoning_effort"], value: str | None
    ) -> None:
        if value is None:
            return
        try:
            self.runtime.update_future_defaults(**{field: value})
        except CredentialBackendUnavailableError:
            self.push_screen(CredentialBackendUnavailableScreen())
        except Exception:
            self._write("设置未更新。")
            return
        composer = self.query_one("#composer", Composer)
        composer.text = ""
        composer.focus()
        self.query_one("#status", Static).update(self._status_text())
        self._refresh_composer_controls()

    def _credential_resolved(self, result: tuple[str, str | None] | None) -> None:
        if result is None:
            return
        operation, value = result
        try:
            if operation == "update":
                if not value:
                    self._write("凭据不能为空。")
                    return
                self.runtime.update_credential(value)
                self._write("凭据已更新。")
                pending = self._pending_submission
                self._pending_submission = None
                if pending is not None:
                    self._submit_continuous(*pending)
            elif operation == "clear_requested":
                self.push_screen(ClearCredentialScreen(), self._clear_credential_resolved)
        except Exception:
            self._write("凭据操作失败。")

    def _clear_credential_resolved(self, confirmed: bool) -> None:
        if not confirmed:
            return
        try:
            self.runtime.clear_credential()
            self._write("凭据已清除。")
        except Exception:
            self._write("凭据操作失败。")

    def _exit_resolved(self, confirmed: bool) -> None:
        if not confirmed:
            return
        self._interrupt_continuous()
        self._session_goal = None
        self.exit()

    def _new_conversation_resolved(self, confirmed: bool) -> None:
        if confirmed:
            self._interrupt_continuous()
            self._new_conversation()

    def _new_conversation(self) -> None:
        self._continuous_session_id = self._conversation_runtime.create_session(str(self.profile.root))
        self._continuous_turn_id = None
        self._transcript_presenter = TranscriptPresenter()
        self._seen_events.clear()
        self._session_goal = None
        self._refresh_composer_controls()
        self.query_one("#transcript", TranscriptLog).reset_continuous()
        composer = self.query_one("#composer", Composer)
        composer.focus()
        composer.cursor_location = composer.document.end

    def _conversation_selected(self, summary_id: UUID | None) -> None:
        if summary_id is None:
            return
        self._continuous_session_id = self._conversation_runtime.create_session(
            str(self.profile.root), summary_id
        )
        summary = self._conversation_runtime.summary(self._continuous_session_id)
        self._continuous_turn_id = None
        self._transcript_presenter = TranscriptPresenter()
        self._seen_events.clear()
        self.query_one("#transcript", TranscriptLog).reset_continuous()
        self._render_summary(summary)
        composer = self.query_one("#composer", Composer)
        composer.focus()
        composer.cursor_location = composer.document.end

    def _render_summary(self, summary: Any) -> None:
        if summary.transcript:
            transcript = self.query_one("#transcript", TranscriptLog)
            for index, entry in enumerate(summary.transcript, start=1):
                text = f"› {entry.text}" if entry.role == "user" else _display_assistant_text(entry.text)
                transcript.apply_update(TranscriptUpdate(UUID(int=index), text))
            return
        for index, turn in enumerate(summary.turns, start=1):
            if turn.final_text:
                self.query_one("#transcript", TranscriptLog).apply_update(
                    TranscriptUpdate(UUID(int=index), turn.final_text)
                )
            terminal = {
                "completed": "turn_completed",
                "interrupted": "turn_interrupted",
                "failed": "turn_failed",
            }.get(turn.terminal_status)
            if terminal is not None:
                self._present_session_event(SessionEvent(summary.id, UUID(int=0), 0, terminal))

    def _write(self, value: object) -> None:
        self.query_one("#transcript", TranscriptLog).write(str(value))

    def action_copy_transcript(self) -> None:
        transcript = self.query_one("#transcript", TranscriptLog)
        self.copy_to_clipboard("\n".join(transcript.text_entries))

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "mode-picker":
            composer = self.query_one("#composer", Composer)
            composer.text = "/"
            composer.focus()
        elif event.button.id == "composer-model":
            self._open_settings("model", ("deepseek-v4-flash", "deepseek-v4-pro"))
        elif event.button.id == "composer-effort":
            self._open_settings("reasoning_effort", ("high", "max"))
        elif event.button.id == "send":
            self.submit(self.query_one("#composer", Composer).text)

    def _set_session_goal(self, value: str) -> None:
        goal = value.strip()
        if not goal:
            self._write("目标不能为空。")
            return
        self._session_goal = goal
        self._refresh_composer_controls()

    def _refresh_composer_controls(self) -> None:
        composer = self.query_one("#composer", Composer)
        chip = self.query_one("#mode-chip", Static)
        label = "[目标]" if self._session_goal else ""
        chip.update(label)
        chip.display = bool(label)
        composer.placeholder = "输入任务"
        self.query_one("#composer-model", Button).label = self.runtime.config.model
        self.query_one("#composer-effort", Button).label = self.runtime.config.reasoning_effort

    def _status_text(self) -> str:
        return f"项目：{self.profile.root}"

    def on_list_view_selected(self, event: ListView.Selected) -> None:
        if event.list_view.id != "command-palette":
            return
        self._accept_palette_item(event.item)


class DemoEventReceived(Message):
    """Deliver an event from the offline scenario worker to the demo surface."""

    def __init__(self, event: SessionEvent) -> None:
        self.event = event
        super().__init__()


class DemoFinished(Message):
    """Finish a visual replay after its actual mock scenario returns."""

    def __init__(self, result: ScenarioResult | None) -> None:
        self.result = result
        super().__init__()


@dataclass
class DemoApprovalRequest:
    """Carry one visual approval decision back to the paused mock turn."""

    event: SessionEvent
    resolved: ThreadEvent = field(default_factory=ThreadEvent)
    accepted: bool = False


class DemoApprovalRequested(Message):
    """Ask the demo surface for a decision on its actual pending approval."""

    def __init__(self, request: DemoApprovalRequest) -> None:
        self.request = request
        super().__init__()


class DemoApp(App[None]):
    """Offline, fixed-scenario replay in the normal GuardedPy TUI shell."""

    CSS = GuardedPyApp.CSS + """
    #demo-statusbar { height: 1; background: $panel; }
    #demo-statusbar #status { width: 1fr; background: transparent; }
    #demo-hint { width: auto; min-width: 0; height: 1; color: $text-muted; padding: 0 1; }
    #send { width: 12; min-width: 12; }
    #composer-model, #composer-effort { padding: 0 1; }
    """

    _scenarios: tuple[ScenarioName, ...] = (
        "delete_requires_approval",
        "feedback_repair",
        "stale_approval_denied",
    )
    _event_presentation_delay_seconds = 0.12

    def __init__(self) -> None:
        super().__init__()
        self._index = 0
        self._scenario_running = False
        self._presenter = TranscriptPresenter()
        self._model = "Mock LLM1"
        self._effort = "high"

    def compose(self) -> ComposeResult:
        yield Horizontal(
            Static(self._status_text(), id="status"),
            Static("按↑↓来切换场景", id="demo-hint"),
            id="demo-statusbar",
        )
        yield Vertical(TranscriptLog(id="transcript"), id="transcript-shell")
        yield ListView(
            *(ListItem(Static(command), id=f"command-{command.removeprefix('/')}") for command in COMMANDS),
            id="command-palette",
        )
        yield Vertical(
            DemoRequest(scenario_request(self._selected()), id="composer"),
            Horizontal(
                Button(Text("[+]", no_wrap=True), id="mode-picker"),
                Static("", id="composer-controls-spacer"),
                Button(self._model, id="composer-model"),
                Button(self._effort, id="composer-effort"),
                Button(Text("[发送]", no_wrap=True), id="send"),
                id="composer-controls",
            ),
            id="composer-shell",
        )

    def on_mount(self) -> None:
        self.query_one("#composer", DemoRequest).focus()
        self._refresh_request()

    def _selected(self) -> ScenarioName:
        return self._scenarios[self._index]

    def _demo_key(self, key: str) -> None:
        if key == "escape":
            self.exit()
            return
        if self._scenario_running:
            return
        if key in {"down", "j"}:
            self._index = (self._index + 1) % len(self._scenarios)
            self._refresh_request()
        elif key in {"up", "k"}:
            self._index = (self._index - 1) % len(self._scenarios)
            self._refresh_request()
        elif key == "enter":
            self._start_scenario()

    def _refresh_request(self) -> None:
        scenario = self._selected()
        self.query_one("#composer", DemoRequest).update(scenario_request(scenario))

    def _open_command_palette(self) -> None:
        palette = self.query_one("#command-palette", ListView)
        palette.display = True
        palette.index = 0
        palette.focus()

    def _start_scenario(self) -> None:
        self._scenario_running = True
        self._presenter = TranscriptPresenter()
        self.query_one("#transcript", TranscriptLog).reset_continuous()
        self.query_one("#status", Static).update(self._status_text())
        scenario = self._selected()
        Thread(target=self._run_scenario, args=(scenario,), daemon=True).start()

    def _run_scenario(self, scenario: ScenarioName) -> None:
        try:
            result = run_scenario(
                scenario, on_event=self._post_demo_event, approval_resolver=self._request_demo_approval
            )
        except Exception:
            self.post_message(DemoFinished(None))
        else:
            self.post_message(DemoFinished(result))

    def _post_demo_event(self, event: SessionEvent) -> None:
        """Pace the visual replay without changing the mock core's event order."""
        self.post_message(DemoEventReceived(event))
        sleep(self._event_presentation_delay_seconds)

    def _request_demo_approval(self, event: SessionEvent) -> bool:
        request = DemoApprovalRequest(event)
        self.post_message(DemoApprovalRequested(request))
        request.resolved.wait()
        return request.accepted

    def on_demo_event_received(self, message: DemoEventReceived) -> None:
        update = self._presenter.present(message.event)
        if update is not None:
            self.query_one("#transcript", TranscriptLog).apply_update(update)

    def on_demo_approval_requested(self, message: DemoApprovalRequested) -> None:
        event = message.request.event
        path = event.data.get("path", "项目路径")
        self.push_screen(
            ApprovalScreen(f"删除 {path}", "demo.delete.approval", False),
            lambda decision: self._resolve_demo_approval(message.request, decision),
        )

    def _resolve_demo_approval(
        self, request: DemoApprovalRequest, decision: str | None
    ) -> None:
        request.accepted = decision == "once"
        request.resolved.set()

    def on_demo_finished(self, message: DemoFinished) -> None:
        self._scenario_running = False
        if message.result is None:
            self.query_one("#transcript", TranscriptLog).write("演示未完成。")
            return
        self.query_one("#status", Static).update(self._status_text())

    def _status_text(self) -> str:
        return "项目：机制演示临时项目"

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "mode-picker":
            self._open_command_palette()
        elif event.button.id == "composer-model":
            self._open_demo_settings("model", ("Mock LLM1", "Mock LLM2"))
        elif event.button.id == "composer-effort":
            self._open_demo_settings("effort", ("high", "max"))
        elif event.button.id == "send":
            self._start_scenario()

    def _open_demo_settings(self, field: Literal["model", "effort"], values: tuple[str, ...]) -> None:
        current = self._model if field == "model" else self._effort
        self.push_screen(
            SettingsScreen(f"demo_{field}", values, current),
            lambda value: self._demo_setting_resolved(field, value),
        )

    def _demo_setting_resolved(self, field: Literal["model", "effort"], value: str | None) -> None:
        if value is None:
            return
        if field == "model":
            self._model = value
        else:
            self._effort = value
        self.query_one(f"#composer-{field}", Button).label = value
        self.query_one("#composer", DemoRequest).focus()

    def on_list_view_selected(self, event: ListView.Selected) -> None:
        if event.list_view.id != "command-palette":
            return
        event.list_view.display = False
        self.query_one("#composer", DemoRequest).focus()
