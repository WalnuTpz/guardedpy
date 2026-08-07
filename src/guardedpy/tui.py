"""Textual session surfaces backed by GuardedPy's deterministic runtime."""

from __future__ import annotations

from dataclasses import dataclass, field
from threading import Thread
from typing import Any, Literal
from uuid import UUID

from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical
from textual import events
from textual.message import Message
from textual.screen import ModalScreen
from textual.widgets import Button, Input, ListItem, ListView, Log, RichLog, Static, TextArea

from guardedpy.conversation import SessionEvent
from guardedpy.credentials import CredentialBackendUnavailableError
from guardedpy.mechanism_demo import ScenarioName, run_scenario


COMMANDS = (
    "/history", "/conversations", "/new", "/clear", "/exit", "/plan", "/review",
    "/credentials", "/model", "/effort", "/goal", "/help", "/stop", "/queue",
)

_HELP_LINES = (
    "会话与对话：/history /conversations /new /clear /exit",
    "任务：/plan <任务> /review <路径> /stop /queue <任务>",
    "设置与安全：/model /effort /goal /credentials",
    "交互：Enter 提交，Shift+Enter 或 Ctrl+J 换行。",
)


_NO_ARGUMENT_COMMANDS = frozenset(
    {
        "/new", "/clear", "/history", "/conversations", "/exit", "/credentials",
        "/help", "/model", "/effort", "/stop",
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

    def present(self, event: SessionEvent) -> TranscriptUpdate | None:
        """Return the visible update for one event without exposing tool payloads."""
        if event.kind == "user_message":
            return TranscriptUpdate(event.item_id, f"› {event.text}")
        if event.kind == "assistant_text_delta":
            if event.item_id is None:
                return None
            text = self._assistant_text.get(event.item_id, "") + event.text
            self._assistant_text[event.item_id] = text
            return TranscriptUpdate(event.item_id, f"助手：{text}", replace=True)
        status = {
            "tool_item_started": "正在使用受控工具。",
            "tool_output": "工具已返回受限结果。",
            "tool_item_completed": "工具执行完成。",
            "approval_requested": "需要精确审批。",
            "approval_resolved": "审批已处理。",
            "turn_completed": "本轮回复已完成。",
            "turn_interrupted": "本轮回复已中断。",
            "turn_failed": "本轮回复未完成。",
        }.get(event.kind)
        if status is None:
            return None
        return TranscriptUpdate(event.item_id, status)


class TranscriptLog(Log):
    """Selectable log that preserves one fixed safe UI projection per line."""

    def __init__(self, *args: object, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)
        self._continuous_entries: list[tuple[UUID | None, str]] = []

    def write(self, data: str, scroll_end: bool | None = None) -> "TranscriptLog":
        super().write(f"{data}\n", scroll_end=scroll_end)
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
            self._continuous_entries.append((update.item_id, update.text))
        self.clear()
        for _item_id, text in self._continuous_entries:
            self.write(text)

    def reset_continuous(self) -> None:
        """Discard the current session's rendered event projection."""
        self._continuous_entries.clear()
        self.clear()


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


class SettingsScreen(ModalScreen[str | None]):
    """Choose a validated future-task setting with keyboard or mouse."""

    def __init__(
        self, field: Literal["model", "reasoning_effort"], values: tuple[str, ...]
    ) -> None:
        super().__init__()
        self._field = field
        self._values = values

    def compose(self) -> ComposeResult:
        yield Vertical(
            Static("选择模型" if self._field == "model" else "选择推理强度"),
            ListView(
                *(ListItem(Static(value), id=f"setting-{value}") for value in self._values),
                id="settings-picker",
            ),
            id="settings-modal",
        )

    def on_mount(self) -> None:
        picker = self.query_one("#settings-picker", ListView)
        picker.index = 0
        picker.focus()

    def on_list_view_selected(self, event: ListView.Selected) -> None:
        self.dismiss(str(event.item.query_one(Static).render()))


class ModePickerScreen(ModalScreen[Literal["plan", "review", "goal"] | None]):
    """Select the next in-composer task mode without attachment semantics."""

    _options = (("plan", "计划"), ("review", "审查"), ("goal", "目标"))

    def compose(self) -> ComposeResult:
        yield Vertical(
            ListView(
                *(ListItem(Static(label), id=f"mode-{mode}") for mode, label in self._options),
                id="mode-picker-list",
            ),
            id="mode-picker-modal",
        )

    def on_mount(self) -> None:
        picker = self.query_one("#mode-picker-list", ListView)
        picker.index = 0
        picker.focus()

    def on_list_view_selected(self, event: ListView.Selected) -> None:
        self.dismiss((event.item.id or "").removeprefix("mode-") or None)


class HelpScreen(ModalScreen[None]):
    """Scrollable safe help, kept outside the task transcript."""

    def compose(self) -> ComposeResult:
        yield Vertical(Log(id="help-content", highlight=False), Button("关闭", id="help-close"), id="help-modal")

    def on_mount(self) -> None:
        content = self.query_one("#help-content", Log)
        for line in _HELP_LINES:
            content.write(line)

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

    CSS = """
    #status { height: 1; padding: 0 1; background: $panel; }
    #transcript-shell { height: 1fr; border: round $primary; }
    #transcript { height: 1fr; }
    #composer-shell { height: 6; layers: composer controls; }
    #composer { height: 6; border: round $accent; padding: 0 35 2 0; layer: composer; }
    #composer-controls { height: 2; layer: controls; dock: bottom; align: right middle; }
    #mode-picker, #mode-chip, #composer-model, #composer-effort, #send { height: 1; color: $text-muted; }
    #mode-picker { width: 3; }
    #mode-chip { width: auto; }
    #composer-model, #composer-effort { width: auto; }
    #send { width: 8; }
    #command-palette { display: none; height: auto; max-height: 12; border: round $secondary; }
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
        self._mode: Literal["coding", "plan", "review", "goal"] = "coding"
        self._conversation_runtime = conversation
        self._continuous_session_id: UUID | None = None
        self._continuous_turn_id: UUID | None = None
        self._continuous_pending_approval: tuple[UUID, UUID, UUID] | None = None
        self._transcript_presenter = TranscriptPresenter()

    def compose(self) -> ComposeResult:
        yield Static(self._status_text(), id="status")
        yield Vertical(
            TranscriptLog(id="transcript", highlight=False),
            id="transcript-shell",
        )
        yield ListView(
            *(ListItem(Static(command), id=f"command-{command.removeprefix('/')}" ) for command in COMMANDS),
            id="command-palette",
        )
        yield Vertical(
            Composer(id="composer"),
            Horizontal(
                Button("＋", id="mode-picker"),
                Static("", id="mode-chip"),
                Button(self.runtime.config.model, id="composer-model"),
                Button(self.runtime.config.reasoning_effort, id="composer-effort"),
                Button("发送", id="send", disabled=True),
                id="composer-controls",
            ),
            id="composer-shell",
        )

    def on_mount(self) -> None:
        self._refresh_composer_controls()
        if self.initial_task:
            self.call_after_refresh(self.submit, self.initial_task)

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
        mode = self._mode
        if mode == "goal":
            self._set_session_goal(request)
            return
        self._mode = "coding"
        self._refresh_composer_controls()
        self._submit_continuous(
            request, {"coding": "normal", "plan": "plan", "review": "review"}[mode]
        )

    def _submit_continuous(self, text: str, mode: Literal["normal", "plan", "review"]) -> None:
        """Start or steer one continuous session without the legacy task renderer."""
        try:
            if not self.runtime.credential_status().configured:
                self._write("需要先在交互终端配置凭据。")
                return
            if self._continuous_session_id is None:
                self._continuous_session_id = self._conversation_runtime.create_session(str(self.profile.root))
            if self._continuous_turn_id is not None:
                event = self._conversation_runtime.steer(
                    self._continuous_session_id, self._continuous_turn_id, text
                )
                self._present_session_event(event)
                return
            turn_id, event = self._conversation_runtime.begin_turn(
                self._continuous_session_id, text, mode
            )
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
        self._present_session_event(message.event)
        if message.event.kind == "approval_requested":
            approval_id = UUID(message.event.data["approval_id"])
            self._continuous_pending_approval = (
                message.event.session_id, message.event.turn_id, approval_id
            )
            self.push_screen(
                ApprovalScreen(
                    message.event.data.get("tool", "受控操作"),
                    message.event.data.get("rule_id", "approval.required"),
                    False,
                ),
                self._continuous_approval_resolved,
            )
        if message.event.kind in {"turn_completed", "turn_interrupted", "turn_failed"}:
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
        event = self._conversation_runtime.interrupt(
            self._continuous_session_id, self._continuous_turn_id
        )
        if event is not None:
            self._present_session_event(event)
            self._continuous_turn_id = None

    def _present_session_event(self, event: SessionEvent) -> None:
        update = self._transcript_presenter.present(event)
        if update is not None:
            self.query_one("#transcript", TranscriptLog).apply_update(update)

    def _submit_command(self, command: str) -> None:
        name, _, argument = command.partition(" ")
        if name not in COMMANDS or (name in _NO_ARGUMENT_COMMANDS and argument):
            self._write("未知命令。")
            return
        if name == "/clear":
            self.query_one("#transcript", TranscriptLog).reset_continuous()
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
        if name == "/history":
            if self._continuous_session_id is None:
                self._write("会话：0")
            else:
                self._render_summary(self._conversation_runtime.summary(self._continuous_session_id))
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
                self._mode = "coding"
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
        self.push_screen(SettingsScreen(field, values), lambda value: self._setting_resolved(field, value))

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
        self._session_goal = None
        self._mode = "coding"
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
        self.query_one("#transcript", TranscriptLog).reset_continuous()
        self._render_summary(summary)
        composer = self.query_one("#composer", Composer)
        composer.focus()
        composer.cursor_location = composer.document.end

    def _render_summary(self, summary: Any) -> None:
        for index, turn in enumerate(summary.turns, start=1):
            if turn.final_text:
                self.query_one("#transcript", TranscriptLog).apply_update(
                    TranscriptUpdate(UUID(int=index), f"助手：{turn.final_text}")
                )
            terminal = {
                "completed": "turn_completed",
                "interrupted": "turn_interrupted",
                "failed": "turn_failed",
            }.get(turn.terminal_status)
            if terminal is not None:
                self._present_session_event(SessionEvent(summary.id, UUID(int=0), 0, terminal))

    def _write(self, value: object) -> None:
        self.query_one("#transcript", Log).write(str(value))

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "mode-picker":
            self.push_screen(ModePickerScreen(), self._mode_resolved)
        elif event.button.id == "composer-model":
            self._open_settings("model", ("deepseek-v4-flash", "deepseek-v4-pro"))
        elif event.button.id == "composer-effort":
            self._open_settings("reasoning_effort", ("high", "max"))
        elif event.button.id == "send":
            self.submit(self.query_one("#composer", Composer).text)

    def _mode_resolved(self, mode: Literal["plan", "review", "goal"] | None) -> None:
        if mode is None:
            return
        self._mode = mode
        self._refresh_composer_controls()
        composer = self.query_one("#composer", Composer)
        composer.focus()

    def _set_session_goal(self, value: str) -> None:
        goal = value.strip()
        if not goal:
            self._write("目标不能为空。")
            return
        self._session_goal = goal
        self._mode = "coding"
        self._refresh_composer_controls()

    def _refresh_composer_controls(self) -> None:
        composer = self.query_one("#composer", Composer)
        chip = self.query_one("#mode-chip", Static)
        labels = {"plan": "[计划]", "review": "[审查]", "goal": "[目标]"}
        label = labels.get(self._mode) or ("[目标]" if self._session_goal else "")
        chip.update(label)
        chip.display = bool(label)
        composer.placeholder = {
            "coding": "输入任务",
            "plan": "输入规划任务",
            "review": "输入项目根内相对审查路径",
            "goal": "输入会话目标",
        }[self._mode]
        self.query_one("#composer-model", Button).label = self.runtime.config.model
        self.query_one("#composer-effort", Button).label = self.runtime.config.reasoning_effort

    def _status_text(self) -> str:
        return f"项目：{self.profile.root}"

    def on_list_view_selected(self, event: ListView.Selected) -> None:
        if event.list_view.id != "command-palette":
            return
        self._accept_palette_item(event.item)


class DemoApp(App[None]):
    """Offline, fixed-scenario Textual mechanism demonstration."""

    _scenarios: tuple[ScenarioName, ...] = (
        "delete_approval_rejected",
        "feedback_repair",
        "stale_approval_denied",
    )
    _requests = {
        "delete_approval_rejected": "Reject an exact deletion approval.",
        "feedback_repair": "Correct the selected assertion failure.",
        "stale_approval_denied": "Reject a forged or stale approval identifier.",
    }

    def __init__(self) -> None:
        super().__init__()
        self._index = 0

    def compose(self) -> ComposeResult:
        yield Static("选择机制演示，按 Enter 运行。", id="demo-title")
        yield Static(self._requests[self._scenarios[self._index]], id="demo-request")
        yield RichLog(id="demo-transcript", wrap=True, highlight=False, markup=False)

    def on_key(self, event: Any) -> None:
        if event.key in {"down", "j"}:
            self._index = (self._index + 1) % len(self._scenarios)
            self.query_one("#demo-request", Static).update(self._requests[self._scenarios[self._index]])
        elif event.key == "enter":
            result = run_scenario(self._scenarios[self._index])
            self.query_one("#demo-transcript", RichLog).write(f"{result.name} status={result.status}")
