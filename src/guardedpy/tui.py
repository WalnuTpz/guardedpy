"""Textual session surfaces backed by GuardedPy's deterministic runtime."""

from __future__ import annotations

from io import StringIO
from typing import Any

from textual.app import App, ComposeResult
from textual.containers import Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, Footer, Input, ListItem, ListView, RichLog, Static, TextArea

from guardedpy.domain import TaskIntent, TaskState, TaskStatus
from guardedpy.mechanism_demo import ScenarioName, run_scenario
from guardedpy.terminal import COMMANDS, lifecycle_lines, run_plain_session


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


class GuardedPyApp(App[None]):
    """The single-project interactive terminal session."""

    CSS = """
    #status { height: 3; padding: 0 1; background: $panel; }
    #transcript { height: 1fr; border: round $primary; }
    #composer { height: 5; border: round $accent; }
    #command-palette { display: none; height: auto; max-height: 12; border: round $secondary; }
    """

    def __init__(self, runtime: Any, profile: Any, initial_task: str | None = None) -> None:
        super().__init__()
        self.runtime = runtime
        self.profile = profile
        self.initial_task = initial_task
        self._pending_approval: tuple[TaskState, str] | None = None
        self._active_task: TaskState | None = None

    def compose(self) -> ComposeResult:
        config = self.runtime.config
        yield Static(self._status_text(), id="status")
        yield RichLog(id="transcript", wrap=True, highlight=False, markup=False)
        yield ListView(
            *(ListItem(Static(command), id=f"command-{command.removeprefix('/')}" ) for command in COMMANDS),
            id="command-palette",
        )
        yield TextArea(id="composer")
        yield Footer()

    def on_mount(self) -> None:
        if self.initial_task:
            self.call_after_refresh(self.submit, self.initial_task)

    def on_text_area_changed(self, event: TextArea.Changed) -> None:
        if event.text_area.id == "composer":
            prefix = event.text_area.text.strip()
            palette = self.query_one("#command-palette")
            palette.display = prefix.startswith("/")
            for item in palette.query(ListItem):
                command = str(item.query_one(Static).render())
                item.display = command.startswith(prefix)

    def on_key(self, event: Any) -> None:
        if event.key == "enter" and self.focused is self.query_one("#composer"):
            event.prevent_default()
            event.stop()
            self.submit(self.query_one("#composer", TextArea).text)
        if event.key == "ctrl+c" and self._active_task is not None:
            event.prevent_default()
            event.stop()
            self._render_task(self.runtime.cancel(self._active_task.id))

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
        self._start_task(request, TaskIntent.CODING)

    def request_approval(
        self,
        task: TaskState,
        *,
        action_projection: str,
        rule_id: str,
        raw_action_hash: str,
        permanent_eligible: bool = False,
    ) -> None:
        """Open a modal while retaining the hash privately for exact resolution."""
        self._pending_approval = (task, raw_action_hash)
        self.push_screen(
            ApprovalScreen(action_projection, rule_id, permanent_eligible),
            self._approval_resolved,
        )

    def _approval_resolved(self, decision: str | None) -> None:
        if decision is None or self._pending_approval is None:
            return
        task, action_hash = self._pending_approval
        self._pending_approval = None
        try:
            if self.runtime.resolve_approval(task.id, action_hash, decision):
                self._render_task(self.runtime.run(task.id))
        except Exception:
            self._write("审批请求已失效。")

    def _submit_command(self, command: str) -> None:
        name, _, argument = command.partition(" ")
        if name not in COMMANDS:
            self._write("未知命令。")
            return
        if name in {"/new", "/clear"}:
            self.query_one("#transcript", RichLog).clear()
            return
        if name == "/help":
            self._write("可用命令：" + " ".join(COMMANDS))
            return
        if name == "/status":
            self._write(self.query_one("#status", Static).render())
            return
        if name == "/credentials":
            try:
                configured = self.runtime.credential_status().configured
            except Exception:
                self._write("凭据状态不可用。")
                return
            self.push_screen(CredentialScreen(configured), self._credential_resolved)
            return
        if name == "/plan":
            self._start_task(argument, TaskIntent.PLAN)
            return
        if name == "/review":
            self._start_task("Review project", TaskIntent.REVIEW, argument or None)
            return
        if name == "/exit":
            if self._active_task is None:
                self.exit()
            else:
                self.push_screen(ExitScreen(), self._exit_resolved)
            return
        if name in {"/tests", "/diff", "/permissions", "/memory", "/model", "/effort", "/history", "/doctor"}:
            output = StringIO()
            run_plain_session(self.runtime, StringIO(f"{command}\n"), output)
            for line in output.getvalue().splitlines():
                self._write(line)

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
        if self._active_task is not None:
            self._render_task(self.runtime.cancel(self._active_task.id))
        self.exit()

    def _start_task(self, description: str, intent: TaskIntent, review_path: str | None = None) -> None:
        if not description.strip():
            self._write("任务描述不能为空。")
            return
        try:
            task = self.runtime.create_task(description, intent, review_path=review_path)
            self._active_task = task
            self._render_task(self.runtime.run(task.id))
        except Exception:
            self._write("无法启动任务。")

    def _render_task(self, task: TaskState) -> None:
        self._active_task = task if task.status in {TaskStatus.PENDING, TaskStatus.RUNNING, TaskStatus.WAITING_APPROVAL} else None
        self.query_one("#status", Static).update(self._status_text(task))
        for line in lifecycle_lines(self.runtime, task):
            self._write(line)
        if task.status is TaskStatus.WAITING_APPROVAL:
            for event in reversed(self.runtime.events(task.id)):
                if event.action_projection and event.policy_rule_id and event.action_hash:
                    self.request_approval(
                        task,
                        action_projection=event.action_projection,
                        rule_id=event.policy_rule_id,
                        raw_action_hash=event.action_hash,
                        permanent_eligible=bool(event.permanent_eligible),
                    )
                    return

    def _write(self, value: object) -> None:
        self.query_one("#transcript", RichLog).write(str(value))

    def _status_text(self, task: TaskState | None = None) -> str:
        config = self.runtime.config
        active = task or self._active_task
        baseline = active.path.value if active is not None else "未开始"
        status = active.status.value if active is not None else "idle"
        return (
            f"项目：{self.profile.root}\n模型：{config.model} · effort：{config.reasoning_effort}\n"
            f"基线：{baseline} · 任务：{status} · 测试：完整套件"
        )

    def on_list_view_selected(self, event: ListView.Selected) -> None:
        if event.list_view.id != "command-palette":
            return
        self.query_one("#composer", TextArea).text = str(event.item.query_one(Static).render())
        self.query_one("#command-palette").display = False
        self.query_one("#composer", TextArea).focus()


class DemoApp(App[None]):
    """Offline, fixed-scenario Textual mechanism demonstration."""

    _scenarios: tuple[ScenarioName, ...] = (
        "dangerous_action_denied",
        "failure_feedback_corrects",
        "tdd_source_patch_denied",
    )
    _requests = {
        "dangerous_action_denied": "Attempt a prohibited privileged action.",
        "failure_feedback_corrects": "Correct the selected assertion failure.",
        "tdd_source_patch_denied": "Attempt a source patch before observing red.",
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
