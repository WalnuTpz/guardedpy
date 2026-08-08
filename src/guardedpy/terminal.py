"""Plain projection for the continuous GuardedPy conversation runtime."""

from __future__ import annotations

from typing import Any, TextIO

from guardedpy.conversation import SessionEvent, safe_event_message
from guardedpy.credentials import CredentialBackendUnavailableError


COMMANDS = (
    "/conversations", "/new", "/delete", "/exit", "/plan", "/review",
    "/tests", "/diff", "/permissions", "/credentials", "/model", "/effort",
    "/doctor", "/goal", "/help", "/stop", "/queue",
)


def render_help() -> tuple[str, ...]:
    return (
        "直接输入：与 Agent 对话，或描述要检查、修复、实现的项目任务。",
        "/new：新建会话。",
        "/conversations：列出已保存会话。",
        "/delete：仅交互终端支持删除当前会话。",
        "/exit：退出 GuardedPy。",
        "/plan <任务>：只读制定计划。",
        "/review <路径>：只读审查。",
        "/goal <目标>：仅交互终端支持的一回合目标。",
        "/stop：非交互终端没有可中断的活跃回合。",
        "/queue <任务>：非交互终端不支持排队。",
        "/tests：运行配置的 pytest。",
        "/diff：查看当前 Git diff。",
        "/doctor：查看本地项目状态。",
        "/permissions：查看自动允许与须审批的操作。",
        "/credentials：查看凭据状态；非交互终端不能录入凭据。",
        "/model：设置后续回合模型。",
        "/effort：设置后续回合思考强度。",
        "/help：显示本帮助。",
        "非交互模式不能录入凭据或批准危险操作。",
    )


def run_plain_conversation(
    runtime: Any,
    project_title: str,
    input_stream: TextIO,
    output: TextIO,
    initial_text: str | None = None,
    local_runtime: Any | None = None,
) -> int:
    """Run the sole continuous path without making approval decisions for a pipe."""
    session_id = runtime.create_session(project_title)
    requests = ([initial_text] if initial_text else []) + list(input_stream)
    for raw in requests:
        text = raw.strip()
        if not text:
            continue
        name, _, argument = text.partition(" ")
        if name == "/exit":
            return 0
        if name.startswith("/"):
            handled, session_id, code = _local_command(
                runtime, local_runtime, project_title, session_id, name, argument, output
            )
            if handled:
                if code:
                    return code
                continue
            if name not in {"/plan", "/review"}:
                output.write("未知命令。\n")
                continue
        mode = "normal"
        if name in {"/plan", "/review"}:
            mode = name.removeprefix("/")
            text = argument or ("Review project" if mode == "review" else "")
            if not text:
                output.write("任务描述不能为空。\n")
                return 2
        turn_id, user_event = runtime.begin_turn(session_id, text, mode)
        _render_event(user_event, output)
        for event in runtime.run_turn(session_id, turn_id):
            if event.kind == "approval_requested":
                output.write("需要精确审批，非交互模式已安全停止。\n")
                return 1
            _render_event(event, output)
    return 0


def _local_command(
    runtime: Any,
    local_runtime: Any | None,
    project_title: str,
    session_id: object,
    name: str,
    argument: str,
    output: TextIO,
) -> tuple[bool, object, int]:
    if name == "/help" and not argument:
        output.write("\n".join(render_help()) + "\n")
        return True, session_id, 0
    if name == "/new" and not argument:
        output.write("已新建会话。\n")
        return True, runtime.create_session(project_title), 0
    if name == "/delete" and not argument:
        output.write("删除会话仅支持交互终端。\n")
        return True, session_id, 0
    if name == "/conversations" and not argument:
        summaries = runtime.store.summaries()
        output.write(f"会话：{len(summaries)}\n")
        for summary in summaries:
            output.write(f"{summary.id} {summary.updated_at.isoformat()}\n")
        return True, session_id, 0
    if name == "/permissions" and not argument:
        output.write("权限：项目内读取、补丁、pytest 与只读 Git 自动允许；删除须逐次审批。\n")
        return True, session_id, 0
    if name in {"/stop", "/queue"}:
        output.write("非交互模式没有可控制的活跃回合。\n")
        return True, session_id, 1
    if name == "/goal":
        output.write("会话目标仅支持交互终端，且不会持久化。\n")
        return True, session_id, 0
    if local_runtime is None:
        return False, session_id, 0
    if name in {"/tests", "/diff", "/doctor"} and not argument:
        try:
            output.write(local_runtime.local_check(name.removeprefix("/")) + "\n")
        except Exception:
            output.write("本地检查不可用。\n")
        return True, session_id, 0
    if name == "/credentials":
        _credentials(local_runtime, argument, output)
        return True, session_id, 0
    if name == "/model":
        _update_default(local_runtime, "model", argument, output)
        return True, session_id, 0
    if name == "/effort":
        _update_default(local_runtime, "reasoning_effort", argument, output)
        return True, session_id, 0
    return False, session_id, 0


def _render_event(event: SessionEvent, output: TextIO) -> None:
    if event.kind == "user_message":
        output.write(f"› {event.text}\n")
    elif event.kind == "assistant_text_delta":
        output.write(f"{event.text}\n")
    else:
        message = safe_event_message(event)
        if message is not None:
            output.write(f"{message}\n")


def _render_summary(summary: Any, output: TextIO) -> None:
    for turn in summary.turns:
        if turn.final_text:
            output.write(f"{turn.final_text}\n")
        status = {"interrupted": "本轮回复已中断。"}.get(turn.terminal_status)
        if status is not None:
            output.write(f"{status}\n")


def _credentials(runtime: Any, operation: str, output: TextIO) -> None:
    if operation in {"", "status"}:
        try:
            configured = runtime.credential_status().configured
        except CredentialBackendUnavailableError:
            output.write("安全系统密钥环不可用；请先安装或启动兼容的安全系统密钥环。\n")
            return
        output.write(f"凭据：{'已配置' if configured else '未配置'}\n")
        return
    output.write("非交互终端不能录入凭据。\n")


def _update_default(runtime: Any, field: str, value: str, output: TextIO) -> None:
    if not value:
        output.write("请输入选项。\n")
        return
    try:
        runtime.update_future_defaults(**{field: value})
    except Exception:
        output.write("设置未更新。\n")
        return
    output.write("设置已更新，仅用于后续任务。\n")
