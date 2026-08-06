"""Safe, provider-free rendering for redirected GuardedPy sessions."""

from __future__ import annotations

from collections.abc import Callable
from io import TextIOBase
from pathlib import Path
import subprocess
import sys
from typing import Any, TextIO

from guardedpy.domain import TaskIntent, TaskState, TaskStatus


COMMANDS = (
    "/new", "/clear", "/history", "/exit", "/plan", "/review", "/tests", "/diff",
    "/permissions", "/credentials", "/memory", "/model", "/effort", "/status", "/doctor",
    "/help",
)
_HELP = "可用命令：" + " ".join(COMMANDS) + "\n"


def lifecycle_lines(runtime: Any, task: TaskState) -> tuple[str, ...]:
    """Return only fixed audit projections for a task lifecycle."""
    lines = [f"任务 {task.id}：{task.status.value}"]
    try:
        events = runtime.events(task.id)
    except Exception:
        events = []
    for event in events:
        if event.action_projection:
            lines.append(f"动作：{event.action_projection}")
        if event.policy_verdict:
            lines.append(f"策略：{event.policy_verdict.value}")
        if event.feedback_kind:
            node = f" {event.feedback_node_id}" if event.feedback_node_id else ""
            lines.append(f"反馈：{event.feedback_kind.value}{node}")
        if event.stop_reason:
            lines.append(f"停止：{event.stop_reason.value}")
    return tuple(lines)


def render_lifecycle(runtime: Any, task: TaskState, output: TextIO) -> None:
    """Write a stable, secret-free lifecycle projection."""
    for line in lifecycle_lines(runtime, task):
        output.write(f"{line}\n")


def run_noninteractive_task(
    runtime: Any, description: str, intent: TaskIntent, output: TextIO, review_path: str | None = None
) -> int:
    """Run one task without any secret prompt or automatic approval decision."""
    if not description.strip():
        output.write("任务描述不能为空。\n")
        return 2
    try:
        task = runtime.create_task(description, intent, review_path=review_path)
        result = runtime.run(task.id)
    except Exception:
        output.write("无法启动任务。\n")
        return 1
    render_lifecycle(runtime, result, output)
    if result.status is TaskStatus.WAITING_APPROVAL:
        output.write("需要人工审批，非交互模式已安全停止。\n")
        return 1
    return 0 if result.status in {
        TaskStatus.COMPLETED,
        TaskStatus.BLOCKED,
        TaskStatus.CANCELLED,
        TaskStatus.INTERRUPTED,
    } else 1


def run_plain_session(runtime: Any, input_stream: TextIO, output: TextIO) -> int:
    """Handle the same visible command language without TTY-only capabilities."""
    for raw_line in input_stream:
        line = raw_line.strip()
        if not line:
            continue
        if line == "/exit":
            return 0
        if line == "/help":
            output.write(_HELP)
            continue
        if line in {"/new", "/clear"}:
            output.write("已清除当前视觉会话。\n")
            continue
        if line == "/history":
            _render_history(runtime, output)
            continue
        if line.startswith("/plan"):
            request = line.removeprefix("/plan").strip()
            code = run_noninteractive_task(runtime, request, TaskIntent.PLAN, output)
            if code:
                return code
            continue
        if line.startswith("/review"):
            review_path = line.removeprefix("/review").strip() or None
            code = run_noninteractive_task(runtime, "Review project", TaskIntent.REVIEW, output, review_path)
            if code:
                return code
            continue
        if line == "/tests":
            _run_tests(runtime, output)
            continue
        if line == "/diff":
            _run_diff(runtime, output)
            continue
        if line == "/permissions":
            _render_permissions(runtime, output)
            continue
        if line == "/credentials" or line.startswith("/credentials "):
            _credentials(runtime, line.removeprefix("/credentials").strip(), output)
            continue
        if line == "/memory":
            _render_memory(runtime, output)
            continue
        if line.startswith("/model"):
            _update_default(runtime, "model", line.removeprefix("/model").strip(), output)
            continue
        if line.startswith("/effort"):
            _update_default(runtime, "reasoning_effort", line.removeprefix("/effort").strip(), output)
            continue
        if line == "/status":
            _render_status(runtime, output)
            continue
        if line == "/doctor":
            _doctor(runtime, output)
            continue
        if line.startswith("/"):
            output.write("未知命令。\n")
            continue
        code = run_noninteractive_task(runtime, line, TaskIntent.CODING, output)
        if code:
            return code
    return 0


def _render_history(runtime: Any, output: TextIO) -> None:
    try:
        tasks = runtime.tasks()
    except Exception:
        output.write("无法读取任务历史。\n")
        return
    for task in tasks:
        render_lifecycle(runtime, task, output)


def _run_tests(runtime: Any, output: TextIO) -> None:
    profile = runtime.config.profile
    result = subprocess.run(profile.pytest_command, cwd=profile.root, capture_output=True, text=True, check=False, shell=False)
    output.write(f"完整测试：{'passed' if result.returncode == 0 else 'failed'}\n")


def _run_diff(runtime: Any, output: TextIO) -> None:
    root = runtime.project_root
    result = subprocess.run(("git", "-C", str(root), "diff", "--"), capture_output=True, text=True, check=False, shell=False)
    if result.returncode:
        output.write("当前目录不是可用 Git 仓库。\n")
        return
    output.write(result.stdout or "没有未提交改动。\n")


def _render_permissions(runtime: Any, output: TextIO) -> None:
    try:
        rules = runtime.command_rules()
    except Exception:
        output.write("无法读取权限规则。\n")
        return
    output.write("权限规则：" + (", ".join(rule.kind.value for rule in rules) or "无") + "\n")


def _credentials(runtime: Any, operation: str, output: TextIO) -> None:
    if operation in {"", "status"}:
        try:
            configured = runtime.credential_status().configured
        except Exception:
            output.write("凭据状态不可用。\n")
            return
        output.write(f"凭据：{'已配置' if configured else '未配置'}\n")
        return
    output.write("非交互终端不能录入凭据。\n")


def _render_memory(runtime: Any, output: TextIO) -> None:
    try:
        entries = [*runtime.memory_proposals(), *runtime.memories()]
    except Exception:
        output.write("无法读取记忆。\n")
        return
    output.write(f"记忆条目：{len(entries)}\n")


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


def _render_status(runtime: Any, output: TextIO) -> None:
    config = runtime.config
    output.write(f"项目：{runtime.project_root}\n")
    output.write(f"模型：{config.model} effort：{config.reasoning_effort}\n")
    _credentials(runtime, "status", output)


def _doctor(runtime: Any, output: TextIO) -> None:
    profile = runtime.config.profile
    output.write(f"项目发现：{profile.discovery_source}\n")
    output.write(f"pytest：{' '.join(profile.pytest_command)}\n")
    output.write(f"终端：{'TTY' if sys.stdin.isatty() and sys.stdout.isatty() else '非交互'}\n")
    _credentials(runtime, "status", output)
