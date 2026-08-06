"""Terminal adapters for the local, governed GuardedPy runtime."""

from __future__ import annotations

import argparse
from collections.abc import Callable, Sequence
from getpass import getpass
from pathlib import Path
import subprocess
import sys
from typing import Any, TextIO
from uuid import UUID

import keyring
from openai import OpenAI

from guardedpy.config import HarnessConfig
from guardedpy.credentials import CredentialService
from guardedpy.domain import TaskMode, TaskState, TaskStatus, is_approval_decision
from guardedpy.llm import DeepSeekClient
from guardedpy.orchestrator import TaskOrchestrator
from guardedpy.runtime import LocalRuntime, RuntimeServices


_TERMINAL_STATUSES = {
    TaskStatus.COMPLETED,
    TaskStatus.BLOCKED,
    TaskStatus.CANCELLED,
    TaskStatus.INTERRUPTED,
}
_HELP = "可用命令：/task /help /status /tasks /memory /rules /credentials /clear /exit\n"


class _ArgumentError(ValueError):
    """A deliberately non-echoing command line syntax error."""


class _ArgumentParser(argparse.ArgumentParser):
    """Keep parser failures from printing a possibly secret argv token."""

    def error(self, message: str) -> None:
        del message
        raise _ArgumentError


def local_runtime() -> LocalRuntime:
    """Compose the provider-backed services used only after CLI parsing."""
    return LocalRuntime(local_services())


def local_services() -> RuntimeServices:
    """Compose injectable CLI runtime services without reading a credential."""
    credentials = CredentialService(_system_keyring())

    def orchestrator_factory(
        project_root: Path, config: HarnessConfig, memory_store: Any
    ) -> TaskOrchestrator:
        def transport_factory(api_key: str) -> Any:
            return _deepseek_transport(
                api_key,
                timeout_seconds=config.timeout_seconds,
            )

        llm = DeepSeekClient(credentials.get_key, config, transport_factory)
        return TaskOrchestrator(
            project_root,
            llm,
            memory_store=memory_store,
            current_branch_provider=lambda: _current_git_branch(project_root),
        )

    return RuntimeServices(credentials=credentials, orchestrator_factory=orchestrator_factory)


def _system_keyring() -> Any:
    return keyring.get_keyring()


def _deepseek_transport(
    api_key: str,
    *,
    timeout_seconds: int,
    openai_factory: Callable[..., Any] | None = None,
) -> Any:
    factory = OpenAI if openai_factory is None else openai_factory
    return factory(
        api_key=api_key,
        base_url="https://api.deepseek.com",
        timeout=timeout_seconds,
        max_retries=0,
    )


def main(
    argv: Sequence[str] | None = None,
    *,
    runtime_factory: Callable[[], LocalRuntime] = local_runtime,
    stdin: TextIO | None = None,
    stdout: TextIO | None = None,
) -> int:
    """Dispatch one local REPL or one prompt."""
    parser = _parser()
    try:
        arguments = parser.parse_args(argv)
    except _ArgumentError:
        return 2
    except SystemExit as error:
        return int(error.code)

    if arguments.mode == TaskMode.BUGFIX.value and not (
        arguments.target and arguments.target.strip()
    ):
        return 2

    runtime = runtime_factory()
    output = stdout or sys.stdout
    source = stdin or sys.stdin
    if arguments.prompt is not None:
        return _run_task(
            runtime,
            arguments.prompt,
            TaskMode(arguments.mode),
            arguments.target,
            output,
            source.readline,
        )
    return run_repl(runtime, source, output, source.isatty)


def run_repl(
    runtime: LocalRuntime,
    stdin: TextIO,
    stdout: TextIO,
    isatty: Callable[[], bool],
) -> int:
    """Run the constrained local command language over a supplied text stream."""
    while raw_line := stdin.readline():
        line = raw_line.strip()
        if not line:
            continue
        if line == "/exit":
            return 0
        if line == "/help":
            stdout.write(_HELP)
            continue
        if line == "/task":
            _interactive_task(runtime, stdout, stdin.readline)
            continue
        if line == "/status":
            _render_status(runtime, stdout)
            continue
        if line == "/tasks":
            _render_tasks(runtime, stdout)
            continue
        if line == "/memory":
            _memory_command(runtime, stdout, stdin.readline)
            continue
        if line == "/rules":
            _rules_command(runtime, stdout, stdin.readline)
            continue
        if line == "/credentials":
            _credentials_command(runtime, stdout, stdin.readline, isatty)
            continue
        if line == "/clear":
            if isatty():
                stdout.write("\033[2J\033[H")
            continue
        if line.startswith("/"):
            stdout.write("未知命令。\n")
            continue
        _run_task(runtime, line, TaskMode.FEATURE, None, stdout, stdin.readline)
    return 0


def _parser() -> _ArgumentParser:
    parser = _ArgumentParser(prog="guardedpy", add_help=True)
    parser.add_argument("--prompt")
    parser.add_argument("--mode", choices=tuple(mode.value for mode in TaskMode), default="feature")
    parser.add_argument("--target")
    return parser


def _current_git_branch(project_root: Path) -> str | None:
    try:
        completed = subprocess.run(
            ["git", "-C", str(project_root), "symbolic-ref", "--quiet", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
            shell=False,
            timeout=5,
        )
    except Exception:
        return None
    branch = completed.stdout.strip()
    return branch if completed.returncode == 0 and branch else None


def _interactive_task(runtime: LocalRuntime, stdout: TextIO, read_line: Callable[[], str]) -> int:
    mode_text = _prompt(stdout, read_line, "任务类型（feature/bugfix）: ")
    if mode_text not in {mode.value for mode in TaskMode}:
        stdout.write("任务类型无效。\n")
        return 2
    description = _prompt(stdout, read_line, "任务描述: ")
    target: str | None = None
    if mode_text == TaskMode.BUGFIX.value:
        target = _prompt(stdout, read_line, "pytest node: ")
        if not target.strip():
            stdout.write("缺陷修复任务必须提供 pytest node。\n")
            return 2
    return _run_task(runtime, description, TaskMode(mode_text), target, stdout, read_line)


def _run_task(
    runtime: LocalRuntime,
    description: str,
    mode: TaskMode,
    target: str | None,
    stdout: TextIO,
    read_line: Callable[[], str],
) -> int:
    if not description.strip():
        stdout.write("任务描述不能为空。\n")
        return 2
    task: TaskState | None = None
    try:
        task = runtime.create_task(description, mode, target)
        result = runtime.run(task.id)
    except KeyboardInterrupt:
        return _cancel_task(runtime, task.id if task is not None else None, stdout)
    except Exception:
        stdout.write("无法启动任务。\n")
        return 1
    _render_task(runtime, result, stdout)
    while result.status is TaskStatus.WAITING_APPROVAL:
        try:
            decision = _prompt(stdout, read_line, "审批（reject/once/always）: ")
        except KeyboardInterrupt:
            return _cancel_task(runtime, result.id, stdout)
        if not is_approval_decision(decision):
            stdout.write("审批输入无效。\n")
            continue
        action_hash = _pending_action_hash(runtime, result.id)
        if action_hash is None:
            stdout.write("审批请求已失效。\n")
            return 1
        try:
            if not runtime.resolve_approval(result.id, action_hash, decision):
                resolved = runtime.task(result.id)
                if resolved.status is TaskStatus.BLOCKED:
                    _render_task(runtime, resolved, stdout)
                    return 0
                stdout.write("审批请求已失效。\n")
                return 1
            result = runtime.run(result.id)
        except KeyboardInterrupt:
            return _cancel_task(runtime, result.id, stdout)
        except Exception:
            stdout.write("审批请求已失效。\n")
            return 1
        _render_task(runtime, result, stdout)
    return 0 if result.status in _TERMINAL_STATUSES else 1


def _cancel_task(runtime: LocalRuntime, task_id: UUID | None, stdout: TextIO) -> int:
    if task_id is None:
        stdout.write("任务已中断。\n")
        return 1
    try:
        task = runtime.cancel(task_id)
    except Exception:
        stdout.write("任务已中断。\n")
        return 1
    _render_task(runtime, task, stdout)
    return 0


def _render_status(runtime: LocalRuntime, stdout: TextIO) -> None:
    project_root = getattr(runtime, "project_root", None)
    config = getattr(runtime, "config", None)
    stdout.write(f"项目：{project_root if project_root is not None else '未设置'}\n")
    if config is not None:
        stdout.write(f"模型：{config.model}\n")
    stdout.write(f"凭据：{'已配置' if _credential_configured(runtime) else '未配置'}\n")


def _render_tasks(runtime: LocalRuntime, stdout: TextIO) -> None:
    try:
        tasks = runtime.tasks()
    except Exception:
        stdout.write("无法读取任务。\n")
        return
    for task in tasks:
        _render_task(runtime, task, stdout)


def _render_task(runtime: LocalRuntime, task: TaskState, stdout: TextIO) -> None:
    stdout.write(f"任务 {task.id}：{task.status.value}\n")
    try:
        events = runtime.events(task.id)
    except Exception:
        return
    for event in events:
        if event.action_projection:
            stdout.write(f"动作：{event.action_projection}\n")
        if event.policy_verdict:
            stdout.write(f"策略：{event.policy_verdict.value}\n")
        if event.feedback_kind:
            node = f" {event.feedback_node_id}" if event.feedback_node_id else ""
            stdout.write(f"反馈：{event.feedback_kind.value}{node}\n")
        if event.stop_reason:
            stdout.write(f"停止：{event.stop_reason.value}\n")


def _pending_action_hash(runtime: LocalRuntime, task_id: UUID) -> str | None:
    try:
        events = runtime.events(task_id)
    except Exception:
        return None
    for event in reversed(events):
        if event.task_status is TaskStatus.WAITING_APPROVAL and event.action_hash:
            return event.action_hash
    return None


def _memory_command(runtime: LocalRuntime, stdout: TextIO, read_line: Callable[[], str]) -> None:
    operation = _prompt(stdout, read_line, "记忆操作（list/approve/delete）: ")
    try:
        if operation == "list":
            for entry in [*runtime.memory_proposals(), *runtime.memories()]:
                stdout.write(f"记忆：{entry.id}\n")
            return
        memory_id = UUID(_prompt(stdout, read_line, "记忆 ID: "))
        if operation == "approve":
            runtime.approve_memory(memory_id)
        elif operation == "delete":
            runtime.delete_memory(memory_id)
        else:
            raise ValueError
    except Exception:
        stdout.write("记忆操作无效。\n")
        return
    stdout.write("记忆操作已完成。\n")


def _rules_command(runtime: LocalRuntime, stdout: TextIO, read_line: Callable[[], str]) -> None:
    operation = _prompt(stdout, read_line, "规则操作（list/delete）: ")
    try:
        if operation == "list":
            for rule in runtime.command_rules():
                stdout.write(f"规则：{rule.id} {rule.kind.value}\n")
            return
        if operation != "delete":
            raise ValueError
        runtime.delete_command_rule(_prompt(stdout, read_line, "规则 ID: "))
    except Exception:
        stdout.write("规则操作无效。\n")
        return
    stdout.write("规则操作已完成。\n")


def _credentials_command(
    runtime: LocalRuntime,
    stdout: TextIO,
    read_line: Callable[[], str],
    isatty: Callable[[], bool],
) -> None:
    operation = _prompt(stdout, read_line, "凭据操作（status/update/clear）: ")
    if operation == "status":
        stdout.write(f"凭据：{'已配置' if _credential_configured(runtime) else '未配置'}\n")
        return
    if operation == "update":
        if not isatty():
            stdout.write("非交互终端不能录入凭据。\n")
            return
        key = getpass("DeepSeek API Key: ")
        if not key:
            stdout.write("凭据不能为空。\n")
            return
        try:
            runtime.update_credential(key)
        except Exception:
            stdout.write("无法更新凭据。\n")
            return
        stdout.write("凭据已更新。\n")
        return
    if operation == "clear":
        try:
            runtime.clear_credential()
        except Exception:
            stdout.write("无法清除凭据。\n")
            return
        stdout.write("凭据已清除。\n")
        return
    stdout.write("凭据操作无效。\n")


def _credential_configured(runtime: LocalRuntime) -> bool:
    try:
        return runtime.credential_status().configured
    except Exception:
        return False


def _prompt(stdout: TextIO, read_line: Callable[[], str], label: str) -> str:
    stdout.write(label)
    return read_line().strip()
