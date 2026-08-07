"""The single GuardedPy CLI entry point."""

from __future__ import annotations

import argparse
from collections.abc import Callable, Sequence
from pathlib import Path
import subprocess
import sys
from typing import Any, TextIO

import keyring
from openai import OpenAI

from guardedpy.config import HarnessConfig
from guardedpy.conversation import ConversationAgent
from guardedpy.conversations import ConversationStore
from guardedpy.credentials import CredentialService
from guardedpy.discovery import ProjectDiscoveryError, discover_project
from guardedpy.executor import ToolExecutor
from guardedpy.governor import ToolGovernor, governed_tool_definitions
from guardedpy.llm import DeepSeekClient, DeepSeekConversationModel
from guardedpy.mechanism_demo import run_all_scenarios
from guardedpy.orchestrator import TaskOrchestrator
from guardedpy.runtime import ConversationRuntime, LocalRuntime, RuntimeServices
from guardedpy.terminal import run_plain_conversation


def local_runtime() -> LocalRuntime:
    """Compose provider services without reading credentials at startup."""
    return LocalRuntime(local_services())


def local_services() -> RuntimeServices:
    """Compose the injected runtime services for the sole local CLI."""
    credentials = CredentialService(_system_keyring())

    def orchestrator_factory(
        project_root: Path, config: HarnessConfig, memory_store: Any
    ) -> TaskOrchestrator:
        llm = DeepSeekClient(
            credentials.get_key,
            config,
            lambda api_key: _deepseek_transport(api_key, timeout_seconds=config.timeout_seconds),
        )
        return TaskOrchestrator(
            project_root,
            llm,
            memory_store=memory_store,
            current_branch_provider=lambda: _current_git_branch(project_root),
        )

    return RuntimeServices(credentials=credentials, orchestrator_factory=orchestrator_factory)


def continuous_runtime(runtime: LocalRuntime, profile: Any) -> ConversationRuntime:
    """Compose the governed continuous runtime used by the interactive surface."""
    config = runtime.config
    if config is None:
        raise RuntimeError("runtime is not configured")
    model = DeepSeekConversationModel(
        CredentialService(_system_keyring()).get_key,
        config,
        lambda api_key, *, max_retries: _deepseek_transport(
            api_key, timeout_seconds=config.timeout_seconds
        ),
    )
    return ConversationRuntime(
        ConversationAgent(
            model,
            governed_tool_definitions(),
            ToolGovernor(config),
            ToolExecutor(profile.root, config),
        ),
        ConversationStore(profile.root),
    )


def _system_keyring() -> Any:
    """Keep the OS credential backend replaceable at the CLI boundary."""
    return keyring.get_keyring()


def _deepseek_transport(
    api_key: str,
    *,
    timeout_seconds: int,
    openai_factory: Callable[..., Any] | None = None,
) -> Any:
    """Build one no-retry DeepSeek-compatible transport for the current snapshot."""
    factory = OpenAI if openai_factory is None else openai_factory
    return factory(
        api_key=api_key,
        base_url="https://api.deepseek.com",
        timeout=timeout_seconds,
        max_retries=0,
    )


def _current_git_branch(project_root: Path) -> str | None:
    """Read the checked-out branch without giving the model a shell capability."""
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


def main(
    argv: Sequence[str] | None = None,
    *,
    runtime_factory: Callable[[], LocalRuntime] = local_runtime,
    stdin: TextIO | None = None,
    stdout: TextIO | None = None,
) -> int:
    """Start a current-directory session, one initial task, or the fixed demo."""
    parser = _parser()
    try:
        arguments = parser.parse_args(argv)
    except SystemExit as error:
        return int(error.code)
    source = stdin or sys.stdin
    output = stdout or sys.stdout

    if arguments.target == "demo":
        return _run_demo(source, output)
    try:
        profile = discover_project(Path.cwd())
    except ProjectDiscoveryError:
        output.write("无法识别 Python pytest 项目。\n")
        return 1
    runtime = runtime_factory()
    runtime.setup(profile, api_key=None)
    if not source.isatty() or not output.isatty():
        if not runtime.credential_status().configured:
            output.write("需要先在交互终端配置凭据。\n")
            return 1
        return run_plain_conversation(
            continuous_runtime(runtime, profile), str(profile.root), source, output,
            arguments.target,
        )

    from guardedpy.tui import GuardedPyApp

    conversation = continuous_runtime(runtime, profile) if isinstance(runtime, LocalRuntime) else None
    GuardedPyApp(runtime, profile, initial_task=arguments.target, conversation=conversation).run()
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="guardedpy")
    parser.add_argument("target", nargs="?", metavar="TASK", help="initial task, or demo")
    parser.add_argument("--version", action="version", version="guardedpy 0.1.0")
    return parser


def _run_demo(source: TextIO, output: TextIO) -> int:
    if source.isatty() and output.isatty():
        from guardedpy.tui import DemoApp

        DemoApp().run()
        return 0
    for result in run_all_scenarios():
        output.write(f"{result.name} status={result.status}\n")
    return 0
