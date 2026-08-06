"""Offline contract tests for the OpenAI-compatible DeepSeek client."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
from openai import APIConnectionError, APITimeoutError, BadRequestError
import pytest

from guardedpy.config import HarnessConfig
from guardedpy.context import LlmContext
from guardedpy.domain import TaskMode, TaskState, TaskStatus
from guardedpy.events import EventStore, StopReason
from guardedpy.llm import DeepSeekClient, TemporaryProviderFailure
from guardedpy.memory import MemoryStore
from guardedpy.orchestrator import TaskOrchestrator


@dataclass
class _Message:
    content: str


@dataclass
class _Choice:
    message: _Message


@dataclass
class _Response:
    choices: list[_Choice]


class _Completions:
    def __init__(self, responses: list[object]) -> None:
        self._responses = iter(responses)
        self.calls: list[dict[str, object]] = []

    def create(self, **kwargs: object) -> _Response:
        self.calls.append(kwargs)
        response = next(self._responses)
        if isinstance(response, Exception):
            raise response
        assert isinstance(response, _Response)
        return response


class _Transport:
    def __init__(self, responses: list[object]) -> None:
        self.chat = type("Chat", (), {"completions": _Completions(responses)})()


def _completion_outcome(client: DeepSeekClient) -> str | Exception:
    try:
        return client.complete(LlmContext.minimal())
    except Exception as error:
        return error


def _sdk_connection_error() -> APIConnectionError:
    return APIConnectionError(request=httpx.Request("POST", "https://api.deepseek.com/chat/completions"))


def _sdk_timeout_error() -> APITimeoutError:
    return APITimeoutError(httpx.Request("POST", "https://api.deepseek.com/chat/completions"))


def test_deepseek_client_gets_key_at_call_time_and_uses_json_mode() -> None:
    """Catches eager key access or a completion request that permits non-JSON output."""
    keys: list[str] = []
    transport = _Transport([_Response([_Choice(_Message('{"kind":"finish","summary":"done","status":"blocked"}'))])])
    client = DeepSeekClient(lambda: keys.append("called") or "secret", "deepseek-v4-flash", lambda key: transport)

    assert keys == []
    assert client.complete(LlmContext.minimal()) == '{"kind":"finish","summary":"done","status":"blocked"}'
    assert keys == ["called"]
    assert transport.chat.completions.calls == [
        {
            "model": "deepseek-v4-flash",
            "messages": LlmContext.minimal().messages(),
            "response_format": {"type": "json_object"},
        }
    ]


def test_deepseek_client_returns_invalid_json_unchanged() -> None:
    """Catches adapter-side repair that would hide malformed model output from the action parser."""
    transport = _Transport([_Response([_Choice(_Message("not json"))])])

    assert DeepSeekClient(lambda: "secret", "deepseek-chat", lambda key: transport).complete(LlmContext.minimal()) == "not json"


def test_deepseek_client_retries_only_temporary_transport_failures() -> None:
    """Catches unbounded provider retry or swallowing a non-temporary provider exception."""
    temporary = _Transport(
        [
            ConnectionError("offline"),
            _Response([_Choice(_Message('{"kind":"finish","summary":"done","status":"blocked"}'))]),
        ]
    )
    client = DeepSeekClient(lambda: "secret", "deepseek-chat", lambda key: temporary)

    assert '"kind":"finish"' in client.complete(LlmContext.minimal())
    assert len(temporary.chat.completions.calls) == 2

    permanent = _Transport([ValueError("bad request")])
    with pytest.raises(ValueError, match="bad request"):
        DeepSeekClient(lambda: "secret", "deepseek-chat", lambda key: permanent).complete(LlmContext.minimal())


@pytest.mark.parametrize("failure_factory", [_sdk_connection_error, _sdk_timeout_error])
def test_deepseek_client_retries_actual_openai_transient_failures_once(
    failure_factory: Any,
) -> None:
    """Catches an adapter that recognizes only Python built-in transport exceptions."""
    transport = _Transport(
        [
            failure_factory(),
            _Response([_Choice(_Message('{"kind":"finish","summary":"done","status":"blocked"}'))]),
        ]
    )

    result = _completion_outcome(
        DeepSeekClient(lambda: "secret", "deepseek-chat", lambda key: transport)
    )

    assert result == '{"kind":"finish","summary":"done","status":"blocked"}'
    assert len(transport.chat.completions.calls) == 2


@pytest.mark.parametrize("failure_factory", [_sdk_connection_error, _sdk_timeout_error])
def test_two_actual_openai_transient_failures_stop_after_two_total_calls(
    failure_factory: Any,
) -> None:
    """Catches SDK retries being compounded by more than one harness-level retry."""
    transport = _Transport([failure_factory(), failure_factory()])

    outcome = _completion_outcome(
        DeepSeekClient(lambda: "secret", "deepseek-chat", lambda key: transport)
    )

    assert isinstance(outcome, TemporaryProviderFailure)
    assert len(transport.chat.completions.calls) == 2


def test_deepseek_client_propagates_actual_openai_permanent_failure_unchanged() -> None:
    """Catches broad SDK exception handling that would misclassify a permanent response error."""
    response = httpx.Response(
        400,
        request=httpx.Request("POST", "https://api.deepseek.com/chat/completions"),
    )
    permanent = BadRequestError("bad request", response=response, body={"error": "invalid"})
    transport = _Transport([permanent])

    with pytest.raises(BadRequestError) as caught:
        DeepSeekClient(lambda: "secret", "deepseek-chat", lambda key: transport).complete(
            LlmContext.minimal()
        )

    assert caught.value is permanent
    assert len(transport.chat.completions.calls) == 1


def test_openai_transport_disables_sdk_retries_and_uses_config_timeout() -> None:
    """Catches the OpenAI SDK silently owning retries or ignoring the project timeout."""
    from guardedpy import cli

    captured: dict[str, object] = {}
    sentinel = object()

    def openai_factory(**kwargs: object) -> object:
        captured.update(kwargs)
        return sentinel

    try:
        created = cli._deepseek_transport(
            "secret", timeout_seconds=17, openai_factory=openai_factory
        )
    except TypeError as error:
        created = error

    assert created is sentinel
    assert captured == {
        "api_key": "secret",
        "base_url": "https://api.deepseek.com",
        "timeout": 17,
        "max_retries": 0,
    }


def test_local_services_composes_the_task_config_timeout_into_openai_transport(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Catches local composition using a default timeout instead of the task snapshot."""
    from guardedpy import cli

    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    captured: dict[str, object] = {}

    class Keyring:
        def get_password(self, service_name: str, username: str) -> str:
            del service_name, username
            return "secret"

        def set_password(self, service_name: str, username: str, password: str) -> None:
            del service_name, username, password

        def delete_password(self, service_name: str, username: str) -> None:
            del service_name, username

    transport = _Transport(
        [_Response([_Choice(_Message('{"kind":"finish","summary":"done","status":"blocked"}'))])]
    )

    def openai_factory(**kwargs: object) -> _Transport:
        captured.update(kwargs)
        return transport

    monkeypatch.setattr(cli, "_system_keyring", lambda: Keyring())
    monkeypatch.setattr(cli, "OpenAI", openai_factory)
    services = cli.local_services()
    config = HarnessConfig(
        source_dirs=(Path("src"),),
        test_dirs=(Path("tests"),),
        pytest_command=("pytest",),
        timeout_seconds=19,
    )
    orchestrator = services.orchestrator_factory(tmp_path, config, MemoryStore(tmp_path))

    orchestrator.run(
        TaskState(
            description="Stop",
            mode=TaskMode.BUGFIX,
            bugfix_target="tests/test_value.py::test_value_is_fixed",
            config=config,
        )
    )

    assert captured["timeout"] == 19
    assert captured["max_retries"] == 0


def test_two_temporary_provider_failures_stop_with_their_own_audit_reason(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Catches retry exhaustion being misrepresented as a model-selected finish action."""
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    transport = _Transport([ConnectionError("offline"), ConnectionError("offline")])
    client = DeepSeekClient(lambda: "secret", "deepseek-chat", lambda key: transport)
    task = TaskState(
        description="Repair the selected failure",
        mode=TaskMode.BUGFIX,
        bugfix_target="tests/test_value.py::test_value_is_fixed",
        config=HarnessConfig(source_dirs=(Path("src"),), test_dirs=(Path("tests"),), pytest_command=("pytest",)),
    )

    stopped = TaskOrchestrator(tmp_path, client).run(task)

    assert stopped.status is TaskStatus.BLOCKED
    assert EventStore(tmp_path).events_for(task.id)[-1].stop_reason is StopReason.PROVIDER_TEMPORARY_FAILURE
    assert "secret" not in repr(EventStore(tmp_path).events_for(task.id))
