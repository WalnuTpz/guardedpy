"""Public demo coverage for fixed, offline governed-agent scenarios."""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Any

import httpx
import pytest

from guardedpy.domain import FeedbackKind, PolicyVerdict, TaskStatus


async def _request(app: Any, method: str, path: str) -> httpx.Response:
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        return await client.request(method, path)


def test_demo_lists_exactly_the_three_literal_fixed_scenarios() -> None:
    """Catches a public demo accepting arbitrary or reordered scenario definitions."""
    from guardedpy.demo import SCENARIOS

    assert SCENARIOS == (
        "dangerous_action_denied",
        "failure_feedback_corrects",
        "tdd_source_patch_denied",
    )


def test_demo_routes_are_read_only_and_exclude_local_control_capabilities() -> None:
    """Catches public demo composition exposing setup, credentials, tasks, or approvals."""
    from guardedpy.demo import create_demo_app

    app = create_demo_app()

    assert asyncio.run(_request(app, "GET", "/")).status_code == 200
    scenarios = asyncio.run(_request(app, "GET", "/demo/scenarios"))
    assert scenarios.status_code == 200
    assert scenarios.json() == [
        "dangerous_action_denied",
        "failure_feedback_corrects",
        "tdd_source_patch_denied",
    ]
    assert asyncio.run(
        _request(app, "GET", "/demo/scenarios/dangerous_action_denied")
    ).status_code == 200
    for path in (
        "/setup",
        "/settings/credentials",
        "/tasks/new",
        "/tasks/any/approval",
        "/memories",
        "/demo/scenarios/not-a-scenario",
    ):
        assert asyncio.run(_request(app, "GET", path)).status_code == 404
    assert asyncio.run(_request(app, "POST", "/demo/scenarios/dangerous_action_denied")).status_code == 405


def test_demo_runs_real_governance_feedback_and_workspace_fixtures_without_command_dispatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Catches scripted demos that skip policy, feedback, workspace, or command safety checks."""
    from guardedpy.demo import run_scenario

    original_state_home = tmp_path / "preexisting-state"
    monkeypatch.setenv("XDG_STATE_HOME", str(original_state_home))

    dangerous = run_scenario("dangerous_action_denied")
    corrected = run_scenario("failure_feedback_corrects")
    tdd_denied = run_scenario("tdd_source_patch_denied")

    assert dangerous.status is TaskStatus.BLOCKED
    assert dangerous.command_dispatches == ()
    assert any(event.policy_verdict is PolicyVerdict.DENY for event in dangerous.events)
    assert corrected.status is TaskStatus.COMPLETED
    assert corrected.source_value == "VALUE = 'fixed'\n"
    assert any(event.feedback_kind is FeedbackKind.ASSERTION_FAILURE for event in corrected.events)
    assert tdd_denied.status is TaskStatus.BLOCKED
    assert tdd_denied.source_value == "VALUE = 'broken'\n"
    assert any(event.policy_verdict is PolicyVerdict.DENY for event in tdd_denied.events)
    assert original_state_home.exists() is False
    assert Path(os.environ["XDG_STATE_HOME"]) == original_state_home


def test_demo_cli_composes_no_local_services(monkeypatch: pytest.MonkeyPatch) -> None:
    """Catches the public demo CLI starting keyring or provider-backed local services."""
    from guardedpy import web

    application = object()
    received: dict[str, object] = {}
    monkeypatch.setattr(web, "create_demo_app", lambda: application, raising=False)
    monkeypatch.setattr(
        web,
        "local_services",
        lambda: pytest.fail("demo CLI must not compose local services"),
    )
    monkeypatch.setattr(
        web.uvicorn,
        "run",
        lambda app, *, host: received.update({"app": app, "host": host}),
    )
    monkeypatch.setattr("sys.argv", ["guardedpy", "demo"])

    web.serve()

    assert received == {"app": application, "host": "127.0.0.1"}
