"""Headless mechanism evidence from the self-owned Harness loop."""

from __future__ import annotations

from dataclasses import fields
import os
from pathlib import Path

import pytest

from guardedpy.context import LlmContext


def test_scenario_result_has_the_exact_frozen_evidence_contract() -> None:
    """Catches a result that hides raw events or omits canonical mechanism facts."""
    from guardedpy.mechanism_demo import ScenarioResult

    assert ScenarioResult.__dataclass_params__.frozen is True
    assert tuple(field.name for field in fields(ScenarioResult)) == (
        "name",
        "status",
        "rule_id",
        "feedback_kind",
        "dispatched_command",
        "event_kinds",
        "workspace_value",
    )


def test_headless_demo_runs_real_policy_feedback_workspace_and_event_mechanisms(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Catches fixed results that skip policy, feedback, event, or workspace execution."""
    from guardedpy.mechanism_demo import run_all_scenarios

    original_state_home = tmp_path / "preexisting-state"
    monkeypatch.setenv("XDG_STATE_HOME", str(original_state_home))

    dangerous, corrected, tdd_denied = run_all_scenarios()

    assert (
        dangerous.name,
        dangerous.status,
        dangerous.rule_id,
        dangerous.dispatched_command,
    ) == ("dangerous_action_denied", "blocked", "command.privileged", False)
    assert "policy_denial" in dangerous.event_kinds

    assert (
        corrected.name,
        corrected.status,
        corrected.feedback_kind,
        corrected.workspace_value,
        corrected.dispatched_command,
    ) == (
        "failure_feedback_corrects",
        "completed",
        "assertion_failure",
        "fixed",
        False,
    )
    assert corrected.event_kinds.index("assertion_feedback") < corrected.event_kinds.index(
        "source_patch"
    )
    assert corrected.event_kinds.index("source_patch") < corrected.event_kinds.index(
        "full_suite_pass"
    )

    assert (
        tdd_denied.name,
        tdd_denied.status,
        tdd_denied.rule_id,
        tdd_denied.dispatched_command,
        tdd_denied.workspace_value,
    ) == ("tdd_source_patch_denied", "blocked", "tdd.red_required", False, "broken")
    assert "policy_denial" in tdd_denied.event_kinds
    assert original_state_home.exists() is False
    assert Path(os.environ["XDG_STATE_HOME"]) == original_state_home


def test_feedback_aware_demo_llm_refuses_to_patch_without_trusted_assertion_feedback() -> None:
    """Catches the corrective mock returning its patch on scripted timing alone."""
    from guardedpy.mechanism_demo import FeedbackAwareDemoLLM

    llm = FeedbackAwareDemoLLM()
    context = LlmContext.minimal()
    llm.complete(context)
    llm.complete(context)

    with pytest.raises(AssertionError, match="trusted assertion feedback"):
        llm.complete(context)


def test_demo_isolates_pytest_controls_and_restores_the_caller_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Catches caller pytest options or plugins changing the fixed mechanism evidence."""
    from guardedpy.mechanism_demo import _isolated_demo_root

    monkeypatch.setenv("PYTEST_ADDOPTS", "--collect-only")
    monkeypatch.setenv("PYTEST_PLUGINS", "untrusted_demo_plugin")
    monkeypatch.setenv("PYTEST_DISABLE_PLUGIN_AUTOLOAD", "caller-value")

    with _isolated_demo_root():
        assert "PYTEST_ADDOPTS" not in os.environ
        assert "PYTEST_PLUGINS" not in os.environ
        assert os.environ["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] == "1"

    assert os.environ["PYTEST_ADDOPTS"] == "--collect-only"
    assert os.environ["PYTEST_PLUGINS"] == "untrusted_demo_plugin"
    assert os.environ["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] == "caller-value"
