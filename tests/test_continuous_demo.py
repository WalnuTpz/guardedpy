"""Offline evidence for Task 22.5's new continuous-agent demo."""

from guardedpy.mechanism_demo import run_all_scenarios, run_scenario


def test_demo_scenarios_use_actual_continuous_events_and_safe_side_effects() -> None:
    rejected, repaired, stale = run_all_scenarios()

    assert rejected.name == "delete_approval_rejected"
    assert rejected.status == "completed"
    assert rejected.workspace_value == "present"
    assert rejected.event_kinds[:2] == ("approval_requested", "approval_resolved")
    assert "tool_item_completed" in rejected.event_kinds

    assert repaired.name == "feedback_repair"
    assert repaired.status == "completed"
    assert repaired.workspace_value == "fixed"
    assert repaired.event_kinds.index("assertion_failure") < repaired.event_kinds.index("patch_applied")
    assert repaired.event_kinds.index("patch_applied") < repaired.event_kinds.index("pytest_passed")

    assert stale.name == "stale_approval_denied"
    assert stale.stale_approval_denied is True
    assert stale.workspace_value == "present"


def test_unknown_continuous_demo_scenario_is_rejected() -> None:
    try:
        run_scenario("unknown")  # type: ignore[arg-type]
    except KeyError:
        pass
    else:
        raise AssertionError("unknown scenario must not run")
