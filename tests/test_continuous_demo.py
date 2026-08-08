"""Offline evidence for Task 22.5's new continuous-agent demo."""

from guardedpy.mechanism_demo import run_all_scenarios, run_scenario, scenario_request


def test_demo_scenarios_use_actual_continuous_events_and_safe_side_effects() -> None:
    rejected, repaired, stale = run_all_scenarios()

    assert rejected.name == "delete_requires_approval"
    assert rejected.status == "completed"
    assert rejected.workspace_value == "present"
    assert rejected.event_kinds[:3] == ("tool_item_completed", "approval_requested", "approval_resolved")
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


def test_demo_scenario_exposes_the_actual_mock_event_stream_for_a_visual_replay() -> None:
    """The interactive demo must render core events, not a hand-written transcript."""
    observed = []

    result = run_scenario("feedback_repair", on_event=observed.append)

    assert observed[0].kind == "user_message"
    assert observed[0].text == scenario_request("feedback_repair")
    assert [event.kind for event in observed].count("tool_item_started") == 4
    assert any(event.data.get("pytest_outcome") == "assertion_failure" for event in observed)
    assert any(event.data.get("changed_paths") for event in observed)
    assert observed[-1].kind == "turn_completed"
    assert result.workspace_value == "fixed"


def test_delete_approval_demo_replays_the_user_decision_through_the_real_agent() -> None:
    approval_events = []

    accepted = run_scenario(
        "delete_requires_approval",
        approval_resolver=lambda event: approval_events.append(event) or True,
    )

    assert len(approval_events) == 1
    assert approval_events[0].kind == "approval_requested"
    assert accepted.workspace_value == "<deleted>"


def test_delete_demo_reads_the_target_before_requesting_approval() -> None:
    observed = []

    run_scenario("delete_requires_approval", on_event=observed.append)

    read_completed = next(
        index for index, event in enumerate(observed)
        if event.kind == "tool_item_completed" and event.data.get("tool") == "read_file"
    )
    approval_requested = next(index for index, event in enumerate(observed) if event.kind == "approval_requested")
    assert read_completed < approval_requested
