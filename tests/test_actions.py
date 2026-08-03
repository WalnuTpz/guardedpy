import pytest
from pydantic import ValidationError

from guardedpy.actions import (
    ApplyPatchAction,
    DeletePathAction,
    FinishAction,
    ListFilesAction,
    ReadFileAction,
    RequestApprovalAction,
    RunCommandAction,
    RunPytestAction,
    parse_action,
    stable_hash,
)
from guardedpy.domain import TddPhase


@pytest.mark.parametrize(
    ("payload", "action_type"),
    [
        ('{"kind":"list_files","summary":"inspect","path":"src"}', ListFilesAction),
        ('{"kind":"read_file","summary":"inspect","path":"src/a.py"}', ReadFileAction),
        ('{"kind":"apply_patch","summary":"change","diff":"--- a/x"}', ApplyPatchAction),
        ('{"kind":"delete_path","summary":"remove","path":"tests/a.py"}', DeletePathAction),
        ('{"kind":"run_pytest","summary":"test","targets":["tests/test_a.py"]}', RunPytestAction),
        ('{"kind":"run_command","summary":"format","args":["ruff","check"]}', RunCommandAction),
        ('{"kind":"request_approval","summary":"ask","reason":"delete generated file"}', RequestApprovalAction),
        ('{"kind":"finish","summary":"stop","status":"blocked"}', FinishAction),
    ],
)
def test_parse_action_returns_the_matching_known_action(payload: str, action_type: type) -> None:
    """Catches a parser branch that cannot construct one of the eight supported actions."""
    assert isinstance(parse_action(payload), action_type)


def test_parse_action_rejects_unknown_kind() -> None:
    """Catches an unsafe parser fallback that accepts an unrecognized action kind."""
    with pytest.raises(ValidationError):
        parse_action('{"kind":"shell","summary":"run"}')


def test_stable_hash_ignores_json_object_key_order() -> None:
    """Catches approval hashes that change when equivalent action JSON is reordered."""
    first = parse_action('{"kind":"read_file","summary":"inspect","path":"src/a.py"}')
    second = parse_action('{"path":"src/a.py","summary":"inspect","kind":"read_file"}')

    assert stable_hash(first) == stable_hash(second)
    assert first.stable_hash() == second.stable_hash()


def test_stable_hash_changes_when_action_content_changes() -> None:
    """Catches hashes that omit a safety-relevant action field."""
    first = parse_action('{"kind":"delete_path","summary":"remove","path":"tests/a.py"}')
    second = parse_action('{"kind":"delete_path","summary":"remove","path":"src/a.py"}')

    assert stable_hash(first) != stable_hash(second)


def test_task_state_can_record_tdd_phase(feature_task: object) -> None:
    """Catches a task-state contract that prevents the TDD workflow from progressing."""
    feature_task.tdd_phase = TddPhase.RED_OBSERVED

    assert feature_task.tdd_phase is TddPhase.RED_OBSERVED
