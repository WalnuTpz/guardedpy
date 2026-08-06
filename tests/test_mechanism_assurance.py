"""Course A.6 evidence must be produced by the shared provider-free core."""

from __future__ import annotations

import pytest

from guardedpy.context import LlmContext
from guardedpy.mechanism_demo import FeedbackAwareDemoLLM, run_scenario


def test_feedback_aware_demo_refuses_patch_without_assertion_feedback() -> None:
    """Catches the demo mock handing out its repair independently of trusted feedback."""
    with pytest.raises(AssertionError, match="assertion_failure"):
        FeedbackAwareDemoLLM().complete(LlmContext.minimal())


def test_feedback_aware_demo_refuses_empty_assertion_repair_set() -> None:
    """Catches a kind-only feedback check accepting no actual failing pytest node."""
    context = LlmContext.minimal().model_copy(
        update={
            "trusted_data": {
                "feedback": {
                    "type": "pytest_feedback",
                    "kind": "assertion_failure",
                    "node_ids": (),
                }
            }
        }
    )

    with pytest.raises(AssertionError, match="assertion_failure"):
        FeedbackAwareDemoLLM().complete(context)


def test_feedback_aware_demo_repairs_only_after_core_feedback_and_finishes_green() -> None:
    """Catches presentation-only evidence that does not execute feedback, patch, and suite."""
    result = run_scenario("failure_feedback_corrects")

    assert result.status == "completed"
    assert result.feedback_kind == "assertion_failure"
    assert result.workspace_value == "fixed"
    assert result.event_kinds.index("assertion_feedback") < result.event_kinds.index(
        "source_patch"
    )
    assert result.event_kinds.index("source_patch") < result.event_kinds.index(
        "full_suite_pass"
    )
    assert result.dispatched_command is False
