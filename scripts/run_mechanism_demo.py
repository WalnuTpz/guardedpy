"""Run and validate the three headless GuardedPy mechanism scenarios."""

from pathlib import Path
import sys


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from guardedpy.mechanism_demo import run_all_scenarios


dangerous, corrected, tdd_denied = run_all_scenarios()

assert (dangerous.name, dangerous.status, dangerous.rule_id) == (
    "dangerous_action_denied",
    "blocked",
    "command.privileged",
)
assert dangerous.dispatched_command is False
assert "policy_denial" in dangerous.event_kinds

assert (corrected.name, corrected.status, corrected.feedback_kind) == (
    "failure_feedback_corrects",
    "completed",
    "assertion_failure",
)
assert corrected.dispatched_command is False
assert corrected.workspace_value == "fixed"
assert corrected.event_kinds.index("assertion_feedback") < corrected.event_kinds.index("source_patch")
assert corrected.event_kinds.index("source_patch") < corrected.event_kinds.index("full_suite_pass")

assert (tdd_denied.name, tdd_denied.status, tdd_denied.rule_id) == (
    "tdd_source_patch_denied",
    "blocked",
    "tdd.red_required",
)
assert tdd_denied.dispatched_command is False

for result in (dangerous, corrected, tdd_denied):
    print(f"{result.name} status={result.status}")
