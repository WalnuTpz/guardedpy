"""Run the three offline continuous-agent mechanism scenarios."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from guardedpy.mechanism_demo import run_all_scenarios


results = run_all_scenarios()
approval, repaired, stale = results
assert approval.name == "delete_requires_approval"
assert approval.workspace_value == "present"
assert approval.event_kinds[:2] == ("approval_requested", "approval_resolved")
assert repaired.name == "feedback_repair"
assert repaired.workspace_value == "fixed"
assert repaired.event_kinds.index("assertion_failure") < repaired.event_kinds.index("patch_applied")
assert repaired.event_kinds.index("patch_applied") < repaired.event_kinds.index("pytest_passed")
assert stale.name == "stale_approval_denied"
assert stale.stale_approval_denied

for result in results:
    print(f"{result.name} status={result.status}")
