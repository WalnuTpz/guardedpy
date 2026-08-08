"""CLI-only delivery boundary and executable headless evidence."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def test_retired_non_cli_surfaces_are_absent() -> None:
    """Catches a distribution retaining an importable or deployable Web surface."""
    retired = (
        "src/guardedpy/api.py",
        "src/guardedpy/web.py",
        "src/guardedpy/demo.py",
        "src/guardedpy/templates",
        "src/guardedpy/static",
        "render.yaml",
    )

    assert [path for path in retired if (ROOT / path).exists()] == []


def test_headless_mechanism_demo_has_no_web_or_provider_dependency() -> None:
    """Catches mechanism evidence that cannot execute as a provider-free script."""
    completed = subprocess.run(
        [sys.executable, "scripts/run_mechanism_demo.py"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
        env={name: value for name, value in os.environ.items() if name != "PYTHONPATH"},
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.splitlines() == [
        "dangerous_action_denied status=blocked",
        "failure_feedback_corrects status=completed",
        "tdd_source_patch_denied status=blocked",
    ]
