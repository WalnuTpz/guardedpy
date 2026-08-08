from hashlib import sha256
from pathlib import Path, PurePosixPath
import sys

import pytest
from pydantic import ValidationError
import yaml

from guardedpy.config import (
    HarnessConfig,
    app_state_dir,
    load_or_create_discovered_config,
    local_state_path,
    project_config_path,
    update_future_defaults,
)
from guardedpy.discovery import ProjectProfile


def _profile(root: Path, *, tests: str = "tests") -> ProjectProfile:
    (root / "src").mkdir(parents=True, exist_ok=True)
    (root / tests).mkdir(parents=True, exist_ok=True)
    return ProjectProfile(
        root=root.resolve(),
        discovery_source="tests_dir",
        source_dirs=(PurePosixPath("src"),),
        test_dirs=(PurePosixPath(tests),),
        pytest_command=(sys.executable, "-m", "pytest"),
    )


def test_discovered_config_defaults_are_closed_and_frozen(tmp_path: Path) -> None:
    profile = _profile(tmp_path)
    config = HarnessConfig(profile=profile)

    assert config.profile == profile
    assert config.source_dirs == (PurePosixPath("src"),)
    assert config.test_dirs == (PurePosixPath("tests"),)
    assert config.pytest_command == (sys.executable, "-m", "pytest")
    assert config.model == "deepseek-v4-flash"
    assert config.thinking_enabled is True
    assert config.reasoning_effort == "high"
    assert config.timeout_seconds == 120
    with pytest.raises(ValidationError):
        config.model = "deepseek-v4-pro"


@pytest.mark.parametrize(
    "changes",
    [
        {"model": "deepseek-chat"},
        {"reasoning_effort": "medium"},
        {"thinking_enabled": False},
    ],
)
def test_discovered_config_rejects_unsupported_provider_choices(
    tmp_path: Path, changes: dict[str, object]
) -> None:
    with pytest.raises(ValidationError):
        HarnessConfig(profile=_profile(tmp_path), **changes)


def test_load_or_create_discovered_config_persists_only_nonsecret_task_defaults(
    tmp_path: Path,
) -> None:
    root = tmp_path / "project"
    state_dir = tmp_path / "state"
    root.mkdir()
    profile = _profile(root)

    config = load_or_create_discovered_config(profile, state_dir)

    root_hash = sha256(str(root.resolve()).encode()).hexdigest()[:16]
    snapshot_path = state_dir / root_hash / "harness.yaml"
    assert config.profile == profile
    assert snapshot_path.is_file()
    assert yaml.safe_load(snapshot_path.read_text()) == {
        "profile": {
            "root": str(root.resolve()),
            "discovery_source": "tests_dir",
            "source_dirs": ["src"],
            "test_dirs": ["tests"],
            "pytest_command": [sys.executable, "-m", "pytest"],
        },
        "model": "deepseek-v4-flash",
        "thinking_enabled": True,
        "reasoning_effort": "high",
        "timeout_seconds": 120,
    }
    assert "key" not in snapshot_path.read_text().lower()


def test_load_or_create_uses_current_discovery_and_retains_valid_future_defaults(
    tmp_path: Path,
) -> None:
    root = tmp_path / "project"
    state_dir = tmp_path / "state"
    root.mkdir()
    old_profile = _profile(root)
    old = load_or_create_discovered_config(old_profile, state_dir)
    changed = update_future_defaults(
        old, model="deepseek-v4-pro", reasoning_effort="max"
    ).model_copy(update={"timeout_seconds": 17})
    snapshot = yaml.safe_load(
        (state_dir / sha256(str(root.resolve()).encode()).hexdigest()[:16] / "harness.yaml").read_text()
    )
    snapshot.update(
        {
            "model": changed.model,
            "reasoning_effort": changed.reasoning_effort,
            "timeout_seconds": changed.timeout_seconds,
        }
    )
    snapshot["profile"]["test_dirs"] = ["old-tests"]
    snapshot_path = state_dir / sha256(str(root.resolve()).encode()).hexdigest()[:16] / "harness.yaml"
    snapshot_path.write_text(yaml.safe_dump(snapshot, sort_keys=False))
    new_profile = _profile(root, tests="checks")

    loaded = load_or_create_discovered_config(new_profile, state_dir)

    assert loaded.profile == new_profile
    assert loaded.model == "deepseek-v4-pro"
    assert loaded.reasoning_effort == "max"
    assert loaded.timeout_seconds == 17


def test_update_future_defaults_returns_a_new_frozen_config(tmp_path: Path) -> None:
    original = HarnessConfig(profile=_profile(tmp_path))

    changed = update_future_defaults(
        original, model="deepseek-v4-pro", reasoning_effort="max"
    )

    assert changed is not original
    assert (original.model, original.reasoning_effort) == ("deepseek-v4-flash", "high")
    assert (changed.model, changed.reasoning_effort) == ("deepseek-v4-pro", "max")


@pytest.mark.parametrize("timeout", [4, 121])
def test_config_rejects_timeout_outside_allowed_range(timeout: int) -> None:
    """Catches a configuration change that allows a test timeout outside 5–120 seconds."""
    with pytest.raises(ValidationError):
        HarnessConfig(profile=_profile(Path.cwd()), timeout_seconds=timeout)


def test_app_state_dir_is_outside_project_and_root_isolated(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Catches state storage that lands in a project or aliases distinct roots."""
    state_home = tmp_path / "state-home"
    project_one = tmp_path / "one"
    project_two = tmp_path / "two"
    monkeypatch.setenv("XDG_STATE_HOME", str(state_home))

    first = app_state_dir(project_one)
    second = app_state_dir(project_two)

    assert first.is_relative_to(state_home)
    assert not first.is_relative_to(project_one)
    assert first != second


def test_project_config_and_local_index_paths_stay_in_external_application_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Catches configuration or the selected-project index being written into a project."""
    state_home = tmp_path / "state-home"
    project_root = tmp_path / "project"
    monkeypatch.setenv("XDG_STATE_HOME", str(state_home))

    config_path = project_config_path(project_root)
    index_path = local_state_path()

    assert config_path == app_state_dir(project_root) / "harness.yaml"
    assert config_path.is_relative_to(state_home)
    assert not config_path.is_relative_to(project_root)
    assert index_path == state_home / "guardedpy" / "local-state.yaml"
