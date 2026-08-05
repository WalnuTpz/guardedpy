from pathlib import Path

import pytest
from pydantic import ValidationError
import yaml

from guardedpy.config import (
    HarnessConfig,
    app_state_dir,
    load_config,
    local_state_path,
    project_config_path,
)


def test_config_rejects_parent_escape(tmp_path: Path) -> None:
    """Catches a configuration change that permits escaping the selected root."""
    (tmp_path / "harness.yaml").write_text(
        "source_dirs: ['../x']\ntest_dirs: [tests]\npytest_command: [pytest]\n"
    )

    with pytest.raises(ValidationError, match="inside the project root"):
        load_config(tmp_path / "harness.yaml", tmp_path)


def test_config_rejects_absolute_directory(tmp_path: Path) -> None:
    """Catches a configuration change that accepts an absolute source directory."""
    (tmp_path / "harness.yaml").write_text(
        "source_dirs: ['/tmp/x']\ntest_dirs: [tests]\npytest_command: [pytest]\n"
    )

    with pytest.raises(ValidationError, match="inside the project root"):
        load_config(tmp_path / "harness.yaml", tmp_path)


@pytest.mark.parametrize("timeout", [4, 121])
def test_config_rejects_timeout_outside_allowed_range(timeout: int) -> None:
    """Catches a configuration change that allows a test timeout outside 5–120 seconds."""
    with pytest.raises(ValidationError):
        HarnessConfig(
            source_dirs=("src",),
            test_dirs=("tests",),
            pytest_command=("pytest",),
            timeout_seconds=timeout,
        )


def test_load_config_returns_relative_directory_configuration(tmp_path: Path) -> None:
    """Catches a loader that loses the configured project-relative boundaries."""
    config_file = tmp_path / "harness.yaml"
    config_file.write_text(
        "source_dirs: [src]\ntest_dirs: [tests]\npytest_command: [pytest, -q]\n"
    )

    config = load_config(config_file, tmp_path)

    assert config.source_dirs == (Path("src"),)
    assert config.test_dirs == (Path("tests"),)
    assert config.pytest_command == ("pytest", "-q")


@pytest.mark.parametrize(
    "snapshot",
    [
        {"source_dirs": [], "test_dirs": ["tests"], "pytest_command": ["pytest"]},
        {"source_dirs": ["src"], "test_dirs": [], "pytest_command": ["pytest"]},
        {"source_dirs": ["src"], "test_dirs": ["tests"], "pytest_command": []},
        {"source_dirs": [""], "test_dirs": ["tests"], "pytest_command": ["pytest"]},
        {"source_dirs": ["src"], "test_dirs": ["tests"], "pytest_command": ["pytest", " "]},
        {"source_dirs": ["src"], "test_dirs": ["tests"], "pytest_command": ["pytest"], "model": "  "},
    ],
)
def test_load_config_rejects_empty_required_values(tmp_path: Path, snapshot: dict[str, object]) -> None:
    """Catches restored configuration accepting an empty required value."""
    config_file = tmp_path / "harness.yaml"
    config_file.write_text(yaml.safe_dump(snapshot))

    with pytest.raises(ValidationError):
        load_config(config_file, tmp_path)


def test_load_config_normalizes_command_tokens_and_model(tmp_path: Path) -> None:
    """Catches persisted command/model whitespace becoming part of the runtime configuration."""
    config_file = tmp_path / "harness.yaml"
    config_file.write_text(
        yaml.safe_dump(
            {
                "source_dirs": ["src"],
                "test_dirs": ["tests"],
                "pytest_command": [" pytest ", " -q "],
                "model": " deepseek-chat ",
            }
        )
    )

    config = load_config(config_file, tmp_path)

    assert config.pytest_command == ("pytest", "-q")
    assert config.model == "deepseek-chat"


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
