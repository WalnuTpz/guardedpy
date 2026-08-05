"""Configuration contracts for a selected project root."""

from __future__ import annotations

from hashlib import sha256
import os
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, field_validator
import yaml


class HarnessConfig(BaseModel):
    """Validated configuration snapshot used by a task."""

    model_config = ConfigDict(frozen=True)

    source_dirs: tuple[Path, ...] = Field(min_length=1)
    test_dirs: tuple[Path, ...] = Field(min_length=1)
    pytest_command: tuple[str, ...] = Field(min_length=1)
    model: str = "deepseek-chat"
    timeout_seconds: int = Field(default=30, ge=5, le=120)

    @field_validator("source_dirs", "test_dirs")
    @classmethod
    def paths_stay_inside_project_root(cls, paths: tuple[Path, ...]) -> tuple[Path, ...]:
        for path in paths:
            if path == Path(".") or path.is_absolute() or ".." in path.parts:
                raise ValueError("configured paths must stay inside the project root")
        return paths

    @field_validator("pytest_command")
    @classmethod
    def normalize_pytest_command(cls, tokens: tuple[str, ...]) -> tuple[str, ...]:
        normalized = tuple(token.strip() for token in tokens)
        if any(not token for token in normalized):
            raise ValueError("pytest command tokens must be non-empty")
        return normalized

    @field_validator("model")
    @classmethod
    def normalize_model(cls, model: str) -> str:
        normalized = model.strip()
        if not normalized:
            raise ValueError("model must be non-empty")
        return normalized


def load_config(path: Path, project_root: Path) -> HarnessConfig:
    """Load a project configuration whose configured directories are root-relative."""
    del project_root
    return HarnessConfig.model_validate(yaml.safe_load(path.read_text()))


def app_state_dir(project_root: Path) -> Path:
    """Return the per-project state location outside the selected project."""
    root_hash = sha256(str(project_root.resolve()).encode()).hexdigest()
    return _application_state_root() / root_hash


def project_config_path(project_root: Path) -> Path:
    """Return the external configuration snapshot for one selected project."""
    return app_state_dir(project_root) / "harness.yaml"


def local_state_path() -> Path:
    """Return the external selected-project and task-root index path."""
    return _application_state_root() / "local-state.yaml"


def _application_state_root() -> Path:
    state_home = Path(os.environ.get("XDG_STATE_HOME", Path.home() / ".local" / "state"))
    return state_home / "guardedpy"
