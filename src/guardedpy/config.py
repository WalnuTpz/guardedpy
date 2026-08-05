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

    source_dirs: tuple[Path, ...]
    test_dirs: tuple[Path, ...]
    pytest_command: tuple[str, ...]
    model: str = "deepseek-chat"
    timeout_seconds: int = Field(default=30, ge=5, le=120)

    @field_validator("source_dirs", "test_dirs")
    @classmethod
    def paths_stay_inside_project_root(cls, paths: tuple[Path, ...]) -> tuple[Path, ...]:
        for path in paths:
            if path.is_absolute() or ".." in path.parts:
                raise ValueError("configured paths must stay inside the project root")
        return paths


def load_config(path: Path, project_root: Path) -> HarnessConfig:
    """Load a project configuration whose configured directories are root-relative."""
    del project_root
    return HarnessConfig.model_validate(yaml.safe_load(path.read_text()))


def app_state_dir(project_root: Path) -> Path:
    """Return the per-project state location outside the selected project."""
    state_home = Path(os.environ.get("XDG_STATE_HOME", Path.home() / ".local" / "state"))
    root_hash = sha256(str(project_root.resolve()).encode()).hexdigest()
    return state_home / "guardedpy" / root_hash
