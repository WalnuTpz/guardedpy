from pathlib import Path, PurePosixPath
import sys

from guardedpy.config import HarnessConfig
from guardedpy.discovery import ProjectProfile


def safe_config(tmp_path: Path) -> HarnessConfig:
    (tmp_path / "src").mkdir(parents=True, exist_ok=True)
    (tmp_path / "tests").mkdir(parents=True, exist_ok=True)
    return HarnessConfig(
        profile=ProjectProfile(
            root=tmp_path.resolve(),
            discovery_source="tests_dir",
            source_dirs=(PurePosixPath("src"),),
            test_dirs=(PurePosixPath("tests"),),
            pytest_command=(sys.executable, "-m", "pytest"),
        )
    )
