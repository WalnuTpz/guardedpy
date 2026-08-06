from __future__ import annotations

from pathlib import Path, PurePosixPath
import sys

import pytest

from guardedpy.discovery import ProjectDiscoveryError, ProjectProfile, discover_project


def _write_source(root: Path, *, under_src: bool = True) -> None:
    directory = root / "src" if under_src else root
    directory.mkdir(exist_ok=True)
    (directory / "app.py").write_text("VALUE = 1\n")


def test_discover_project_prefers_existing_pytest_testpaths(tmp_path: Path) -> None:
    (tmp_path / "pyproject.toml").write_text(
        "[tool.pytest.ini_options]\ntestpaths = ['checks']\n"
    )
    (tmp_path / "checks").mkdir()
    _write_source(tmp_path)

    profile = discover_project(tmp_path)

    assert profile.root == tmp_path.resolve()
    assert profile.discovery_source == "pytest_config"
    assert profile.source_dirs == (PurePosixPath("src"),)
    assert profile.test_dirs == (PurePosixPath("checks"),)
    assert profile.pytest_command == (sys.executable, "-m", "pytest")


@pytest.mark.parametrize(
    ("filename", "content"),
    [
        ("pytest.ini", "[pytest]\ntestpaths = checks nested/checks\n"),
        ("tox.ini", "[pytest]\ntestpaths = checks nested/checks\n"),
        ("setup.cfg", "[pytest]\ntestpaths = checks nested/checks\n"),
    ],
)
def test_discover_project_reads_supported_ini_testpaths(
    tmp_path: Path, filename: str, content: str
) -> None:
    (tmp_path / filename).write_text(content)
    (tmp_path / "checks").mkdir()
    (tmp_path / "nested" / "checks").mkdir(parents=True)
    _write_source(tmp_path)

    profile = discover_project(tmp_path)

    assert profile.discovery_source == "pytest_config"
    assert profile.test_dirs == (
        PurePosixPath("checks"),
        PurePosixPath("nested/checks"),
    )


def test_first_existing_pytest_config_controls_fallback(tmp_path: Path) -> None:
    (tmp_path / "pyproject.toml").write_text("[tool.pytest.ini_options]\n")
    (tmp_path / "pytest.ini").write_text("[pytest]\ntestpaths = checks\n")
    (tmp_path / "checks").mkdir()
    (tmp_path / "tests").mkdir()
    _write_source(tmp_path)

    profile = discover_project(tmp_path)

    assert profile.discovery_source == "tests_dir"
    assert profile.test_dirs == (PurePosixPath("tests"),)


def test_discover_project_uses_tests_and_src_directories(tmp_path: Path) -> None:
    (tmp_path / "tests").mkdir()
    _write_source(tmp_path)

    profile = discover_project(tmp_path)

    assert profile.discovery_source == "tests_dir"
    assert profile.source_dirs == (PurePosixPath("src"),)
    assert profile.test_dirs == (PurePosixPath("tests"),)


def test_project_profile_rejects_a_non_discovery_source(tmp_path: Path) -> None:
    (tmp_path / "src").mkdir()
    (tmp_path / "tests").mkdir()

    with pytest.raises(ValueError, match="discovery source"):
        ProjectProfile(
            root=tmp_path.resolve(),
            discovery_source="manual",
            source_dirs=(PurePosixPath("src"),),
            test_dirs=(PurePosixPath("tests"),),
            pytest_command=(sys.executable, "-m", "pytest"),
        )


@pytest.mark.parametrize("test_name", ["test_app.py", "app_test.py"])
def test_discover_project_accepts_root_tests_and_a_single_module(
    tmp_path: Path, test_name: str
) -> None:
    (tmp_path / test_name).write_text("def test_value():\n    assert True\n")
    _write_source(tmp_path, under_src=False)

    profile = discover_project(tmp_path)

    assert profile.discovery_source == "root_tests"
    assert profile.source_dirs == (PurePosixPath("."),)
    assert profile.test_dirs == (PurePosixPath("."),)


def test_discover_project_accepts_a_non_test_root_package(tmp_path: Path) -> None:
    (tmp_path / "test_app.py").write_text("def test_value():\n    assert True\n")
    (tmp_path / "package").mkdir()
    (tmp_path / "package" / "__init__.py").write_text("")

    assert discover_project(tmp_path).source_dirs == (PurePosixPath("."),)


@pytest.mark.parametrize(
    ("filename", "content", "expected_code"),
    [
        ("pyproject.toml", "not = [valid", "invalid_pytest_config"),
        (
            "pyproject.toml",
            "[tool.pytest.ini_options]\ntestpaths = 'checks'\n",
            "invalid_pytest_config",
        ),
        (
            "pyproject.toml",
            "[tool.pytest.ini_options]\ntestpaths = []\n",
            "invalid_pytest_config",
        ),
        ("pytest.ini", "[pytest]\ntestpaths =\n", "invalid_pytest_config"),
        ("pytest.ini", "[pytest\ntestpaths = checks\n", "invalid_pytest_config"),
        (
            "pyproject.toml",
            "[tool.pytest.ini_options]\ntestpaths = ['missing']\n",
            "configured_testpath_missing",
        ),
    ],
)
def test_discover_project_maps_pytest_configuration_errors(
    tmp_path: Path, filename: str, content: str, expected_code: str
) -> None:
    (tmp_path / filename).write_text(content)
    (tmp_path / "tests").mkdir()
    _write_source(tmp_path)

    with pytest.raises(ProjectDiscoveryError) as caught:
        discover_project(tmp_path)

    assert caught.value.code == expected_code


@pytest.mark.parametrize("testpath", ["/tmp", "../outside", "checks.py"])
def test_discover_project_rejects_invalid_configured_testpath(
    tmp_path: Path, testpath: str
) -> None:
    candidate = tmp_path / testpath
    if testpath == "checks.py":
        candidate.write_text("")
    (tmp_path / "pyproject.toml").write_text(
        f"[tool.pytest.ini_options]\ntestpaths = [{testpath!r}]\n"
    )
    (tmp_path / "tests").mkdir()
    _write_source(tmp_path)

    with pytest.raises(ProjectDiscoveryError) as caught:
        discover_project(tmp_path)

    assert caught.value.code == "invalid_pytest_config"


def test_discover_project_rejects_configured_symlink_escaping_root(tmp_path: Path) -> None:
    outside = tmp_path.parent / f"{tmp_path.name}-outside"
    outside.mkdir()
    (tmp_path / "linked-checks").symlink_to(outside, target_is_directory=True)
    (tmp_path / "pyproject.toml").write_text(
        "[tool.pytest.ini_options]\ntestpaths = ['linked-checks']\n"
    )
    _write_source(tmp_path)

    with pytest.raises(ProjectDiscoveryError) as caught:
        discover_project(tmp_path)

    assert caught.value.code == "invalid_pytest_config"


def test_discover_project_does_not_read_a_config_symlink_outside_root(tmp_path: Path) -> None:
    outside = tmp_path.parent / f"{tmp_path.name}-outside.toml"
    outside.write_text("[tool.pytest.ini_options]\ntestpaths = ['checks']\n")
    (tmp_path / "pyproject.toml").symlink_to(outside)
    (tmp_path / "checks").mkdir()
    _write_source(tmp_path)

    with pytest.raises(ProjectDiscoveryError) as caught:
        discover_project(tmp_path)

    assert caught.value.code == "invalid_pytest_config"


def test_discover_project_maps_malformed_pytest_table_to_bounded_error(tmp_path: Path) -> None:
    (tmp_path / "pyproject.toml").write_text("tool = []\n")
    (tmp_path / "tests").mkdir()
    _write_source(tmp_path)

    with pytest.raises(ProjectDiscoveryError) as caught:
        discover_project(tmp_path)

    assert caught.value.code == "invalid_pytest_config"


def test_discover_project_does_not_follow_test_or_source_symlinks_outside_root(
    tmp_path: Path,
) -> None:
    outside = tmp_path.parent / f"{tmp_path.name}-outside"
    outside.mkdir()
    (outside / "test_external.py").write_text("def test_external(): pass\n")
    (outside / "app.py").write_text("VALUE = 1\n")
    (tmp_path / "external_tests").symlink_to(outside, target_is_directory=True)
    (tmp_path / "app.py").symlink_to(outside / "app.py")

    with pytest.raises(ProjectDiscoveryError) as caught:
        discover_project(tmp_path)

    assert caught.value.code == "unsupported_project"


def test_discover_project_rejects_tests_only_and_no_evidence_layouts(tmp_path: Path) -> None:
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_only.py").write_text("def test_only(): pass\n")

    with pytest.raises(ProjectDiscoveryError) as tests_only:
        discover_project(tmp_path)
    assert tests_only.value.code == "unsupported_project"

    (tmp_path / "tests" / "test_only.py").unlink()
    (tmp_path / "tests").rmdir()
    _write_source(tmp_path)

    with pytest.raises(ProjectDiscoveryError) as no_tests:
        discover_project(tmp_path)
    assert no_tests.value.code == "unsupported_project"
