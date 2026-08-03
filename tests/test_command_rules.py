"""Offline coverage for constrained, project-scoped command approval rules."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from guardedpy.actions import RunCommandAction
from guardedpy.command_rules import CommandRuleStore
from guardedpy.config import app_state_dir
from guardedpy.domain import CommandApprovalRule, CommandRuleKind


def _command(*args: str, summary: str = "approved command") -> RunCommandAction:
    return RunCommandAction(kind="run_command", summary=summary, args=args)


@pytest.mark.parametrize(
    ("action", "current_branch", "kind"),
    [
        (_command("git", "diff", "--no-ext-diff", "--check"), None, CommandRuleKind.GIT_DIFF_CHECK),
        (_command("git", "push", "origin", "main"), "main", CommandRuleKind.GIT_PUSH),
        (
            _command("python", "-m", "pip", "install", "ruff==0.5.0", "httpx[http2]>=0.27"),
            None,
            CommandRuleKind.PIP_INSTALL,
        ),
    ],
)
def test_exact_command_family_rule_survives_restart(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    action: RunCommandAction,
    current_branch: str | None,
    kind: CommandRuleKind,
) -> None:
    """Catches an approved constrained family being stored only in process memory."""
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    project = tmp_path / "project"
    project.mkdir()

    rule = CommandRuleStore(project).add_from(action, current_branch)
    restarted = CommandRuleStore(project)

    assert rule.kind is kind
    assert restarted.matches(action, current_branch) is True
    assert restarted.list_rules() == [rule]


def test_rule_is_immutable_listable_and_revocable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Catches mutable or undeletable permission state silently widening later commands."""
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    store = CommandRuleStore(tmp_path)
    action = _command("git", "diff", "--no-ext-diff", "--check")
    rule = store.add_from(action, None)

    with pytest.raises(FrozenInstanceError):
        rule.kind = CommandRuleKind.PIP_INSTALL  # type: ignore[misc]

    assert isinstance(rule, CommandApprovalRule)
    assert store.delete(rule.id) is True
    assert store.list_rules() == []
    assert store.matches(action, None) is False


def test_rules_are_isolated_by_resolved_project_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Catches a permanent approval leaking to a different selected project."""
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()
    action = _command("python", "-m", "pip", "install", "ruff")

    CommandRuleStore(first).add_from(action, None)

    assert CommandRuleStore(first).matches(action, None) is True
    assert CommandRuleStore(second).matches(action, None) is False
    assert CommandRuleStore(second).list_rules() == []


def test_changed_branch_never_matches_git_push_rule(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Catches a push permission following its old branch or matching a new branch."""
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    store = CommandRuleStore(tmp_path)
    main_push = _command("git", "push", "origin", "main")
    store.add_from(main_push, "main")

    assert store.matches(main_push, "feature") is False
    assert store.matches(_command("git", "push", "origin", "feature"), "feature") is False


@pytest.mark.parametrize(
    "action",
    [
        _command("git", "diff", "--no-ext-diff", "--check", "README.md"),
        _command("git", "push", "--force", "origin", "main"),
        _command("git", "push", "--tags", "origin", "main"),
        _command("git", "push", "origin", "main:release"),
        _command("git", "-c", "core.hooksPath=/tmp/hooks", "push", "origin", "main"),
        _command("git", "push", "origin", "main", "--no-verify"),
        _command("python", "-m", "pip", "install", "https://example.invalid/pkg.whl"),
        _command("python", "-m", "pip", "install", "../local-package"),
        _command("python", "-m", "pip", "install", "-e", "."),
        _command("python", "-m", "pip", "install", "-r", "requirements.txt"),
        _command("python", "-m", "pip", "install", "--index-url", "https://example.invalid"),
        _command("python", "-m", "pip", "install", "ruff;id"),
        _command("python", "-m", "pip", "install", "ruff==1.*"),
        _command("python", "-m", "pip", "install"),
    ],
)
def test_unsafe_or_expanded_command_cannot_become_a_rule(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    action: RunCommandAction,
) -> None:
    """Catches permanent approval accepting options, paths, URLs, refspecs, or metacharacters."""
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))

    with pytest.raises(ValueError, match="eligible command family"):
        CommandRuleStore(tmp_path).add_from(action, "main")


def test_persisted_rule_omits_raw_action_summary_and_unsafe_values(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Catches app state retaining raw pending actions or user-facing secret text."""
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    summary_secret = "pending sk-task3-secret command"
    action = _command("python", "-m", "pip", "install", "ruff", summary=summary_secret)

    CommandRuleStore(tmp_path).add_from(action, None)

    state_files = list(app_state_dir(tmp_path).glob("*.json"))
    assert len(state_files) == 1
    persisted = state_files[0].read_text()
    assert summary_secret not in persisted
    assert '"summary"' not in persisted
    assert '"args"' not in persisted
