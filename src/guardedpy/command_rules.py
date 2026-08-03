"""Constrained, project-scoped persistent command approvals."""

from __future__ import annotations

from hashlib import sha256
import json
import os
from pathlib import Path
import re
from tempfile import NamedTemporaryFile
from threading import RLock

from guardedpy.actions import RunCommandAction
from guardedpy.config import app_state_dir
from guardedpy.domain import CommandApprovalRule, CommandRuleKind


_GIT_DIFF_CHECK = ("git", "diff", "--no-ext-diff", "--check")
_PIP_INSTALL_PREFIX = ("python", "-m", "pip", "install")
_BRANCH = re.compile(r"[A-Za-z0-9](?:[A-Za-z0-9._/-]*[A-Za-z0-9_-])?")
_PACKAGE_NAME = re.compile(r"[A-Za-z0-9](?:[A-Za-z0-9._-]*[A-Za-z0-9])?")
_VERSION = (
    r"(?:[0-9]+!)?[0-9]+(?:\.[0-9]+)*"
    r"(?:(?:a|b|rc)[0-9]+)?(?:\.post[0-9]+)?(?:\.dev[0-9]+)?"
    r"(?:\+[A-Za-z0-9]+(?:[._-][A-Za-z0-9]+)*)?"
)
_VERSION_CONSTRAINT = re.compile(rf"(?:==|~=|!=|<=|>=|<|>){_VERSION}")
_ARCHIVE_SUFFIXES = (
    ".whl",
    ".zip",
    ".tar",
    ".tar.gz",
    ".tar.bz2",
    ".tar.xz",
    ".tgz",
    ".tbz",
    ".tbz2",
    ".txz",
    ".egg",
)
_SHELL_METACHARACTERS = frozenset(";&|<>$`(){}*?[]\n\r\0\\")


def has_shell_metacharacter(arguments: tuple[str, ...]) -> bool:
    """Return whether an argument contains shell syntax or escape characters."""
    package_command = arguments[:4] == _PIP_INSTALL_PREFIX
    return any(
        character in _SHELL_METACHARACTERS
        for index, argument in enumerate(arguments)
        if not (package_command and index >= 4 and _valid_package_spec(argument))
        for character in argument
    )


def command_rule_kind(
    action: RunCommandAction, current_branch: str | None
) -> CommandRuleKind | None:
    """Classify an action only when it obeys one exact safe family grammar."""
    if has_shell_metacharacter(action.args):
        return None
    if action.args == _GIT_DIFF_CHECK:
        return CommandRuleKind.GIT_DIFF_CHECK
    if (
        len(action.args) == 4
        and action.args[:3] == ("git", "push", "origin")
        and current_branch is not None
        and action.args[3] == current_branch
        and _valid_branch(current_branch)
    ):
        return CommandRuleKind.GIT_PUSH
    if (
        action.args[:4] == _PIP_INSTALL_PREFIX
        and len(action.args) > 4
        and all(_valid_package_spec(spec) for spec in action.args[4:])
    ):
        return CommandRuleKind.PIP_INSTALL
    return None


class CommandRuleStore:
    """Atomically persist structured approvals in one project's app state."""

    def __init__(self, project_root: Path) -> None:
        self._project_root = project_root.resolve()
        self._project_hash = sha256(str(self._project_root).encode()).hexdigest()
        self._path = app_state_dir(self._project_root) / "command-rules.json"
        self._lock = RLock()
        self._rules = self._load()

    def list_rules(self) -> list[CommandApprovalRule]:
        """Return durable rules in deterministic identifier order."""
        with self._lock:
            self._rules = self._load()
            return sorted(self._rules.values(), key=lambda rule: rule.id)

    @property
    def project_root(self) -> Path:
        """Return the resolved root this store is constrained to."""
        return self._project_root

    def add_from(
        self, action: RunCommandAction, current_branch: str | None
    ) -> CommandApprovalRule:
        """Derive and persist one immutable rule from an eligible command action."""
        rule = self._derive(action, current_branch)
        if rule is None:
            raise ValueError("action is not an eligible command family")
        with self._lock:
            self._rules = self._load()
            if rule.id not in self._rules:
                self._rules[rule.id] = rule
                self._save()
            return self._rules[rule.id]

    def matches(self, action: RunCommandAction, current_branch: str | None) -> bool:
        """Match only the same constrained rule in this root and current branch."""
        candidate = self._derive(action, current_branch)
        if candidate is None:
            return False
        with self._lock:
            self._rules = self._load()
            return self._rules.get(candidate.id) == candidate

    def delete(self, rule_id: str) -> bool:
        """Revoke one rule by its opaque identifier."""
        with self._lock:
            self._rules = self._load()
            if rule_id not in self._rules:
                return False
            del self._rules[rule_id]
            self._save()
            return True

    def _derive(
        self, action: RunCommandAction, current_branch: str | None
    ) -> CommandApprovalRule | None:
        kind = command_rule_kind(action, current_branch)
        if kind is None:
            return None
        branch = current_branch if kind is CommandRuleKind.GIT_PUSH else None
        package_specs = action.args[4:] if kind is CommandRuleKind.PIP_INSTALL else ()
        canonical = json.dumps(
            {
                "branch": branch,
                "kind": kind.value,
                "package_specs": package_specs,
                "project_hash": self._project_hash,
            },
            separators=(",", ":"),
            sort_keys=True,
        )
        return CommandApprovalRule(
            id=sha256(canonical.encode()).hexdigest(),
            kind=kind,
            project_hash=self._project_hash,
            branch=branch,
            package_specs=package_specs,
        )

    def _load(self) -> dict[str, CommandApprovalRule]:
        if not self._path.exists():
            return {}
        records = json.loads(self._path.read_text())
        rules = {
            record["id"]: CommandApprovalRule(
                id=record["id"],
                kind=CommandRuleKind(record["kind"]),
                project_hash=record["project_hash"],
                branch=record["branch"],
                package_specs=tuple(record["package_specs"]),
            )
            for record in records
        }
        if any(rule.project_hash != self._project_hash for rule in rules.values()):
            raise ValueError("command rule belongs to another project root")
        return rules

    def _save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        records = [
            {
                "id": rule.id,
                "kind": rule.kind.value,
                "project_hash": rule.project_hash,
                "branch": rule.branch,
                "package_specs": rule.package_specs,
            }
            for rule in sorted(self._rules.values(), key=lambda item: item.id)
        ]
        temporary_path: Path | None = None
        try:
            with NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=self._path.parent,
                prefix=".command-rules-",
                suffix=".tmp",
                delete=False,
            ) as temporary:
                json.dump(records, temporary, separators=(",", ":"), sort_keys=True)
                temporary.flush()
                os.fsync(temporary.fileno())
                temporary_path = Path(temporary.name)
            os.replace(temporary_path, self._path)
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)


def _valid_branch(branch: str) -> bool:
    return bool(
        _BRANCH.fullmatch(branch)
        and ".." not in branch
        and "//" not in branch
        and "@{" not in branch
        and not branch.endswith((".", "/", ".lock"))
        and all(not part.startswith(".") for part in branch.split("/"))
    )


def _valid_package_spec(spec: str) -> bool:
    operator = next((index for index, character in enumerate(spec) if character in "<>=!~"), None)
    name = spec if operator is None else spec[:operator]
    if not _PACKAGE_NAME.fullmatch(name) or name.lower().endswith(_ARCHIVE_SUFFIXES):
        return False
    if operator is None:
        return True
    return all(_VERSION_CONSTRAINT.fullmatch(part) for part in spec[operator:].split(","))
