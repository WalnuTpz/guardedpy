"""Deterministic, bounded feedback extracted from one pytest invocation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path, PurePosixPath
import re

from guardedpy.domain import FeedbackKind


_MAX_EXCERPT_CHARS = 800
_FAILED_NODE = re.compile(r"^FAILED (?P<node>\S+)(?P<detail>[^\n]*)$", re.MULTILINE)
_COLLECTION_NODE = re.compile(r"^ERROR collecting (?P<node>\S+)(?:\s|$)", re.MULTILINE)
_USEFUL_OUTPUT = re.compile(
    r"^(FAILED |ERROR collecting |INTERNALERROR|E\s{7}|.*(?:AssertionError|assert ).*)"
)
_ASSERTION_EVIDENCE = re.compile(r"^E\s+(?:AssertionError\b|assert\b)", re.MULTILINE)
_ASSERTION_SUMMARY = re.compile(r"(?:AssertionError\b|\bassert(?:ion failed)?\b)", re.IGNORECASE)


@dataclass(frozen=True)
class PytestRun:
    """The unprocessed result of running the configured pytest command."""

    exit_code: int
    stdout: str
    stderr: str
    timed_out: bool


@dataclass(frozen=True)
class PytestFeedback:
    """The bounded pytest facts safe to return to the orchestration loop."""

    kind: FeedbackKind
    node_ids: tuple[str, ...]
    excerpt: str


class FeedbackCollector:
    """Classify pytest output without returning the full, untrusted report."""

    def collect(self, run: PytestRun) -> PytestFeedback:
        output = "\n".join(part for part in (run.stdout, run.stderr) if part)
        if run.timed_out:
            return PytestFeedback(FeedbackKind.TIMEOUT, (), self._excerpt(output))
        if run.exit_code == 0:
            return PytestFeedback(FeedbackKind.PASSED, (), "")

        collection_nodes = self._node_ids(_COLLECTION_NODE, output)
        if collection_nodes:
            return PytestFeedback(
                FeedbackKind.COLLECTION_ERROR, collection_nodes, self._excerpt(output)
            )

        failed_matches = tuple(_FAILED_NODE.finditer(output))
        failed_nodes = tuple(match.group("node") for match in failed_matches)
        if failed_nodes:
            excerpt = self._excerpt(output)
            per_node_assertions = tuple(
                bool(_ASSERTION_SUMMARY.search(match.group("detail")))
                for match in failed_matches
            )
            all_assertions = all(per_node_assertions) or (
                len(failed_nodes) == 1 and _ASSERTION_EVIDENCE.search(excerpt) is not None
            )
            if not all_assertions:
                return PytestFeedback(FeedbackKind.EXECUTION_ERROR, failed_nodes, excerpt)
            return PytestFeedback(
                FeedbackKind.ASSERTION_FAILURE, failed_nodes, excerpt
            )
        return PytestFeedback(FeedbackKind.EXECUTION_ERROR, (), self._excerpt(output))

    @staticmethod
    def normalize_nodes(
        feedback: PytestFeedback,
        project_root: Path,
        test_dirs: tuple[PurePosixPath, ...],
    ) -> PytestFeedback:
        """Keep ordered, existing pytest nodes contained by configured test roots."""
        root = project_root.resolve()
        roots = tuple((root / directory).resolve() for directory in test_dirs)
        normalized: list[str] = []
        for node_id in feedback.node_ids:
            path_text, separator, suffix = node_id.partition("::")
            candidate = (root / path_text).resolve()
            if (
                candidate.is_file()
                and candidate.is_relative_to(root)
                and any(candidate.is_relative_to(test_root) for test_root in roots)
            ):
                projected = candidate.relative_to(root).as_posix()
                if separator:
                    projected = f"{projected}::{suffix}"
                if projected not in normalized:
                    normalized.append(projected)
        return PytestFeedback(feedback.kind, tuple(normalized), feedback.excerpt)

    @staticmethod
    def _node_ids(pattern: re.Pattern[str], output: str) -> tuple[str, ...]:
        return tuple(match.group("node") for match in pattern.finditer(output))

    @staticmethod
    def _excerpt(output: str) -> str:
        useful_lines = [line for line in output.splitlines() if _USEFUL_OUTPUT.match(line)]
        if not useful_lines:
            useful_lines = output.splitlines()[-1:]
        return "\n".join(useful_lines)[:_MAX_EXCERPT_CHARS]
