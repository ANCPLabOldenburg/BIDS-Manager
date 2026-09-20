"""Validate a dataset's ``CITATION.cff`` file.

``CITATION.cff`` is an optional Citation File Format file. The reference validator
checks it against the full CFF schema; bidsval does the conservative subset that
cannot produce a false positive: the file must be valid YAML, be a mapping, and
carry the keys the CFF format always requires (``cff-version``, ``message``,
``title``). Anything beyond that is left to a future, schema-complete check.
"""

from __future__ import annotations

from ..files import FileTree
from ..issues import Issue, Severity

_REQUIRED = ("cff-version", "message", "title")


def _issue(message: str, suggestion: str) -> Issue:
    return Issue(
        code="CITATION_CFF_VALIDATION_ERROR",
        severity=Severity.ERROR,
        location="CITATION.cff",
        message=message,
        suggestion=suggestion,
    )


def citation_checks(tree: FileTree) -> list[Issue]:
    """Validate ``CITATION.cff`` at the dataset root, if present."""
    cff = tree.get("CITATION.cff")
    if cff is None or cff.is_symlink or cff.size() == 0:
        return []
    try:
        import yaml
    except ImportError:  # pragma: no cover - pyyaml is a declared dependency
        return []
    try:
        text = cff.read_text()
    except OSError:
        return []
    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError:
        return [
            _issue(
                "CITATION.cff is not valid YAML",
                "Fix the YAML syntax. See https://citation-file-format.github.io for the format.",
            )
        ]
    if not isinstance(data, dict):
        return [
            _issue(
                "CITATION.cff must be a YAML mapping of fields",
                "Use top-level keys: cff-version, message, title (and authors).",
            )
        ]
    missing = [key for key in _REQUIRED if key not in data]
    if missing:
        return [
            _issue(
                f"CITATION.cff is missing required key(s): {', '.join(missing)}",
                "Add the required CFF keys: cff-version, message, and title (and authors).",
            )
        ]
    return []
