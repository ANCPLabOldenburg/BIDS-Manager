"""Per-file integrity checks that the schema does not express as rules.

These catch files that are structurally broken before any schema rule can apply: a
JSON sidecar that is not valid JSON, or a TSV table that is not a clean rectangle.
The reference validator reports these (``JSON_INVALID``, ``TSV_EQUAL_ROWS`` ...);
bidsval reports them too, with an explanation of what a correct file looks like.

They are deliberately conservative: a finding fires only on a genuine structural
problem, never on a merely unusual-but-valid file, and a symlink (an unfetched
git-annex file with no local content) is skipped.
"""

from __future__ import annotations

import gzip
from collections.abc import Mapping
from typing import Any

from ..files import BIDSFile
from ..issues import Fix, Issue, Severity

# Read at most this many lines when scanning a TSV's structure. Beyond it the file
# is assumed to be a large data table; the partial scan still reports real problems
# and never invents one.
_TSV_LINE_CAP = 100_000

_JSON_ERROR_DETAIL = {
    "JSON_INVALID": (
        "the file is not valid JSON",
        'A BIDS JSON file must be a single JSON object. Example: {"RepetitionTime": 2.0}. '
        "Check for a trailing comma, a missing quote, or a missing brace.",
    ),
    "JSON_NOT_AN_OBJECT": (
        "the JSON does not contain an object",
        'A BIDS JSON file must contain one object {...}, not a list or a bare value. '
        'Example: {"Name": "My dataset", "BIDSVersion": "1.11.1"}.',
    ),
    "INVALID_FILE_ENCODING": (
        "the file is not valid UTF-8 text",
        "Re-save the file using UTF-8 encoding (the encoding BIDS text files require).",
    ),
    "FILE_READ": (
        "the file could not be read",
        "Check that the file exists and is readable (permissions, broken link).",
    ),
}


def integrity_checks(bids_file: BIDSFile, context: Mapping[str, Any]) -> list[Issue]:
    """Run the non-schema structural checks for one file."""
    if bids_file.is_symlink or bids_file.size() == 0:
        return []  # no local content to judge (empty files are reported elsewhere)
    extension = str(context.get("extension", ""))
    if extension == ".json":
        return _json_integrity(bids_file, context)
    if extension == ".tsv":  # gzipped TSVs are not scanned line-by-line
        return _tsv_integrity(bids_file)
    if extension == ".tsv.gz":
        return _gzip_integrity(bids_file)
    return []


def _gzip_integrity(bids_file: BIDSFile) -> list[Issue]:
    """Confirm a gzipped TSV decompresses (matches the reference's INVALID_GZIP, which
    is raised for `.tsv.gz`; a corrupt `.nii.gz` is covered by the NIfTI header check
    instead, so it is not checked here)."""
    try:
        with gzip.open(bids_file.abspath, "rb") as handle:
            while handle.read(1 << 20):  # decompress fully so truncation is caught
                pass
    except (OSError, EOFError):
        return [
            Issue(
                code="INVALID_GZIP",
                severity=Severity.ERROR,
                location=bids_file.relpath,
                message=f"{bids_file.name}: the gzip stream could not be decompressed",
                suggestion=(
                    "The file is named .tsv.gz but is not a valid gzip stream (it may be "
                    "truncated or corrupted). Re-create it with proper gzip compression."
                ),
            )
        ]
    return []


def _json_integrity(bids_file: BIDSFile, context: Mapping[str, Any]) -> list[Issue]:
    error = context.get("__json_error__")
    if not error:
        return []
    reason, suggestion = _JSON_ERROR_DETAIL.get(
        error, ("the file could not be used", "Check the file's contents.")
    )
    return [
        Issue(
            code=error,
            severity=Severity.ERROR,
            location=bids_file.relpath,
            message=f"{bids_file.name}: {reason}",
            suggestion=suggestion,
            fix=Fix(action="fix_json", label="Make the file a valid JSON object"),
        )
    ]


def _tsv_integrity(bids_file: BIDSFile) -> list[Issue]:
    lines = _read_lines(bids_file, _TSV_LINE_CAP)
    if not lines:
        return []
    # A trailing newline (or a final blank line) is allowed and common; drop any
    # trailing empty lines so they are not flagged. Interior blank lines remain.
    while lines and lines[-1] == "":
        lines.pop()
    if not lines:
        return []
    location = bids_file.relpath
    issues: list[Issue] = []

    header = lines[0].split("\t")
    n_cols = len(header)

    seen: set[str] = set()
    for name in header:
        if name in seen:
            issues.append(
                Issue(
                    code="TSV_COLUMN_HEADER_DUPLICATE",
                    sub_code=name,
                    severity=Severity.ERROR,
                    location=location,
                    message=f"duplicate column header {name!r} in the first row",
                    suggestion=(
                        "Each column header must be unique. Rename or remove the "
                        f"duplicate {name!r} so every column has a distinct name."
                    ),
                )
            )
        seen.add(name)

    for offset, line in enumerate(lines[1:], start=2):  # 1-based, +1 for the header row
        if line == "":
            issues.append(
                Issue(
                    code="TSV_EMPTY_LINE",
                    severity=Severity.ERROR,
                    location=location,
                    line=offset,
                    message=f"empty line at line {offset}",
                    suggestion=(
                        "Remove the blank line. Every row must have one value per column "
                        "(use 'n/a' for a missing value)."
                    ),
                )
            )
            continue
        width = len(line.split("\t"))
        if width != n_cols:
            issues.append(
                Issue(
                    code="TSV_EQUAL_ROWS",
                    severity=Severity.ERROR,
                    location=location,
                    line=offset,
                    message=(
                        f"line {offset} has {width} value(s) but the header has {n_cols} column(s)"
                    ),
                    suggestion=(
                        f"Every row must have exactly {n_cols} tab-separated value(s), one per "
                        "column. Use 'n/a' for any value that is missing."
                    ),
                )
            )
            break  # one ragged-row finding per file is enough to act on
    return issues


def _read_lines(bids_file: BIDSFile, cap: int) -> list[str] | None:
    """Read up to ``cap`` lines of a text file, stripped of line endings.

    Returns ``None`` if the file cannot be read as UTF-8 text (such a TSV is not
    flagged here; it is malformed in a way outside this check's scope)."""
    lines: list[str] = []
    try:
        with bids_file.abspath.open("r", encoding="utf-8") as handle:
            for index, line in enumerate(handle):
                if index >= cap:
                    break
                lines.append(line.rstrip("\n").rstrip("\r"))
    except (OSError, UnicodeDecodeError):
        return None
    return lines
