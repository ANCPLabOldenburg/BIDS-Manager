"""Per-file checks that are not expressed in the schema as rules.

These mirror what the reference validator hard-codes (empty files, unreadable
NIfTI headers), but bidsval reports them more usefully: with an explanation of
what the finding does and does not mean, and a machine-actionable fix.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ..files import BIDSFile
from ..issues import Fix, Issue, Severity


def bespoke_checks(
    bids_file: BIDSFile, context: Mapping[str, Any], *, read_headers: bool
) -> list[Issue]:
    """Run the non-schema per-file checks."""
    issues: list[Issue] = []
    location = bids_file.relpath

    # A symlink (e.g. an unfetched git-annex file) has no local content to judge;
    # do not flag it as empty or try to read its header.
    if bids_file.is_symlink:
        return issues

    if bids_file.size() == 0:
        # An empty file has no data, which is a violation (matches the reference).
        issues.append(
            Issue(
                code="EMPTY_FILE",
                severity=Severity.ERROR,
                location=location,
                message="file is empty (0 bytes): it exists but contains no data",
                suggestion=(
                    "The file name and location are valid, but there is no content. Replace it "
                    "with real data. (Some example datasets ship empty placeholder files; those "
                    "datasets are reported invalid for this reason.)"
                ),
                fix=Fix(action="replace_empty_file", label="Provide real data for this file"),
            )
        )
        # The empty-file check does not short-circuit: the reference validator also
        # reports NIFTI_HEADER_UNREADABLE on an empty/truncated NIfTI, so an empty
        # .nii(.gz) yields both EMPTY_FILE and NIFTI_HEADER_UNREADABLE below.

    extension = str(context.get("extension", ""))
    if read_headers and extension.startswith(".nii") and context.get("nifti_header") is None:
        issues.append(
            Issue(
                code="NIFTI_HEADER_UNREADABLE",
                severity=Severity.ERROR,
                location=location,
                message="the NIfTI header could not be read",
                suggestion=(
                    "The NIfTI header could not be read: the file may be empty, truncated, "
                    "compressed oddly, or not a valid NIfTI. Replace it with a valid NIfTI or "
                    "verify it opens in a NIfTI reader."
                ),
                fix=Fix(action="inspect_file", label="Check that the file is a valid NIfTI"),
            )
        )
    return issues
