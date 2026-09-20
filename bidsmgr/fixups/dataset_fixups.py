"""Repairs a whole dataset can be given, at conversion time or afterwards.

The same repairs are wanted at two moments. Right after a conversion, when the
dataset is fresh and the user would rather not be told about a missing
``events.tsv`` at all; and later in the Editor, on a dataset that arrived from
somewhere else. Writing them twice would guarantee they drift, so they are
written once here and both surfaces call in.

Everything is **off by default**. Each of these adds content to a dataset or
moves a field between files, and a tool that quietly does either is a tool
whose output you cannot trust. The user turns them on in Settings, once, having
read what they do.

Order matters and is fixed: companion files first, then the citation file. The
citation file is written from ``dataset_description.json``, and the metadata
step that fills the description in has already run by the time this does.

Qt-free, so the conversion chain, the CLI and the Editor share one path.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field as dc_field
from pathlib import Path

log = logging.getLogger(__name__)


@dataclass
class FixupReport:
    """What ran, and what it did. Rendered into the conversion log."""

    companions_written: list[Path] = dc_field(default_factory=list)
    citation_written: bool = False
    skipped: list[str] = dc_field(default_factory=list)
    failed: list[tuple[str, str]] = dc_field(default_factory=list)

    @property
    def did_anything(self) -> bool:
        return bool(self.companions_written or self.citation_written)

    def lines(self) -> list[str]:
        """Human-readable progress lines, one per outcome."""
        out: list[str] = []
        if self.companions_written:
            out.append(
                f"fixups: generated {len(self.companions_written)} companion "
                "file(s). Stubs carry TODO rows and are reported by "
                "validation, an events table as an error, until filled in"
            )
        if self.citation_written:
            out.append(
                "fixups: wrote CITATION.cff and moved the fields it owns out "
                "of dataset_description.json"
            )
        out.extend(f"fixups: skipped, {why}" for why in self.skipped)
        out.extend(f"fixups: {what} failed, {why}" for what, why in self.failed)
        return out


def run_dataset_fixups(
    bids_root: Path,
    *,
    generate_companions: bool = False,
    write_citation_file: bool = False,
) -> FixupReport:
    """Apply the enabled repairs to one BIDS root.

    Each repair is independent: one failing does not stop the next, because a
    dataset that got its citation file and not its events tables is better off
    than one that got neither.
    """
    root = Path(bids_root)
    report = FixupReport()
    if not root.is_dir():
        report.skipped.append(f"{root} is not a directory")
        return report

    if generate_companions:
        try:
            from .associations import find_missing, generate

            missing = find_missing(root)
            if missing:
                created, failed = generate(root, missing)
                report.companions_written.extend(created)
                report.failed.extend(
                    (str(p), why) for p, why in failed
                )
        except Exception as exc:  # noqa: BLE001 - never fail a conversion here
            log.warning("companion generation failed for %s: %s", root, exc)
            report.failed.append(("companion files", str(exc)))

    if write_citation_file:
        try:
            from .citation import citation_path, write_citation

            if citation_path(root).exists():
                report.skipped.append(
                    "CITATION.cff already exists and was left alone"
                )
            else:
                report.citation_written = write_citation(root) is not None
        except Exception as exc:  # noqa: BLE001
            log.warning("citation write failed for %s: %s", root, exc)
            report.failed.append(("CITATION.cff", str(exc)))

    return report


__all__ = ["FixupReport", "run_dataset_fixups"]
