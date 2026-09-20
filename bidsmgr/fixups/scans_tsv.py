"""Update ``*_scans.tsv`` filename columns after fmap renames.

NOT a no-op, whatever an older version of this docstring said. Two things
write a scans table into the staging tree before this runs:

* **mne-bids**, for every EEG / MEG / iEEG recording it converts. It reads
  ``meas_date`` off the recording and writes a real ``acq_time``, which is
  the only place that value exists in the dataset.
* the metadata engine, afterwards, for everything (see
  ``metadata/engine._refresh_scans_tsv``).

So a fieldmap rename in phase 2 can land on a table that already has rows,
and the ``filename`` column has to follow the file. The rows themselves are
not touched: only the basename portion is substituted.

Reference: v0.2.5 ``post_conv_renamer.update_scans_tsv``
(BIDS-Manager/bids_manager/post_conv_renamer.py L419-466).
"""

from __future__ import annotations

import csv
import logging
from pathlib import Path

log = logging.getLogger(__name__)


def update_scans_tsv(
    subject_staging_dir: Path,
    rename_map: dict[Path, Path],
) -> int:
    """Rewrite the ``filename`` column of any ``*_scans.tsv`` under the staging dir.

    Parameters
    ----------
    subject_staging_dir
        Per-subject staging tree (``<bids_root>/.tmp_bidsmgr/sub-<id>/``).
    rename_map
        Output of :func:`bidsmgr.fixups.fieldmaps.apply_fieldmap_renames`.
        Maps ``old_path → new_path``. Both paths share the same parent
        (the rename happens in place); the ``filename`` column in
        ``*_scans.tsv`` records paths relative to the subject root, so
        we substitute the basename portion.

    Returns
    -------
    int
        Number of ``*_scans.tsv`` files actually rewritten. ``0`` is normal
        for an MRI-only subject with no fieldmap rename, and for any subject
        whose staged table names no renamed file.
    """
    if not subject_staging_dir.is_dir() or not rename_map:
        return 0

    name_substitutions: dict[str, str] = {
        old.name: new.name for old, new in rename_map.items()
    }

    n_rewritten = 0
    for scans_tsv in subject_staging_dir.rglob("*_scans.tsv"):
        try:
            with open(scans_tsv, newline="", encoding="utf-8") as fh:
                reader = csv.DictReader(fh, delimiter="\t")
                fieldnames = reader.fieldnames or []
                rows = list(reader)
        except OSError as exc:
            log.warning("could not read %s: %s", scans_tsv, exc)
            continue

        if "filename" not in fieldnames:
            continue

        changed = False
        for row in rows:
            current = row.get("filename", "")
            if not current:
                continue
            base = Path(current).name
            if base in name_substitutions:
                row["filename"] = current.replace(base, name_substitutions[base])
                changed = True

        if not changed:
            continue

        try:
            with open(scans_tsv, "w", newline="", encoding="utf-8") as fh:
                writer = csv.DictWriter(fh, fieldnames=fieldnames, delimiter="\t")
                writer.writeheader()
                writer.writerows(rows)
            n_rewritten += 1
            log.info("scans.tsv: rewrote %s", scans_tsv)
        except OSError as exc:
            log.warning("could not write %s: %s", scans_tsv, exc)

    return n_rewritten


__all__ = ["update_scans_tsv"]
