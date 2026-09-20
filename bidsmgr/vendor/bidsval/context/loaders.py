"""Read file content into the shapes the schema's checks expect.

Each loader degrades safely: a missing optional dependency, an unreadable file,
or a parse error yields an empty / ``None`` result rather than raising, so a
malformed dataset never crashes a run (the rule that needed the content simply
does not fire).

JSON is always available. TSV columns need ``pandas`` and NIfTI headers need
``nibabel`` (both in the ``content`` extra); without them, those tiers are
skipped.
"""

from __future__ import annotations

import json
from typing import Any

from ..files import BIDSFile


def load_json(bids_file: BIDSFile) -> dict[str, Any]:
    """Parse a JSON file to a dict, or ``{}`` if it cannot be read/parsed."""
    data, _error = load_json_checked(bids_file)
    return data


def load_json_checked(bids_file: BIDSFile) -> tuple[dict[str, Any], str | None]:
    """Parse a JSON file, returning ``(data, error_code)``.

    ``error_code`` is ``None`` on success; otherwise it is one of the BIDS issue
    codes describing why the file could not be used (``INVALID_FILE_ENCODING``,
    ``JSON_INVALID``, ``JSON_NOT_AN_OBJECT``, ``FILE_READ``), so a caller can both
    skip the unusable data and report a precise, actionable finding. ``data`` is
    ``{}`` whenever ``error_code`` is set.
    """
    try:
        raw = bids_file.read_bytes()
    except OSError:
        return {}, "FILE_READ"
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        return {}, "INVALID_FILE_ENCODING"
    try:
        data = json.loads(text)
    except ValueError:
        return {}, "JSON_INVALID"
    if not isinstance(data, dict):
        return {}, "JSON_NOT_AN_OBJECT"
    return data, None


def load_columns(bids_file: BIDSFile, max_rows: int = 1000) -> dict[str, list[str]]:
    """Read a TSV file as ``{column: [values]}`` of strings.

    Values are read as text (BIDS tabular data is text; numeric coercion happens
    in the expression functions). Returns ``{}`` if pandas is unavailable or the
    file cannot be read.
    """
    try:
        import pandas as pd
    except ImportError:
        return {}
    try:
        nrows = None if max_rows < 0 else max_rows
        frame = pd.read_csv(
            bids_file.abspath,
            sep="\t",
            dtype=str,
            keep_default_na=False,
            nrows=nrows,
        )
    except Exception:
        return {}
    return {str(col): frame[col].tolist() for col in frame.columns}


def load_nifti_header(bids_file: BIDSFile) -> dict[str, Any] | None:
    """Read a NIfTI header into the fields the schema references.

    Returns ``None`` if nibabel is unavailable or the file cannot be read, so
    checks selecting on ``nifti_header != null`` are skipped rather than failing.
    """
    try:
        import nibabel as nib
    except ImportError:
        return None
    try:
        image = nib.load(str(bids_file.abspath), mmap=False)
        header = image.header
    except Exception:
        return None

    def _ints(key: str) -> list[int]:
        try:
            return [int(x) for x in header[key]]
        except Exception:
            return []

    def _floats(key: str) -> list[float]:
        try:
            return [float(x) for x in header[key]]
        except Exception:
            return []

    try:
        xyz_unit, t_unit = header.get_xyzt_units()
    except Exception:
        xyz_unit, t_unit = "unknown", "unknown"

    # Orientation codes. A value (rather than null) means the affine is usable;
    # the schema's AMBIGUOUS_AFFINE check fires only when these are null.
    try:
        axis_codes: list[str] | None = list(nib.aff2axcodes(image.affine))
    except Exception:
        axis_codes = None

    return {
        "dim": _ints("dim"),
        "pixdim": _floats("pixdim"),
        "shape": list(getattr(image, "shape", []) or []),
        "voxel_sizes": (
            [float(z) for z in header.get_zooms()] if hasattr(header, "get_zooms") else []
        ),
        "qform_code": int(header["qform_code"]) if "qform_code" in header.keys() else None,
        "sform_code": int(header["sform_code"]) if "sform_code" in header.keys() else None,
        "xyzt_units": {"xyz": xyz_unit, "t": t_unit},
        "axis_codes": axis_codes,
    }
