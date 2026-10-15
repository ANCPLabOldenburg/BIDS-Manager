"""Quality results on disk: one JSON per image and a table per suffix, in a
BIDS derivative dataset (``derivatives/bidsmgr-qc/``).

The JSON sits at the image's own path inside the derivative, named as the
image is (``sub-01/anat/sub-01_T1w.json``), as MRIQC names its own; the
group table ``group_T1w.tsv`` holds one row per image and one column per
measure. The viewer and the Editor read both back, so a dataset checked
once shows its numbers, and where each image sits among the others, without
computing anything again.

Where an image sits among the others is a ROBUST z within the images of the
same suffix and the same ``acq``: a protocol is only comparable with itself,
and one bad image must not move the yardstick it is measured with.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np

from . import stats
from .types import Finding, Metric, QCResult

PIPELINE = "bidsmgr-qc"
#: A value this many robust SDs from its group is flagged.
GROUP_Z = 3.0


def _plain(value: Any) -> Any:
    """JSON-safe: arrays to lists, numpy scalars to Python, NaN to None."""
    if isinstance(value, np.ndarray):
        return [_plain(v) for v in value.tolist()]
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (np.floating, float)):
        f = float(value)
        return f if np.isfinite(f) else None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def to_json(result: QCResult, *, root: Optional[Path] = None) -> dict:
    """The result as JSON (the maps are left out: they are the viewer's)."""
    from .. import __version__

    rel = result.path
    if root is not None and result.path:
        try:
            rel = Path(result.path).resolve().relative_to(Path(root).resolve()).as_posix()
        except ValueError:
            rel = Path(result.path).name
    return {
        "path": rel,
        "kind": result.kind,
        "suffix": result.suffix,
        "metrics": {m.key: m.value for m in result.metrics},
        "metric_info": {m.key: {"title": m.title, "unit": m.unit, "group": m.group,
                                "better": m.better, "mriqc": m.mriqc, "help": m.help,
                                "fmt": m.fmt, "why_missing": m.why_missing}
                        for m in result.metrics},
        "findings": [{"key": f.key, "title": f.title, "level": f.level,
                      "message": f.message, "evidence": f.evidence}
                     for f in result.findings],
        "facts": _plain(result.facts),
        "tracks": _plain(result.tracks),
        "generated_by": {"name": "BIDS Manager", "version": __version__},
    }


def from_json(data: dict) -> QCResult:
    """A result read back (no maps)."""
    info = data.get("metric_info", {})
    metrics = []
    for key, value in (data.get("metrics") or {}).items():
        meta = info.get(key, {})
        metrics.append(Metric(key=key, title=meta.get("title", key), value=value,
                              unit=meta.get("unit", ""), group=meta.get("group", ""),
                              help=meta.get("help", ""), better=meta.get("better", ""),
                              mriqc=meta.get("mriqc", ""), fmt=meta.get("fmt", "{:.3g}"),
                              why_missing=meta.get("why_missing", "")))
    findings = [Finding(f["key"], f["title"], f["level"], f["message"], f.get("evidence"))
                for f in data.get("findings", [])]
    tracks = {}
    for key, t in (data.get("tracks") or {}).items():
        t = dict(t)
        if "ys" in t:
            t["ys"] = [np.asarray([np.nan if v is None else v for v in y], dtype=float)
                       for y in t["ys"]]
        if "image" in t:
            t["image"] = np.asarray([[np.nan if v is None else v for v in row]
                                     for row in t["image"]], dtype=np.float32)
        tracks[key] = t
    return QCResult(path=data.get("path", ""), kind=data.get("kind", ""),
                    suffix=data.get("suffix", ""), metrics=metrics, findings=findings,
                    tracks=tracks, facts=data.get("facts", {}))


def derivative_root(root: Path) -> Path:
    return Path(root) / "derivatives" / PIPELINE


def json_path(root: Path, image: Path) -> Path:
    """Where the result of ``image`` (inside ``root``) is written."""
    from ..viz import bids as VB

    image = Path(image)
    try:
        rel = image.resolve().relative_to(Path(root).resolve())
    except ValueError:
        rel = Path(image.name)
    return derivative_root(root) / rel.parent / (VB.stem_of(image) + ".json")


def write_description(root: Path) -> Path:
    """The derivative's ``dataset_description.json`` (written once)."""
    from .. import __version__
    from ..deface.derivatives import dataset_description

    path = derivative_root(root) / "dataset_description.json"
    if not path.is_file():
        doc = dataset_description(root, pipeline=PIPELINE, version=__version__)
        source = doc.get("SourceDatasets", [{}])[0].get("Name", "")
        doc["Name"] = f"{source} (quality check)" if source else "Quality check"
        doc["GeneratedBy"][0]["Description"] = (
            "Fast quality check of anatomical and diffusion images "
            "(bidsmgr.qc): image quality measures, BIDS-aware checks.")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
    return path


def save(result: QCResult, root: Path) -> Path:
    """Write ``result`` into the dataset's QC derivative; returns the file."""
    write_description(root)
    path = json_path(root, Path(result.path))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_json(result, root=root), indent=1) + "\n", encoding="utf-8")
    return path


def load(root: Path, image: Path) -> Optional[QCResult]:
    """The saved result of ``image``, or None."""
    path = json_path(root, image)
    if not path.is_file():
        return None
    try:
        return from_json(json.loads(path.read_text(encoding="utf-8")))
    except (OSError, ValueError, KeyError, TypeError):
        return None


def load_all(root: Path) -> list[dict]:
    """Every saved result in the dataset's QC derivative, as JSON."""
    base = derivative_root(root)
    out = []
    if not base.is_dir():
        return out
    for path in sorted(base.rglob("sub-*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if isinstance(data, dict) and "metrics" in data:
            out.append(data)
    return out


def group_key(data: dict) -> tuple[str, str]:
    """(suffix, acq): what an image is comparable with."""
    from ..viz import bids as VB

    name = Path(str(data.get("path", ""))).name
    return str(data.get("suffix", "")), VB.parse_entities(name).get("acq", "")


def group_z(rows: Iterable[dict]) -> dict[str, dict[str, float]]:
    """``{path: {metric: robust z}}`` within each (suffix, acq) group of
    three images or more."""
    rows = list(rows)
    groups: dict[tuple, list[dict]] = {}
    for r in rows:
        groups.setdefault(group_key(r), []).append(r)
    out: dict[str, dict[str, float]] = {}
    for members in groups.values():
        if len(members) < 3:
            continue
        keys = sorted({k for r in members for k in r.get("metrics", {})})
        for key in keys:
            vals = np.array([np.nan if r["metrics"].get(key) is None else r["metrics"][key]
                             for r in members], dtype=float)
            if np.count_nonzero(np.isfinite(vals)) < 3:
                continue
            finite = vals[np.isfinite(vals)]
            med = float(np.median(finite))
            spread = stats.mad(finite)
            if not spread or not np.isfinite(spread):
                continue
            for r, v in zip(members, vals):
                if np.isfinite(v):
                    out.setdefault(str(r["path"]), {})[key] = (float(v) - med) / spread
    return out


def protocol_findings(rows: Iterable[dict], sidecars: dict[str, dict]) -> dict[str, list[dict]]:
    """Per image, the acquisition parameters that differ from the rest of its
    (suffix, acq) group: voxel size, TR, TE, TI, flip angle."""
    fields = ("RepetitionTime", "EchoTime", "InversionTime", "FlipAngle")
    rows = list(rows)
    groups: dict[tuple, list[dict]] = {}
    for r in rows:
        groups.setdefault(group_key(r), []).append(r)
    out: dict[str, list[dict]] = {}
    for members in groups.values():
        if len(members) < 3:
            continue

        def mode(values):
            vals = [v for v in values if v is not None]
            if not vals:
                return None
            uniq, counts = np.unique(np.asarray(vals, dtype=object).astype(str),
                                     return_counts=True)
            return uniq[int(np.argmax(counts))]

        zooms = {str(r["path"]): tuple(round(float(z), 2) for z in r.get("facts", {})
                                       .get("zooms", [])) for r in members}
        common_z = mode([str(z) for z in zooms.values()])
        for r in members:
            p = str(r["path"])
            notes = []
            if common_z is not None and str(zooms[p]) != common_z:
                notes.append(f"voxel size {zooms[p]} (most: {common_z})")
            side = sidecars.get(p, {})
            for f in fields:
                common = mode([sidecars.get(str(m["path"]), {}).get(f) for m in members])
                v = side.get(f)
                if common is not None and v is not None and str(v) != common:
                    notes.append(f"{f} {v} (most: {common})")
            if notes:
                out[p] = [{"key": "protocol", "title": "Protocol differs from the rest",
                           "level": "warning",
                           "message": "Differs from the other images of its kind: "
                                      + "; ".join(notes) + "."}]
    return out


def write_group(root: Path, rows: Optional[list[dict]] = None) -> list[Path]:
    """``group_<suffix>.tsv`` per suffix, one row per image: the measures,
    then how many lie beyond :data:`GROUP_Z` within the image's group."""
    rows = rows if rows is not None else load_all(root)
    base = derivative_root(root)
    zs = group_z(rows)
    by_suffix: dict[str, list[dict]] = {}
    for r in rows:
        by_suffix.setdefault(str(r.get("suffix", "")), []).append(r)
    written = []
    for suffix, members in sorted(by_suffix.items()):
        keys = []
        for r in members:
            for k in r.get("metrics", {}):
                if k not in keys:
                    keys.append(k)
        path = base / f"group_{suffix or 'unknown'}.tsv"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh, delimiter="\t", lineterminator="\n")
            w.writerow(["path", *keys, "outliers", "findings"])
            for r in sorted(members, key=lambda x: str(x.get("path", ""))):
                vals = ["n/a" if r["metrics"].get(k) is None else f"{r['metrics'][k]:.6g}"
                        for k in keys]
                z = zs.get(str(r["path"]), {})
                n_out = sum(1 for v in z.values() if abs(v) > GROUP_Z)
                n_find = sum(1 for f in r.get("findings", []) if f.get("level") in
                             ("warning", "error"))
                w.writerow([r["path"], *vals, n_out, n_find])
        written.append(path)
    return written


__all__ = ["GROUP_Z", "PIPELINE", "derivative_root", "from_json", "group_key", "group_z",
           "json_path", "load", "load_all", "protocol_findings", "save", "to_json",
           "write_description", "write_group"]
