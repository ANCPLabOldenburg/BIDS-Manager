"""ECAT PET scanner.

ECAT is the native format of Siemens HRRT and older CTI/ECAT scanners. dcm2niix
cannot read it, so PET support that stopped at DICOM would leave those scanners
out. nibabel reads ECAT and is already a BIDS Manager dependency, so this costs
nothing extra.

The module mirrors :mod:`bidsmgr.inventory.eeg_meg`: probe each candidate file,
emit one inventory row per recording, and let the shared orchestrator merge the
rows into the unified TSV.

Detection is by the ``MATRIX7x`` magic at byte 0, not by the ``.v`` extension.
``.v`` is far too generic to trust on its own (Verilog, GNU V, patch files all
use it), and conversely a valid ECAT file is sometimes shipped without it.

Qt-free.
"""

from __future__ import annotations

import logging
import os
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd

log = logging.getLogger(__name__)

# ECAT7 main headers begin with this signature. The trailing digit is the
# sub-version (70, 72, 73 all appear in the OpenNeuroPET phantom set).
ECAT_MAGIC = b"MATRIX7"

# Candidate extensions, used only to skip an open() on obviously unrelated
# files. Anything extensionless still gets its magic checked.
ECAT_EXTS = (".v", ".img", ".ecat")


@dataclass(frozen=True)
class EcatProbe:
    """What one ECAT file tells us before any user input."""

    source: Path
    n_frames: int
    shape: tuple[int, ...]
    isotope: str = ""
    half_life: Optional[float] = None
    dose: Optional[float] = None
    tracer: str = ""
    facility: str = ""
    scan_start: Optional[int] = None
    frame_durations: tuple[float, ...] = ()
    frame_starts: tuple[float, ...] = ()
    patient_id: str = ""


def is_ecat_file(path: str | os.PathLike) -> bool:
    """True if ``path`` starts with the ECAT7 ``MATRIX7x`` signature."""
    p = Path(path)
    if p.suffix and p.suffix.lower() not in ECAT_EXTS:
        return False
    try:
        with open(p, "rb") as fh:
            return fh.read(len(ECAT_MAGIC)) == ECAT_MAGIC
    except OSError:
        return False


def _decode(value) -> str:
    """ECAT header strings are null-padded bytes inside a numpy record.

    nibabel hands them back as 0-d numpy arrays or numpy byte scalars, not as
    plain ``bytes``, so unwrap with ``.item()`` first. Skipping that step leaves
    the numpy repr (``np.bytes_(b'F-18')``) in the cell.
    """
    if value is None:
        return ""
    if hasattr(value, "item"):
        try:
            value = value.item()
        except (ValueError, AttributeError):
            pass
    if isinstance(value, bytes):
        return value.split(b"\0")[0].decode("latin-1", "replace").strip()
    return str(value).strip()


def _number(value) -> Optional[float]:
    """Unwrap a numpy scalar to a plain float, or ``None`` if it is not one."""
    if value is None:
        return None
    if hasattr(value, "item"):
        try:
            value = value.item()
        except (ValueError, AttributeError):
            return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def probe_ecat(path: Path) -> Optional[EcatProbe]:
    """Read one ECAT file's headers. Returns ``None`` if it cannot be read."""
    try:
        import nibabel
    except ImportError:  # pragma: no cover - nibabel is a hard dependency
        log.warning("nibabel unavailable; cannot probe ECAT %s", path)
        return None

    try:
        img = nibabel.ecat.load(str(path))
        mh = img.header
        subheaders = img.get_subheaders()
    except Exception as exc:  # noqa: BLE001 - a bad file must not kill the scan
        log.warning("could not read ECAT %s: %s", path, exc)
        return None

    def _main(key: str):
        try:
            return mh[key]
        except (KeyError, ValueError, TypeError):
            return None

    def _sub(key: str) -> tuple[float, ...]:
        out = []
        for sh in subheaders.subheaders:
            try:
                out.append(float(sh[key]))
            except (KeyError, ValueError, TypeError):
                out.append(0.0)
        return tuple(out)

    isotope = _decode(_main("isotope_name") or "")
    tracer = _decode(_main("radiopharmaceutical") or "")
    if tracer.lower() in ("unknown", "none"):
        tracer = ""

    dose = _number(_main("dosage"))
    half_life = _number(_main("isotope_halflife"))
    scan_start = _number(_main("scan_start_time"))

    # ECAT frame durations and start times are milliseconds; BIDS wants seconds.
    durations = tuple(v / 1000.0 for v in _sub("frame_duration"))
    starts = tuple(v / 1000.0 for v in _sub("frame_start_time"))

    return EcatProbe(
        source=path,
        n_frames=len(subheaders.subheaders),
        shape=tuple(img.shape),
        isotope=isotope,
        half_life=half_life,
        dose=dose,
        tracer=tracer,
        facility=_decode(_main("facility_name") or ""),
        scan_start=int(scan_start) if scan_start is not None else None,
        frame_durations=durations,
        frame_starts=starts,
        patient_id=_decode(_main("patient_id") or ""),
    )


def find_ecat_files(root: str | os.PathLike) -> list[Path]:
    """Every ECAT file under ``root``, detected by signature."""
    out: list[Path] = []
    for dirpath, _dirnames, filenames in os.walk(root):
        for name in filenames:
            fp = Path(dirpath) / name
            if is_ecat_file(fp):
                out.append(fp)
    return sorted(out)


def _subject_from_path(path: Path, root: Path) -> str:
    """Best-effort subject label, mirroring the EEG/MEG path heuristics.

    A literal ``sub-XXX`` component wins; otherwise the parent folder name,
    which is how the phantom archives are laid out (one folder per scanner).

    The parent folder, not the top one. This used to take the FIRST component
    of the relative path, which is the same folder for every file in a nested
    archive: all three ECAT phantoms landed on one subject with one basename,
    collided in staging, and none of them converted. The failure was quiet,
    because a collision reads as an ordinary per-task warning.
    """
    parts = path.relative_to(root).parts
    for part in parts:
        if part.lower().startswith("sub-"):
            return part[4:]
    return path.parent.name if len(parts) > 1 else path.stem


def scan_ecat(
    root_dir: str | os.PathLike,
    *,
    cancel_check=None,
) -> pd.DataFrame:
    """Inventory every ECAT file under ``root_dir``.

    Returns a DataFrame in the unified inventory's shape (the orchestrator
    backfills the columns this scanner does not own). One row per file, which
    for ECAT is also one row per recording: unlike DICOM, a whole dynamic study
    with all its frames lives in a single file.
    """
    from ..util.cancel import is_cancelled

    import json

    from .. import schema as schema_mod

    root = Path(root_dir)
    rows: list[dict] = []
    # One BIDS label per distinct source subject, numbered in encounter order.
    # Mirrors the EEG/MEG scanner: the label is provisional and the user
    # reconciles it by editing BIDS_name in the inventory.
    bids_id_for_subject: dict[str, str] = {}

    for idx, fp in enumerate(find_ecat_files(root)):
        if (idx & 7) == 0 and is_cancelled(cancel_check):
            from ..util.cancel import OperationCancelled
            raise OperationCancelled("scan cancelled by user")

        probe = probe_ecat(fp)
        if probe is None:
            continue

        rel = fp.relative_to(root)
        subject = _subject_from_path(fp, root)
        if subject not in bids_id_for_subject:
            bids_id_for_subject[subject] = f"sub-{len(bids_id_for_subject) + 1:03d}"
        bids_name = bids_id_for_subject[subject]

        entities = {"subject": bids_name[len("sub-"):]}
        try:
            basename = schema_mod.build_basename(entities, "pet", "pet")
        except Exception as exc:  # noqa: BLE001 - fall back to a literal name
            log.debug("schema.build_basename failed for %s: %s", bids_name, exc)
            basename = f"{bids_name}_pet"

        rows.append({
            "subject": subject,
            "BIDS_name": bids_name,
            "proposed_datatype": "pet",
            "proposed_basename": basename,
            "Proposed BIDS name": basename,
            "entities": json.dumps(entities, sort_keys=True),
            "source_folder": str(rel.parent) if rel.parent != Path(".") else root.name,
            "source_file": str(rel),
            "format": "ECAT",
            "sequence": fp.stem,
            "include": 1,
            "n_files": 1,
            "modality": "pet",
            "modality_bids": "pet",
            "bids_guess_datatype": "pet",
            "bids_guess_suffix": "pet",
            "bids_guess_classifier": "ecat_header",
            "bids_guess_confidence": 0.90,
            "bids_guess_skip": False,
            "PatientID": probe.patient_id,
            # The PET suggestion columns this scanner can fill from the header.
            "radionuclide_suggestion": _isotope_to_bids(probe.isotope),
            "tracer_suggestion": probe.tracer,
            # No unit is appended on purpose. The ECAT "dosage" field has no
            # unit declared in the header and vendors disagree on whether it is
            # mCi or MBq, so asserting one would be a guess dressed up as a
            # fact. The user confirms the unit in the metadata step.
            "injected_dose_suggestion": (
                f"{round(probe.dose, 4)}" if probe.dose else ""
            ),
            "_ecat_frames": probe.n_frames,
            "_ecat_frame_durations": probe.frame_durations,
            "_ecat_frame_starts": probe.frame_starts,
            "_ecat_facility": probe.facility,
            "_ecat_scan_start": probe.scan_start,
        })

    return pd.DataFrame(rows)


# Keys pet2bids writes that describe the conversion or the array rather than the
# acquisition. Not BIDS fields, and not the user's business.
_NOT_METADATA: frozenset[str] = frozenset({
    "ConversionSoftware", "ConversionSoftwareVersion", "Filename",
    "ImageSize", "PixelDimensions",
})


# Words a reader writes when it does not know. They are placeholders, and a
# placeholder in a sidecar is a wrong answer rather than a missing one: it stops
# the form asking and it satisfies a validator that should have complained.
_NON_ANSWERS: frozenset[str] = frozenset({"unknown", "n/a", "na", "none", "null"})


def _is_answer(value) -> bool:
    """A blank is not an answer, and writing one is worse than writing nothing.

    pet2bids returns its full template with every field present, most of them
    empty. An empty string in a sidecar reads as "we know it is nothing" and
    hides the field from the form that would otherwise ask for it.
    """
    if value is None:
        return False
    if isinstance(value, str):
        return value.strip() != "" and value.strip().lower() not in _NON_ANSWERS
    if isinstance(value, (list, tuple, dict)):
        return len(value) > 0
    return True


# ECAT unit labels, in the spelling BIDS asks for. CMIXF capitalises the litre,
# so a header saying "Bq/ml" means "Bq/mL"; the quantity is identical and only
# the spelling differs. Longest first, because the labels share prefixes.
_ECAT_UNITS: tuple[tuple[str, str], ...] = (
    ("ecat counts/sec", "counts/s"),
    ("counts/sec", "counts/s"),
    ("counts/s", "counts/s"),
    ("kbq/ml", "kBq/mL"),
    ("kbq/cc", "kBq/mL"),
    ("mbq/ml", "MBq/mL"),
    ("mbq/cc", "MBq/mL"),
    ("nci/ml", "nCi/mL"),
    ("nci/cc", "nCi/mL"),
    ("uci/ml", "uCi/mL"),
    ("uci/cc", "uCi/mL"),
    ("bq/ml", "Bq/mL"),
    ("bq/cc", "Bq/mL"),
)


def _units_from_ecat(header: dict) -> str:
    """The image units, recovered from the ECAT header's label field.

    BIDS REQUIRES ``Units`` on a PET image and the ECAT header does record it,
    but not cleanly: ``DATA_UNITS`` is a fixed-width character array, and a
    reader that does not stop at the terminator runs on into whatever the
    previous write left behind. A real HRRT file reads back as
    ``"Bq/mlounts/s"``, which is ``"Bq/ml"`` followed by the tail of an earlier
    ``"counts/s"``.

    So the label is matched as a PREFIX against the units ECAT actually uses,
    rather than compared whole. An unrecognised label yields nothing: a
    required field left visibly missing is recoverable, and a units string
    invented by us is a quantitative error nobody would catch.
    """
    raw = str(header.get("DATA_UNITS", "") or "").strip().lower()
    if not raw:
        return ""
    for label, bids in _ECAT_UNITS:
        if raw.startswith(label):
            return bids
    log.info("ECAT units label %r not recognised; Units left for the form", raw)
    return ""


# Guards the module-global template described in _isolated_template. Held
# across the whole read, because that is how long the global is in use.
_TEMPLATE_LOCK = threading.Lock()

# A clean copy of pet2bids' sidecar template, taken before any of their code has
# had a chance to write into it.
_PRISTINE_TEMPLATE: Optional[dict] = None


def _isolated_template(pet2bids_sidecar) -> None:
    """Give this read its own sidecar template, because it does not get one.

    ``Ecat.__init__`` assigns ``self.sidecar_template = sidecar.sidecar_template_full``,
    which is a REFERENCE to a module-level dict, not a copy. Every Ecat object
    in a process therefore shares one template, and ``populate_sidecar``
    appends to the lists inside it.

    The consequence is not cosmetic. Converting two ECAT files in one process
    gives the second one ``FrameTimesStart`` of length two: its own frame plus
    the first file's. Frame timing is what PET quantification is built on, so a
    batch conversion would silently produce wrong sidecars that get worse the
    further down the list you go. ``TimeZero`` sticks at the first file's value
    for the same reason, which is how this was noticed: three phantom scans
    from three sites, acquired years apart, all reported the same second.

    So before each read the global is replaced with a fresh deep copy of a
    pristine one. The snapshot is taken on first use, which is safe because
    this function is the only place BIDS Manager constructs an ``Ecat``.

    Reported upstream. Removing this becomes possible once each instance owns
    its template; until then it is load-bearing.
    """
    global _PRISTINE_TEMPLATE
    import copy

    if _PRISTINE_TEMPLATE is None:
        _PRISTINE_TEMPLATE = copy.deepcopy(pet2bids_sidecar.sidecar_template_full)
    pet2bids_sidecar.sidecar_template_full = copy.deepcopy(_PRISTINE_TEMPLATE)


def ecat_sidecar_fields(path: Path) -> dict:
    """What the ECAT header states, read by pet2bids.

    ECAT7 headers and subheaders carry roughly sixty fields: the scanner model,
    the reconstruction method with its iterations and subsets, per-frame decay
    and scale factors, the scan start, and the acquisition time BIDS wants as
    ``TimeZero``. Reading them properly is a solved problem, solved by the group
    that wrote this part of the standard, so this delegates to
    ``pypet2bids.ecat.Ecat`` rather than parsing the binary again.

    What is ours is the filtering. Only names the SCHEMA declares for a PET
    sidecar survive, because which fields exist is a fact about BIDS and facts
    about BIDS have one source. Empty values are dropped, since a blank field
    is worse than an absent one: it answers a question nobody asked and hides
    it from the form.

    Returns ``{}`` for anything unreadable. Enrichment is a bonus and must
    never cost a user their conversion.
    """
    from ..util.pet2bids import quiet_pet2bids, telemetry_off

    telemetry_off()
    try:
        from pypet2bids import sidecar as pet2bids_sidecar
        from pypet2bids.ecat import Ecat
    except Exception as exc:  # noqa: BLE001
        log.info(
            "ECAT enrichment skipped for %s: pypet2bids is unavailable (%s)",
            Path(path).name, exc,
        )
        return {}

    try:
        # collect_pixel_data is required: one derived field is computed from the
        # array. It costs about a tenth of a second on a real HRRT scan.
        #
        # Serialised and re-isolated: see _isolated_template. Conversion runs
        # several files at once, and the state being guarded is global.
        with _TEMPLATE_LOCK, quiet_pet2bids():
            _isolated_template(pet2bids_sidecar)
            reader = Ecat(ecat_file=str(path), collect_pixel_data=True)
            reader.populate_sidecar()
            reader.prune_sidecar()
            raw = dict(reader.sidecar_template or {})
            header = dict(reader.ecat_header or {})
    except Exception as exc:  # noqa: BLE001 - a bonus, never a failure
        log.warning("ECAT enrichment failed for %s: %s", Path(path).name, exc)
        return {}

    try:
        from .. import schema as schema_mod

        declared = {f.name for f in schema_mod.sidecar_fields("pet", "pet")}
    except Exception:  # noqa: BLE001
        declared = set()

    kept = {
        name: value
        for name, value in raw.items()
        if name not in _NOT_METADATA
        and _is_answer(value)
        and (not declared or name in declared)
    }

    # They spell the radionuclide as the header does, "F-18"; BIDS spells it
    # "F18" and has a vocabulary for it. The same normaliser the DICOM side uses.
    isotope = kept.get("TracerRadionuclide")
    if isinstance(isotope, str) and isotope:
        kept["TracerRadionuclide"] = _isotope_to_bids(isotope)

    # They leave Units empty; the header states it, and BIDS requires it.
    if not kept.get("Units"):
        units = _units_from_ecat(header)
        if units:
            kept["Units"] = units

    # A scan that started before the epoch did not start then. One phantom file
    # in the OpenNeuroPET set carries a negative SCAN_START_TIME, which renders
    # as a plausible-looking time of day in 1936, and TimeZero is what every
    # frame time and every blood sample is measured against. A wrong one is
    # silently wrong; an absent one is a question the form asks.
    start = header.get("SCAN_START_TIME")
    if isinstance(start, (int, float)) and start <= 0:
        dropped = [n for n in ("TimeZero", "ScanStart", "InjectionStart") if n in kept]
        for name in dropped:
            kept.pop(name, None)
        if dropped:
            log.warning(
                "%s records a scan start of %s, which is before the epoch; "
                "%s left for the form rather than derived from it",
                Path(path).name, start, ", ".join(dropped),
            )
    return kept


def _isotope_to_bids(raw: str) -> str:
    """``"F-18"`` to ``"F18"``, reusing the DICOM-side normaliser."""
    from .pet import normalise_radionuclide

    return normalise_radionuclide(raw)


__all__ = [
    "ECAT_EXTS",
    "ECAT_MAGIC",
    "EcatProbe",
    "ecat_sidecar_fields",
    "find_ecat_files",
    "is_ecat_file",
    "probe_ecat",
    "scan_ecat",
]
