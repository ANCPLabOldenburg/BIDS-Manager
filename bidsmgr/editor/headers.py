"""What a recording's own file says about it, as sidecar fields.

A sidecar is mostly a copy of what was in the recording's header when it was
converted, plus what nobody's header records. When a sidecar is lost, or a
dataset arrives without any, the recording itself is still there, and much
of what the sidecar said can be read off it again. This module does that
reading, one reader per kind of file, and nothing else: it never looks at
the dataset around the file and never writes.

What each kind of file can give, and what it cannot:

* **NIfTI (MRI).** The header has room for the repetition time and the slice
  timing, and dcm2niix writes the echo time and the multiband factor into its
  description (``TE=30;Time=102012.133;phase=1;mb=4``). The repetition time is
  believed only where it is meant: in a 4-D image, or where the description
  shows dcm2niix wrote the header (other tools leave 1.0 in a 3-D image). The
  phase encoding direction is NOT read: the header knows the axis but not the
  sign, and a wrong sign silently ruins distortion correction.
* **NIfTI-MRS.** The acquisition parameters are in the header extension the
  standard defines, and are copied as the conversion copies them.
* **PET NIfTI.** Nothing the standard asks for is in the header.
* **EEG and MEG.** Read with MNE and described with MNE-BIDS, the library the
  conversion writes them with, so the fields and their spelling are the ones
  the conversion would have written. A format that stores no channel types
  (EDF, BrainVision, EEGLAB) takes them from the recording's
  ``*_channels.tsv``.
* **Physiological and eye-tracking tables.** A ``.tsv.gz`` has no header:
  its sampling rate, start time and column names exist only in its sidecar
  or in the data it was converted from.
"""

from __future__ import annotations

import logging
import re
import tempfile
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)

#: Datatypes whose NIfTI header is an MR image's.
MRI_DATATYPES = frozenset({"anat", "func", "dwi", "fmap", "perf"})

#: Extensions MNE reads, by the datatype they are recorded under in BIDS.
MEEG_EXTENSIONS = frozenset({
    ".fif", ".fif.gz", ".ds", ".con", ".sqd", ".kdf", ".pdf", ".edf", ".bdf",
    ".vhdr", ".set", ".gdf", ".cnt", ".mef", ".mefd", ".nwb", ".mff",
})

#: Formats that store channel types in the file itself; the rest take them
#: from ``*_channels.tsv``.
_TYPED_FORMATS = frozenset({".fif", ".fif.gz", ".ds", ".con", ".sqd", ".kdf", ".pdf"})

#: The NIfTI datatypes, in the order a suffix found in a name is looked up.
NIFTI_DATATYPES = ("anat", "func", "dwi", "fmap", "perf", "pet", "mrs")

#: What a recording in a MEG format may be recorded as.
MEG_DATATYPES = ("meg", "eeg")

#: What a recording in an EEG format may be recorded as.
EEG_DATATYPES = ("eeg",)

#: A readable file outside BIDS: these and the MNE formats. ``.pdf`` is a 4D
#: Neuroimaging file only inside a BIDS ``meg/`` folder; anywhere else it is a
#: document.
READABLE_EXTENSIONS = frozenset({".nii", ".nii.gz"}) | (MEEG_EXTENSIONS - {".pdf"})

#: The suffix assumed for a datatype when nothing names one.
_DEFAULT_SUFFIX = {"anat": "T1w", "func": "bold", "dwi": "dwi", "fmap": "epi",
                   "perf": "asl", "pet": "pet", "mrs": "svs", "meg": "meg", "eeg": "eeg"}

#: dcm2niix's NIfTI description: ``TE=<ms>;Time=<hhmmss>;...``.
_DCM2NIIX = re.compile(r"^TE=(?P<te>[0-9.eE+-]+);Time=")
_MULTIBAND = re.compile(r"(?:^|;)mb=(?P<mb>\d+)(?:;|$)")

_DOUBLE = (".nii.gz", ".tsv.gz", ".fif.gz")

#: Said wherever an echo time comes from dcm2niix's description.
ROUNDED_TE = ("The echo time is read from the image description, where dcm2niix "
               "writes it to two significant figures (2.07 ms reads as 2.1 ms).")


@dataclass
class HeaderRead:
    """What one file's header gave."""

    #: Field -> value, in the order a reader of the sidecar expects.
    fields: dict = field(default_factory=dict)
    #: What was read, for the person deciding whether to trust it
    #: ("the NIfTI header", "the EDF file, through MNE-BIDS").
    reader: str = ""
    #: Why little or nothing could be read, or what to know about what was.
    note: str = ""


def extension(name: str) -> str:
    """The extension of a BIDS file name, with the double ones whole."""
    lowered = name.lower()
    for double in _DOUBLE:
        if lowered.endswith(double):
            return double
    return Path(lowered).suffix


def read_header(path: Path, datatype: str, suffix: str) -> HeaderRead:
    """What ``path``'s own header says, as sidecar fields.

    Never raises for a file it cannot read: the reason goes into
    :attr:`HeaderRead.note`, because one unreadable file must not stop the
    rest of a dataset.
    """
    path = Path(path)
    ext = extension(path.name)
    try:
        if ext in (".nii", ".nii.gz"):
            if datatype == "mrs":
                return _mrs(path)
            return _nifti(path, datatype, suffix)
        if ext in MEEG_EXTENSIONS:
            return _meeg(path, datatype)
        if ext == ".tsv.gz":
            return HeaderRead(note=(
                "A table has no header: its sampling rate, start time and "
                "column names exist only in its sidecar or in the data it was "
                "converted from."))
    except Exception as exc:  # noqa: BLE001 - one bad file, reported
        log.debug("could not read the header of %s", path, exc_info=True)
        return HeaderRead(note=f"The header could not be read: {exc}")
    return HeaderRead(note="Nothing in this kind of file can be read as metadata.")


def datatypes_for(path: Path) -> tuple[str, ...]:
    """The datatypes a file of this format can be recorded as."""
    ext = extension(Path(path).name)
    if ext in (".nii", ".nii.gz"):
        return NIFTI_DATATYPES
    if ext in _TYPED_FORMATS:
        return MEG_DATATYPES
    if ext in MEEG_EXTENSIONS:
        return EEG_DATATYPES
    return ()


def default_suffix(datatype: str) -> str:
    """The suffix assumed for ``datatype`` when nothing names one."""
    return _DEFAULT_SUFFIX.get(datatype, "")


def guess_kind(path: Path) -> tuple[str, str, str]:
    """``(datatype, suffix, why)`` for a file whose name says nothing, read
    from the file: a spectroscopy header, a 4-D MR image, a PET image, a
    3-D MR image, a MEG or an EEG recording. A guess, and said to be one."""
    path = Path(path)
    ext = extension(path.name)
    if ext in (".nii", ".nii.gz"):
        try:
            import nibabel as nib

            header = nib.load(str(path)).header
        except Exception:  # noqa: BLE001 - unreadable: the commonest image
            return "anat", "T1w", "the image could not be read; taken as an anatomical image"
        if any(e.get_code() == 44 for e in header.extensions):
            return "mrs", "svs", "it carries a spectroscopy (NIfTI-MRS) header"
        descrip = _description(header)
        shape = header.get_data_shape()
        frames = shape[3] if len(shape) > 3 else 1
        if descrip.startswith("Time=") and not _DCM2NIIX.match(descrip):
            return "pet", "pet", "dcm2niix described it without an echo time, as it does PET"
        if frames > 1:
            return "func", "bold", f"a 4-D image of {frames} volumes, taken as a BOLD run"
        return "anat", "T1w", "a 3-D image, taken as an anatomical image"
    if ext in _TYPED_FORMATS:
        return "meg", "meg", f"a {ext.lstrip('.').upper()} recording, a MEG format"
    if ext in MEEG_EXTENSIONS:
        return "eeg", "eeg", f"a {ext.lstrip('.').upper()} recording, an EEG format"
    return "", "", "not a format BIDS Manager reads"


def echo_time(path: Path) -> Optional[float]:
    """The echo time dcm2niix wrote into a NIfTI's description, in seconds."""
    try:
        import nibabel as nib

        descrip = _description(nib.load(str(path)).header)
    except Exception:  # noqa: BLE001 - unreadable is "not known"
        return None
    m = _DCM2NIIX.match(descrip)
    return _seconds_from_ms(m.group("te")) if m else None


# ---------------------------------------------------------------------------
# NIfTI
# ---------------------------------------------------------------------------


def _description(header) -> str:
    raw = header["descrip"].tobytes().split(b"\x00", 1)[0]
    return raw.decode("latin-1").strip()


def _seconds_from_ms(text: str) -> Optional[float]:
    try:
        value = float(text)
    except ValueError:
        return None
    return round(value / 1000.0, 6) if value > 0 else None


_TIME_SCALE = {"sec": 1.0, "msec": 1e-3, "usec": 1e-6}


def _nifti(path: Path, datatype: str, suffix: str) -> HeaderRead:
    import nibabel as nib

    header = nib.load(str(path)).header
    if datatype not in MRI_DATATYPES:
        return HeaderRead(reader="the NIfTI header", note=(
            "A PET image's header holds none of the fields the standard asks "
            "for: the tracer, the dose and the frame times are only in the "
            "sidecar or the data it was converted from." if datatype == "pet"
            else "Nothing in this image's header is sidecar metadata."))

    out: dict = {}
    notes: list[str] = []
    descrip = _description(header)
    from_dcm2niix = _DCM2NIIX.match(descrip)
    shape = header.get_data_shape()
    four_d = len(shape) > 3 and shape[3] > 1

    # pixdim[4], read directly: a 3-D image reports no fourth zoom, and
    # dcm2niix writes the repetition time there all the same.
    unit = header.get_xyzt_units()[1]
    if four_d or from_dcm2niix:
        scale = _TIME_SCALE.get(unit, 1.0 if from_dcm2niix else None)
        tr = float(header["pixdim"][4]) * scale if scale else 0.0
        if tr > 0:
            out["RepetitionTime"] = round(tr, 6)

    if from_dcm2niix:
        te = _seconds_from_ms(from_dcm2niix.group("te"))
        if te is not None and suffix != "phasediff":
            out["EchoTime"] = te
            notes.append(ROUNDED_TE)
        mb = _MULTIBAND.search(descrip)
        if mb and int(mb.group("mb")) > 1:
            out["MultibandAccelerationFactor"] = int(mb.group("mb"))

    slice_dim = header.get_dim_info()[2]
    if int(header["slice_code"]) and float(header["slice_duration"]) > 0 \
            and slice_dim is not None:
        try:
            times = header.get_slice_times()
        except Exception:  # noqa: BLE001 - incomplete slice fields
            times = None
        if times and all(t is not None for t in times):
            scale = _TIME_SCALE.get(unit, 1.0)
            out["SliceTiming"] = [round(float(t) * scale, 6) for t in times]
            out["SliceEncodingDirection"] = "ijk"[slice_dim]

    if not out:
        notes.append("The header holds no timing: it was written by a tool that "
                     "leaves those fields empty.")
    return HeaderRead(out, "the NIfTI header", " ".join(notes))


def _mrs(path: Path) -> HeaderRead:
    from ..fixups.mrs_header import read_mrs_header, sidecar_fields

    header = read_mrs_header(path)
    if header is None:
        return HeaderRead(reader="the NIfTI header", note=(
            "This spectrum carries no NIfTI-MRS header extension, which is "
            "where its acquisition parameters would be."))
    fields = {k: _float32_tidy(v) for k, v in sidecar_fields(header).items()}
    return HeaderRead(fields, "the NIfTI-MRS header extension")


def _float32_tidy(value):
    """A number as the single-precision value it was stored as: the header
    extension spells 123.262604 as 123.26260375976562."""
    if isinstance(value, list):
        return [_float32_tidy(v) for v in value]
    if isinstance(value, float):
        return float(f"{value:.9g}")
    return value


# ---------------------------------------------------------------------------
# EEG and MEG
# ---------------------------------------------------------------------------


def _meeg(path: Path, datatype: str) -> HeaderRead:
    import json

    import mne
    import mne_bids
    from mne_bids.config import MANUFACTURERS

    ext = extension(path.name)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        raw = None
        if ext not in _TYPED_FORMATS:
            # The channel types live in *_channels.tsv; MNE-BIDS applies them.
            try:
                raw = mne_bids.read_raw_bids(
                    mne_bids.get_bids_path_from_fname(path), verbose=False)
            except Exception:  # noqa: BLE001 - fall back to the file alone
                log.debug("read_raw_bids failed for %s", path, exc_info=True)
        if raw is None:
            raw = mne.io.read_raw(str(path), preload=False, verbose=False)

    task = _entity(path.name, "task") or ""
    manufacturer = MANUFACTURERS.get(ext, "n/a")
    reader = f"the {ext.lstrip('.').upper()} recording, through MNE-BIDS"
    try:
        from mne_bids.write import _sidecar_json

        with tempfile.TemporaryDirectory(prefix="bidsmgr-header-", ignore_cleanup_errors=True) as tmp:
            out = Path(tmp) / "sidecar.json"
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                _sidecar_json(raw, task, manufacturer, out, datatype)
            fields = json.loads(out.read_text(encoding="utf-8"))
        # A name without a task gives TaskName "": not an answer.
        return HeaderRead({k: v for k, v in fields.items() if v != ""}, reader)
    except Exception as exc:  # noqa: BLE001 - MNE-BIDS changed: the essentials
        log.debug("MNE-BIDS could not describe %s: %s", path, exc)
    fields: dict = {}
    if task:
        fields["TaskName"] = task
    fields["SamplingFrequency"] = float(raw.info["sfreq"])
    line = raw.info.get("line_freq")
    fields["PowerLineFrequency"] = float(line) if line else "n/a"
    fields["RecordingDuration"] = round(raw.n_times / raw.info["sfreq"], 6)
    return HeaderRead(fields, reader.replace(", through MNE-BIDS", ", through MNE"))


def _entity(name: str, key: str) -> Optional[str]:
    m = re.search(rf"(?:^|_){key}-([A-Za-z0-9]+)", name)
    return m.group(1) if m else None


__all__ = ["EEG_DATATYPES", "HeaderRead", "MEEG_EXTENSIONS", "MEG_DATATYPES", "MRI_DATATYPES",
           "NIFTI_DATATYPES", "READABLE_EXTENSIONS", "ROUNDED_TE", "datatypes_for",
           "default_suffix", "echo_time", "extension", "guess_kind", "read_header"]
