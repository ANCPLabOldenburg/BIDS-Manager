"""Everything the MRI quality checks can be told: which method makes the
masks, and every threshold, for the BOLD time series, anatomical images
and diffusion.

One model for every front end: the viewer and the Editor read it from the
viewer settings (Settings > Quality control), ``bidsmgr-qc --config``
reads it from a JSON file, and each result records the configuration it
was made with. The defaults are the values the checks were built and
measured with; this module is their ONE definition.

Each field carries what the Settings page needs to draw it, as data (the
same hints as ``bidsmgr.viz.settings``): a title, a help text, and in
``json_schema_extra`` the range, step, unit and labels. Percentages are
stored as the user reads them (40, not 0.4).

Pure data, Qt-free. Validation clamps rather than rejects (as the viewer
settings do), so a hand-edited or older file still loads: a number outside
its range is brought inside it, an unknown choice falls back to the
default.
"""

from __future__ import annotations

import typing
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class _Section(BaseModel):
    model_config = ConfigDict(extra="ignore", validate_assignment=True)

    @model_validator(mode="before")
    @classmethod
    def _lenient(cls, data: Any) -> Any:
        """Numbers clamped into their range, unknown choices dropped (the
        default takes their place)."""
        if not isinstance(data, dict):
            return data
        out = dict(data)
        for name, info in cls.model_fields.items():
            if name not in out:
                continue
            value = out[name]
            if typing.get_origin(info.annotation) is Literal:
                if value not in typing.get_args(info.annotation):
                    del out[name]
                continue
            extra = info.json_schema_extra or {}
            if "range" in extra and isinstance(value, (int, float)) \
                    and not isinstance(value, bool):
                lo, hi = extra["range"]
                value = min(max(value, lo), hi)
                out[name] = int(round(value)) if info.annotation is int else float(value)
        return out


def _hint(**extra: Any) -> dict:
    return {"json_schema_extra": extra}


def _num(default: float, title: str, description: str, lo: float, hi: float, *,
         step: float = 1.0, unit: str = "", zero: str = "", enabled_by: str = "") -> Any:
    """A number field with its range (clamped on load, drawn on the page);
    ``enabled_by`` names the switch it depends on."""
    hints = {"range": (lo, hi), "step": step}
    if enabled_by:
        hints["enabled_by"] = enabled_by
    if unit:
        hints["unit"] = unit
    if zero:
        hints["zero"] = zero
    return Field(default, title=title, description=description, **_hint(**hints))


class QcMethods(_Section):
    """Which method makes the masks, the tissue classes and the registration."""

    model_config = ConfigDict(extra="ignore", validate_assignment=True, title="Methods")

    brain: Literal["mindgrab", "template"] = Field(
        "mindgrab", title="Brain mask (anatomical)",
        description="mindgrab: brainchop's network, any contrast, about 12 s. Template: "
                    "the template's brain registered to the image, seconds faster and "
                    "approximate (it follows the registration, not the image).",
        **_hint(labels={"mindgrab": "mindgrab network (brainchop)",
                        "template": "Registered template (fast, approximate)"}))
    tissues: Literal["robust_tissue", "em"] = Field(
        "robust_tissue", title="Tissue classes",
        description="robust_tissue: brainchop's grey and white matter network (T1w, T2w, "
                    "FLAIR). EM: three Gaussian classes with the template's tissue maps "
                    "as priors and a bias field (T1w and T2w only).",
        **_hint(labels={"robust_tissue": "robust_tissue network (brainchop)",
                        "em": "EM with template priors (fast)"}))
    registration: Literal["niimath", "numpy"] = Field(
        "niimath", title="Registration to the template",
        description="niimath: AFNI's 3dAllineate (affine), the brain against the "
                    "template's brain when the brain mask is mindgrab's. Outline and "
                    "intensity: BIDS Manager's own affine fit, no external tool.",
        **_hint(labels={"niimath": "niimath -allineate (affine)",
                        "numpy": "Outline and intensity (no external tool)"}))
    dwi_brain: Literal["mindgrab", "median_otsu"] = Field(
        "mindgrab", title="Brain mask (diffusion)",
        description="mindgrab on the mean b=0, or a median filter and Otsu's threshold "
                    "(as dipy's median_otsu): faster, and looser at the skull base.",
        **_hint(labels={"mindgrab": "mindgrab network (brainchop)",
                        "median_otsu": "Median filter and Otsu (fast)"}))
    bold_motion: Literal["confounds", "estimate"] = Field(
        "confounds", title="Head motion of a BOLD run",
        description="Read it from fMRIPrep's confounds table when the run has one (the "
                    "same numbers the analysis will use), else estimate it here. Or always "
                    "estimate it here.",
        **_hint(labels={"confounds": "fMRIPrep's confounds when there, else estimated",
                        "estimate": "Always estimated here"}))
    tool_timeout_s: int = _num(
        600, "Time limit for a network or niimath",
        "A network that runs longer is stopped and the check falls back to the fast "
        "methods. The first run of a network downloads it.", 60, 3600, step=60, unit="s")


class QcBold(_Section):
    """The QC plots under a BOLD run's time course and its quality maps."""

    model_config = ConfigDict(extra="ignore", validate_assignment=True,
                              title="Functional time series (BOLD)")

    skip_nonsteady: bool = Field(
        True, title="Leave out non-steady-state volumes",
        description="Volumes at the start that are far brighter than the rest (dummy "
                    "scans the scanner kept) are left out of every measure; left in, "
                    "they dominate the statistics.")
    nonsteady_z: float = _num(
        5.0, "Non-steady-state above",
        "A starting volume is non-steady-state when its mean is this many robust "
        "standard deviations above the run's median.", 2.0, 20.0, step=0.5,
        unit="robust SD", enabled_by="skip_nonsteady")
    fd_radius_mm: float = _num(
        50.0, "Head radius for FD",
        "Rotations are turned into millimetres as arcs on a sphere of this radius "
        "(Power 2012: 50 mm, about the distance from the centre of the head to the "
        "cortex).", 30.0, 100.0, step=5.0, unit="mm")
    fd_threshold_mm: float = _num(
        0.5, "FD worth a look above",
        "Volumes whose framewise displacement is above this are marked (fMRIPrep's "
        "and Power 2012's default; stricter studies use 0.2 to 0.3 mm).", 0.05, 5.0,
        step=0.05, unit="mm")
    motion_grid_mm: float = _num(
        3.5, "Motion estimated on a grid of",
        "Finer images are averaged down to about this voxel size before the head's "
        "motion is estimated: faster, and as accurate for whole-head motion.", 2.0, 8.0,
        step=0.5, unit="mm")
    dvars_fence_iqr: float = _num(
        1.5, "DVARS fence",
        "Volumes whose DVARS is above the 75th percentile plus this many interquartile "
        "ranges are marked (1.5 is the box plot's fence, FSL's default).", 0.5, 5.0,
        step=0.1, unit="IQR")
    outlier_limit_pct: float = _num(
        5.0, "Outlier voxels worth a look above",
        "Volumes with more than this share of outlier voxels are marked (afni_proc.py's "
        "censoring default).", 0.5, 50.0, step=0.5, unit="%")
    spike_z: float = _num(
        6.0, "Slice spike above",
        "A slice is a spike when its mean is this many robust standard deviations from "
        "its own course.", 2.0, 20.0, step=0.5, unit="robust SD")
    sample_voxels: int = _num(
        8000, "Voxels sampled",
        "The outlier, spike and carpet plots read this many voxels of the head: more is "
        "smoother and slower.", 1000, 100_000, step=1000)
    detrend: Literal[0, 1, 2] = Field(
        2, title="Trend removed from the maps",
        description="The standard deviation and temporal SNR maps are taken around a "
                    "polynomial trend of this order: slow scanner drift is not "
                    "instability (AFNI and MRIQC detrend too).",
        **_hint(labels={0: "None", 1: "Linear", 2: "Quadratic"}))
    tsnr_low: float = _num(
        20.0, "Low temporal SNR below",
        "The temporal SNR map's summary reports the share of the head below this.",
        1.0, 200.0, step=1.0)


class QcAnat(_Section):
    """The quality check of an anatomical image (T1w, T2w, FLAIR, PDw, T2starw)."""

    model_config = ConfigDict(extra="ignore", validate_assignment=True,
                              title="Anatomical images")

    work_mm: float = _num(
        2.0, "Working grid",
        "Masks, registration and tissue classes are made on a grid of about this voxel "
        "size; the measures that depend on noise go back to the image's own voxels.",
        1.0, 4.0, step=0.5, unit="mm")
    defaced_face_pct: float = _num(
        40.0, "Defaced when the face is blank over",
        "With no defacing in the sidecar, the image is taken as defaced when more than "
        "this share of the face region is exactly zero. A defaced image's air is not "
        "measured.", 5.0, 95.0, step=5.0, unit="%")
    stripped_shell_pct: float = _num(
        20.0, "Skull stripped when the shell holds under",
        "The image holds the brain only when less than this share of the shell 2 to 8 mm "
        "outside the brain holds signal (measured: 1 % stripped, 73 to 89 % intact).",
        1.0, 80.0, step=1.0, unit="%")
    zero_air_pct: float = _num(
        50.0, "Air set to zero above",
        "When more than this share of the air is exactly zero, the scanner or a tool "
        "blanked the background and no noise is left to measure.", 5.0, 95.0, step=5.0,
        unit="%")
    slab_mm: float = _num(
        120.0, "A slab below",
        "A field of view thinner than this covers part of the brain on purpose: the part "
        "outside is reported as coverage, not as a cut.", 40.0, 250.0, step=5.0,
        unit="mm")
    no_registration_mm: float = _num(
        70.0, "Not registered below",
        "Thinner than this, the template is not registered at all: a whole head cannot "
        "be placed in a few centimetres of it.", 20.0, 150.0, step=5.0, unit="mm")
    min_registration_dice: float = _num(
        0.8, "Registration trusted above",
        "The registered template's brain must overlap the brain found (Dice) at least "
        "this much, or the registration is reported as uncertain.", 0.5, 0.99,
        step=0.01, unit="Dice")
    fov_warning_pct: float = _num(
        0.5, "Field of view cuts the brain above",
        "A warning when more than this share of the template's brain falls outside the "
        "image.", 0.0, 10.0, step=0.1, unit="%")
    fov_error_pct: float = _num(
        3.0, "Field of view: error above",
        "An error when more than this share of the brain falls outside the image.",
        0.0, 30.0, step=0.5, unit="%")
    partial_volume_p: float = _num(
        0.9, "Partial volume below",
        "A brain voxel counts as partial volume when no tissue class holds it with at "
        "least this probability.", 0.5, 0.99, step=0.01)
    air_artefact_pct: float = _num(
        0.5, "Structured air worth a look above",
        "A warning when more than this share of the air above the face holds signal far "
        "above its noise (QI1).", 0.01, 10.0, step=0.05, unit="%")
    saturation_pct: float = _num(
        0.5, "Saturation worth a look above",
        "A warning when more than this share of the head sits at the image's maximum.",
        0.01, 10.0, step=0.05, unit="%")


class QcDwi(_Section):
    """The quality check of a diffusion series."""

    model_config = ConfigDict(extra="ignore", validate_assignment=True, title="Diffusion")

    b0_max: float = _num(
        50.0, "b=0 up to",
        "Volumes with a b-value at or below this are b=0 volumes (scanners write small "
        "values such as 5 for them).", 0.0, 200.0, step=5.0, unit="s/mm²")
    dti_max_b: float = _num(
        1500.0, "Tensor fitted up to",
        "The diffusion tensor is fitted on b-values up to this: above it the signal is "
        "no longer Gaussian enough for a tensor.", 500.0, 5000.0, step=100.0,
        unit="s/mm²")
    check_flips: bool = Field(
        True, title="Look for flipped or swapped b-vectors",
        description="Fits the tensor with the stored table and with every flip and swap "
                    "of its axes, and reports a table whose principal directions are "
                    "clearly more continuous from voxel to voxel than the stored one's. "
                    "Experimental; adds a few seconds.")
    flip_margin_pct: float = _num(
        2.0, "Another table must be more coherent by",
        "How much more coherent a flipped or swapped table must be before it is "
        "reported.", 0.5, 20.0, step=0.5, unit="%", enabled_by="check_flips")
    duplicate_deg: float = _num(
        2.0, "Repeated directions within",
        "Two directions closer than this (either sign) are reported as repeats.",
        0.5, 10.0, step=0.5, unit="degrees")
    fd_radius_mm: float = _num(
        50.0, "Head radius for displacement",
        "Rotations are turned into millimetres as arcs on a sphere of this radius, as for "
        "framewise displacement.", 30.0, 100.0, step=5.0, unit="mm")
    motion_b0_mm: float = _num(
        1.0, "Head motion worth a look above",
        "A finding when the head moved more than this across the b=0 volumes.",
        0.1, 10.0, step=0.1, unit="mm")
    far_mm: float = _num(
        2.0, "A volume far out of place beyond",
        "A volume this far from the first b=0, and an outlier of its shell (robust z "
        "above 5), moved during its acquisition.", 0.5, 10.0, step=0.5, unit="mm")
    lost_mm: float = _num(
        30.0, "Registration lost beyond",
        "A volume registered further than this from the first b=0 is reported as not "
        "placed, not as moved: a head does not move that far between two volumes.",
        5.0, 100.0, step=5.0, unit="mm")
    lost_deg: float = _num(
        15.0, "Registration lost beyond a turn of",
        "As above, for rotation.", 2.0, 45.0, step=1.0, unit="degrees")
    eddy_explained_pct: float = _num(
        30.0, "Eddy currents when the gradient explains",
        "Shifts are taken as eddy currents (and out of the motion) only when the "
        "gradient direction explains at least this share of them.", 5.0, 90.0, step=5.0,
        unit="%")
    dropout_z: float = _num(
        5.0, "Dropout below",
        "A slice is a dropout when its signal is this many robust spreads below what the "
        "tensor predicts, against the same slice in the shell's other volumes, and lost "
        "at least the share below.",
        2.0, 20.0, step=0.5, unit="robust SD")
    dropout_loss_pct: float = _num(
        10.0, "Dropout loses at least",
        "A dropout slice is also at least this much below its prediction.", 1.0, 80.0,
        step=1.0, unit="%")
    edge_slice_pct: float = _num(
        25.0, "Slices judged from",
        "A slice is judged only when it holds at least this share of a typical slice's "
        "brain: the top slices' slivers read as dropouts otherwise.", 5.0, 80.0, step=5.0,
        unit="%")
    interleave_z: float = _num(
        5.0, "Interleave artefact above",
        "A volume whose odd and even slices disagree this many robust spreads beyond "
        "the rest of its shell.", 2.0, 20.0, step=0.5, unit="robust SD")
    spike_z: float = _num(
        8.0, "Spiking voxel above",
        "A voxel this many robust spreads above the tensor's prediction, against the same "
        "voxel in the shell's other volumes.", 3.0, 30.0, step=0.5, unit="robust SD")
    drift_pct: float = _num(
        5.0, "Signal drift worth a look above",
        "A warning when the b=0 signal changes by more than this over the scan.", 0.5,
        50.0, step=0.5, unit="%")


class QcConfig(_Section):
    """The whole configuration, as the checks take it."""

    methods: QcMethods = Field(default_factory=QcMethods)
    bold: QcBold = Field(default_factory=QcBold)
    anat: QcAnat = Field(default_factory=QcAnat)
    dwi: QcDwi = Field(default_factory=QcDwi)


#: The defaults, for the functions whose keyword defaults are these values.
DEFAULT = QcConfig()


def fast(config: QcConfig) -> QcConfig:
    """``config`` with every method switched to the fast one that needs no
    external tool (``bidsmgr-qc --fast``, the Editor's Fast masks)."""
    out = config.model_copy(deep=True)
    out.methods.brain = "template"
    out.methods.tissues = "em"
    out.methods.registration = "numpy"
    out.methods.dwi_brain = "median_otsu"
    return out


def key(config: QcConfig) -> str:
    """A short fingerprint: a cached result is reused only for the same
    configuration."""
    import hashlib

    return hashlib.sha1(config.model_dump_json().encode()).hexdigest()[:12]


__all__ = ["DEFAULT", "QcAnat", "QcBold", "QcConfig", "QcDwi", "QcMethods", "fast", "key"]
