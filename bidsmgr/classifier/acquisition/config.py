"""Everything the parameter-based classifier can be told: whether it names
the series or only reports, how sure it must be, how much each kind of
evidence counts, and the physics ranges that separate one contrast from
another.

One model for every front end: Settings > Classification, ``bidsmgr-scan
--classifier-config`` / ``bidsmgr-classify --config`` (JSON), and Python.
Every number the classifier tests is here; the evidence rules themselves
(which feature counts for which name, and how much) are ``rulebook.json``,
changed through ``rulebook``.
The defaults are the values the rulebook was built and measured with; this
module is their ONE definition. Each field carries what the Settings page
needs to draw it (title, help, range, step, unit), the same hints as
``qc/config.py``. Pure data, Qt-free. Validation clamps rather than rejects.
"""

from __future__ import annotations

import hashlib
from typing import Literal

from pydantic import ConfigDict, Field

from ...qc.config import _hint, _num, _Section


class ClsGeneral(_Section):
    """What the classifier does with what it finds."""

    model_config = ConfigDict(extra="ignore", validate_assignment=True,
                              title="Classification")

    mode: Literal["applied", "shadow", "off"] = Field(
        "applied", title="Use the acquisition parameters",
        description="Applied: the parameters name the series, with dcm2niix's guess and "
                    "the series name as second opinions. Shadow: the names stay as before "
                    "and the classifier only reports what it would do. Off: not run.",
        **_hint(labels={"applied": "To name the series", "shadow": "Only to report",
                        "off": "Not at all"}))
    apply_at: float = _num(
        0.80, "Name without asking above",
        "A name this sure (0 to 1) is used as it is. Below it, the row is named but "
        "marked to check.", 0.5, 0.99, step=0.01)
    check_at: float = _num(
        0.40, "Leave to the other opinions below",
        "Below this, the classifier does not decide: dcm2niix's guess or the series name "
        "does, and the row is marked to check.", 0.1, 0.9, step=0.01)


class ClsWeights(_Section):
    """How much each kind of evidence counts (1 is as built)."""

    model_config = ConfigDict(extra="ignore", validate_assignment=True,
                              title="Weight of the evidence")

    parameters: float = _num(
        1.0, "Acquisition parameters",
        "Echo, repetition and inversion times, flip angle, readout, volumes, diffusion "
        "weighting: what the scanner did.", 0.0, 3.0, step=0.1)
    names: float = _num(
        1.0, "Series name",
        "Words in the series description and protocol name (T1, FLAIR, bold, SBRef, a "
        "ReproIn name): what the operator called it.", 0.0, 3.0, step=0.1)
    context: float = _num(
        1.0, "Neighbouring series",
        "What the series next to it are: an SBRef before its run, a reversed-phase "
        "reference beside a run, the two inversions of an MP2RAGE.", 0.0, 3.0, step=0.1)


class ClsAnatomy(_Section):
    """The ranges that tell anatomical contrasts apart."""

    model_config = ConfigDict(extra="ignore", validate_assignment=True,
                              title="Anatomical contrasts")

    t1_ti_min: float = _num(
        0.3, "T1-weighted inversion from",
        "An inversion-recovery gradient echo (MPRAGE) inverts this long or longer before "
        "it reads.", 0.05, 2.0, step=0.05, unit="s")
    t1_ti_max: float = _num(
        1.5, "T1-weighted inversion up to", "... and no longer than this.", 0.5, 3.0,
        step=0.05, unit="s")
    flair_ti_min: float = _num(
        1.6, "FLAIR inversion from",
        "A spin echo inverting at least this long before it reads nulls the fluid: "
        "FLAIR.", 1.0, 4.0, step=0.1, unit="s")
    t2_te_min: float = _num(
        60.0, "T2-weighted echo from",
        "A spin echo with an echo time this long or longer is T2-weighted.", 30.0, 200.0,
        step=5.0, unit="ms")
    pd_te_max: float = _num(
        30.0, "Proton density echo up to",
        "A spin echo with a short echo time and a long repetition time is proton density "
        "weighted.", 5.0, 60.0, step=1.0, unit="ms")
    long_tr_min: float = _num(
        2.0, "Long repetition time from",
        "Repetition times this long or longer let the tissue recover (T2 and proton "
        "density weighting).", 1.0, 6.0, step=0.1, unit="s")
    t2star_te_min: float = _num(
        15.0, "T2*-weighted echo from",
        "A gradient echo this long or longer (and not an EPI) is T2*-weighted.", 5.0,
        60.0, step=1.0, unit="ms")
    mp2rage_tr_min: float = _num(
        3.5, "MP2RAGE repetition from",
        "An inversion-recovery 3-D gradient echo repeating this slowly or slower reads two "
        "inversions per repetition: an MP2RAGE. Faster, it is an MPRAGE (T1-weighted).",
        2.0, 10.0, step=0.1, unit="s")
    tse_etl_min: int = _num(
        2, "Turbo spin echo from",
        "A spin echo reading this many echoes per excitation or more (its echo train) is a "
        "turbo or fast spin echo, how T2-weighted, PD-weighted and FLAIR images are made.",
        2, 64, step=1, unit="echoes")
    high_res_max_mm: float = _num(
        1.3, "High-resolution 3-D up to",
        "A 3-D series with voxels this small or smaller in every direction is a structural "
        "image (and no localizer).", 0.3, 3.0, step=0.1, unit="mm")


class ClsGradientEcho(_Section):
    """The ranges that make a gradient echo T1-weighted, or an angiogram."""

    model_config = ConfigDict(extra="ignore", validate_assignment=True,
                              title="Gradient-echo contrasts")

    spoiled_3d_tr_max: float = _num(
        50.0, "3-D spoiled gradient echo, repetition up to",
        "A 3-D gradient echo repeating this fast or faster, with a small flip angle, is "
        "T1-weighted (FLASH, SPGR, a T1 VIBE).", 5.0, 200.0, step=1.0, unit="ms")
    spoiled_3d_flip_max: float = _num(
        30.0, "3-D spoiled gradient echo, flip angle up to",
        "... and a flip angle this small or smaller.", 2.0, 90.0, step=1.0, unit="°")
    spoiled_2d_te_max: float = _num(
        10.0, "2-D spoiled gradient echo, echo up to",
        "A 2-D gradient echo with an echo this short or shorter, a short repetition and a "
        "large flip angle is T1-weighted.", 1.0, 30.0, step=0.5, unit="ms")
    spoiled_2d_tr_max: float = _num(
        1.0, "2-D spoiled gradient echo, repetition up to",
        "... repeating this fast or faster ...", 0.05, 3.0, step=0.05, unit="s")
    spoiled_2d_flip_min: float = _num(
        20.0, "2-D spoiled gradient echo, flip angle from",
        "... with a flip angle this large or larger.", 5.0, 90.0, step=1.0, unit="°")
    mt_te_max: float = _num(
        10.0, "Magnetization transfer echo up to",
        "A gradient echo with magnetization transfer and an echo this short or shorter "
        "is a T1-weighted MT image, not a T2*-weighted one.", 2.0, 30.0, step=0.5,
        unit="ms")
    angio_tr_max: float = _num(
        50.0, "Time-of-flight angiography, repetition up to",
        "A flow-compensated 3-D gradient echo repeating this fast or faster, with a "
        "moderate flip angle, is a time-of-flight angiogram: inflowing blood is bright.",
        5.0, 150.0, step=1.0, unit="ms")
    angio_te_max: float = _num(
        10.0, "Time-of-flight angiography, echo up to", "... with an echo this short ...",
        1.0, 20.0, step=0.5, unit="ms")
    angio_flip_min: float = _num(
        10.0, "Time-of-flight angiography, flip angle from", "... a flip angle from ...",
        1.0, 60.0, step=1.0, unit="°")
    angio_flip_max: float = _num(
        40.0, "Time-of-flight angiography, flip angle up to", "... up to this.",
        5.0, 90.0, step=1.0, unit="°")


class ClsEchoes(_Section):
    """Echo counts and fieldmaps."""

    model_config = ConfigDict(extra="ignore", validate_assignment=True,
                              title="Echoes and fieldmaps")

    multi_echo_min: int = _num(
        3, "Multi-echo from",
        "A series read at this many echo times or more is multi-echo (MEGRE, MESE), not "
        "one contrast. Two echoes are a fieldmap's or a proton density and T2 pair.",
        2, 12, step=1, unit="echoes")
    fieldmap_te_max: float = _num(
        15.0, "Fieldmap first echo up to",
        "A two-echo gradient echo whose first echo is this short or shorter is a "
        "fieldmap's magnitude.", 2.0, 40.0, step=0.5, unit="ms")


class ClsSeries(_Section):
    """The counts that tell time series from references and diffusion."""

    model_config = ConfigDict(extra="ignore", validate_assignment=True,
                              title="Time series and diffusion")

    run_min_volumes: int = _num(
        10, "A run has at least",
        "An EPI series with this many volumes or more is a run (BOLD); fewer is a "
        "reference or a test.", 3, 100, step=1, unit="volumes")
    reference_max_volumes: int = _num(
        9, "A reference has at most",
        "An EPI series with this many volumes or fewer, phase-encoded opposite to a run, "
        "is that run's fieldmap.", 1, 30, step=1, unit="volumes")
    dwi_min_b: float = _num(
        50.0, "Diffusion weighting from",
        "A b-value above this is diffusion weighting.", 0.0, 500.0, step=10.0,
        unit="s/mm²")
    dwi_min_directions: int = _num(
        3, "A diffusion series has at least",
        "Distinct gradient directions in a diffusion series.", 1, 30, step=1,
        unit="directions")
    localizer_max_slices: int = _num(
        5, "A localizer has at most",
        "A non-EPI series with this many slices or fewer (in several orientations) is a "
        "localizer, left out.", 1, 20, step=1, unit="slices")
    bold_te_min: float = _num(
        15.0, "BOLD echo from",
        "A gradient-echo EPI time series with an echo this long or longer (and no longer "
        "than the next setting) has the T2* weighting of BOLD.", 5.0, 60.0, step=1.0,
        unit="ms")
    bold_te_max: float = _num(
        60.0, "BOLD echo up to", "... and no longer than this.", 20.0, 120.0, step=1.0,
        unit="ms")
    bold_tr_max: float = _num(
        4.0, "A BOLD run repeats at most every",
        "A time series repeating more slowly than this is unlikely to be BOLD.", 1.0, 20.0,
        step=0.1, unit="s")
    sbref_lookahead: int = _num(
        3, "An SBRef precedes its run by at most",
        "A one-volume EPI is the SBRef of a run with its geometry among the next this many "
        "series.", 1, 10, step=1, unit="series")


class ClsContrast(_Section):
    """A contrast agent, and the series made with one or with labelled blood."""

    model_config = ConfigDict(extra="ignore", validate_assignment=True,
                              title="Contrast agent and perfusion")

    entity: Literal["agent", "fixed", "none"] = Field(
        "agent", title="Name the contrast agent (ce-)",
        description="A series acquired after a contrast agent was injected carries BIDS's "
                    "ce- entity: named after the agent in its header (ce-multihance), a "
                    "fixed label, or not at all.",
        **_hint(labels={"agent": "After the agent in the header", "fixed": "With a fixed label",
                        "none": "Not at all"}))
    fixed_label: str = Field(
        "gadolinium", title="Fixed label",
        description="The ce- label when the agent is not named, or always when 'With a fixed "
                    "label' is chosen. Letters and digits only.")
    from_names: bool = Field(
        True, title="Also when only the series name says so",
        description="A series named post-contrast (post, +C, Gd, KM, contrast) whose header "
                    "names no agent gets the ce- entity too, marked to check.")
    dsc: Literal["exclude", "cbv"] = Field(
        "exclude", title="A contrast bolus series (DSC perfusion)",
        description="An EPI time series acquired during a contrast bolus measures perfusion "
                    "(dynamic susceptibility contrast). BIDS 1.11 has no raw suffix for it: "
                    "leave it out (keep it in sourcedata/), or name it func/cbv, BIDS's "
                    "contrast-agent blood-volume series.",
        **_hint(labels={"exclude": "Leave it out", "cbv": "Name it func/cbv"}))
    dsc_min_volumes: int = _num(
        20, "A bolus series has at least",
        "An EPI series with a contrast agent and this many volumes or more follows a bolus.",
        3, 300, step=1, unit="volumes")
    m0_max_volumes: int = _num(
        2, "An ASL M0 reference has at most",
        "An unlabelled ASL readout with this many volumes or fewer is the M0 reference.",
        1, 10, step=1, unit="volumes")


class ClassifierConfig(_Section):
    """The whole configuration, as the classifier takes it."""

    general: ClsGeneral = Field(default_factory=ClsGeneral)
    weights: ClsWeights = Field(default_factory=ClsWeights)
    anatomy: ClsAnatomy = Field(default_factory=ClsAnatomy)
    gradient_echo: ClsGradientEcho = Field(default_factory=ClsGradientEcho)
    echoes: ClsEchoes = Field(default_factory=ClsEchoes)
    series: ClsSeries = Field(default_factory=ClsSeries)
    contrast: ClsContrast = Field(default_factory=ClsContrast)
    #: Changes to the evidence rules, laid over the shipped ``rulebook.json``
    #: (``{"candidates": {name: {...}}}``: a candidate given replaces the
    #: shipped one with its evidence lines and weights, ``null`` removes it).
    rulebook: dict = Field(default_factory=dict)


DEFAULT = ClassifierConfig()


def key(config: ClassifierConfig) -> str:
    """A short fingerprint of a configuration."""
    return hashlib.sha1(config.model_dump_json().encode()).hexdigest()[:12]


def load_json(text: str) -> ClassifierConfig:
    """A configuration from JSON (unknown keys ignored, numbers clamped)."""
    import json

    return ClassifierConfig.model_validate(json.loads(text or "{}"))


__all__ = ["DEFAULT", "ClassifierConfig", "ClsAnatomy", "ClsContrast", "ClsEchoes",
           "ClsGeneral", "ClsGradientEcho", "ClsSeries", "ClsWeights", "key", "load_json"]
