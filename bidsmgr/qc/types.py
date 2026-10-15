"""What a quality check returns: metrics, findings, and the maps behind them.

Pure data. The JSON form (``report.to_json``) keeps the metrics and findings;
the maps (masks, segmentations, the bias field) are for the viewer, which
draws them over the image as the evidence for a number.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Optional

import numpy as np

Level = Literal["ok", "info", "warning", "error"]


@dataclass
class Metric:
    """One number, with what it takes to read it."""

    #: The key in the JSON and the dataset table (``snr_wm``).
    key: str
    #: What it is called on screen.
    title: str
    #: None when it was not computed; ``why_missing`` says why.
    value: Optional[float]
    unit: str = ""
    #: Section of the report: noise, artefacts, tissues, coverage, header,
    #: motion, gradients, diffusion.
    group: str = ""
    #: What it measures and how to read it, in a sentence or two.
    help: str = ""
    #: "higher" or "lower" is better, or "" when neither (a volume fraction).
    better: str = ""
    #: The MRIQC IQM with the SAME definition, else "". Values still differ
    #: through masks and segmentation; the name says the formula is shared.
    mriqc: str = ""
    why_missing: str = ""
    #: Formatting for the table ("{:.2f}").
    fmt: str = "{:.3g}"

    def text(self) -> str:
        if self.value is None:
            return "not computed"
        try:
            body = self.fmt.format(self.value)
        except (ValueError, TypeError):
            body = str(self.value)
        return f"{body} {self.unit}".strip()


@dataclass
class Finding:
    """A check with a verdict: a field of view that cuts the brain, a
    b-vector that is not a unit vector, a slice that dropped out."""

    key: str
    title: str
    level: Level
    message: str
    #: What the viewer shows as evidence: a map id (``"fov_cut"``), or
    #: ``{"volume": 12, "slice": 30}`` for a diffusion volume.
    evidence: Any = None


@dataclass
class QCMap:
    """A mask or map behind a number, to draw over the image. On its own
    grid (``affine``): the check works on a coarser grid than the image's,
    and the viewer places an overlay by world position anyway."""

    key: str
    title: str
    data: np.ndarray
    affine: np.ndarray
    #: "mask" (one colour), "labels" (classes 1..n), or "field" (a scalar map).
    kind: str = "mask"
    help: str = ""
    #: Colour map or colour name for the overlay.
    colour: str = "red"
    #: Class names for "labels", by value.
    labels: dict = field(default_factory=dict)


@dataclass
class QCResult:
    """The quality check of one image."""

    path: str
    #: "anat" or "dwi".
    kind: str
    suffix: str
    metrics: list[Metric] = field(default_factory=list)
    findings: list[Finding] = field(default_factory=list)
    #: Masks and maps behind the numbers, for overlays. Not written to JSON.
    maps: dict[str, QCMap] = field(default_factory=dict)
    #: Per-volume series (diffusion): {id: {"title", "unit", "values", ...}}.
    tracks: dict[str, dict] = field(default_factory=dict)
    #: Facts about the image and the run: shape, spacing, whether defaced,
    #: the engine's version, timings in seconds.
    facts: dict[str, Any] = field(default_factory=dict)

    def metric(self, key: str) -> Optional[Metric]:
        for m in self.metrics:
            if m.key == key:
                return m
        return None

    def value(self, key: str) -> Optional[float]:
        m = self.metric(key)
        return None if m is None else m.value

    @property
    def worst_level(self) -> Level:
        order = {"ok": 0, "info": 1, "warning": 2, "error": 3}
        worst: Level = "ok"
        for f in self.findings:
            if order[f.level] > order[worst]:
                worst = f.level
        return worst


__all__ = ["Finding", "Level", "Metric", "QCMap", "QCResult"]
