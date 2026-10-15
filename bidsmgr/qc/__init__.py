"""Fast quality checks of anatomical and diffusion MRI.

The check runs the same on Windows, Linux and macOS with nothing to install
beside BIDS Manager: no ANTs, AFNI, FSL, FreeSurfer, MRtrix or PyTorch. The
brain, the tissue classes and the registration to the template come from
the tools BIDS Manager already ships for defacing (brainchop's networks,
niimath's affine registration; ``tools``), run in subprocesses. Where they
cannot run (no niimath wheel for the Python in use, a model not yet
downloaded and no network), numpy equivalents good enough to screen a
dataset take over and the result says so (``register``, ``masks``,
``bias``, ``segment``). Every metric is computed with numpy and scipy.

The metric DEFINITIONS follow the literature MRIQC also builds on, and each
says when it equals an MRIQC measure (``Metric.mriqc``). Where MRIQC's code
does something else than its definition (its QI1 is always 0, its diffusion
drift is multiplied instead of divided out) we follow the definition; the
plan records each case: workspace
``ancp_development_context_mds/active/ANAT_DWI_FAST_QC_PLAN.md``.

Qt-free. Three front ends share it: the viewer's Check quality, the Editor's
Quality check over a dataset, and ``bidsmgr-qc``.
"""

from __future__ import annotations

from .types import Finding, Metric, QCMap, QCResult

__all__ = ["Finding", "Metric", "QCMap", "QCResult"]
